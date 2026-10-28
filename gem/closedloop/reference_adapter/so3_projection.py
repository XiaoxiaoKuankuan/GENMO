"""根姿态有限时域投影的数值求解与精确隐式微分分离。

数值前向在双精度中解与原始十帧提案最近的受约束问题，历史锚点和名义姿态均只读。
SO(3)约束只在内部留小幅安全余量，外部角速度/角加速度预算与certificate容差不变；
最终float32输出必须重新检查。SQP迭代不保留计算图，反向通过最终活跃约束的完整
非线性KKT系统计算，包含拉格朗日Hessian及历史锚点/名义旋转梯度，不是直通估计。
逐环境张量批量求解，不读取未来执行状态。不可恢复的数值或历史不可行通过携带失败
样本mask的异常交给采集器；调用方不得发送不满足证书的参考。
"""

import torch


class ProjectionNumericalFailure(RuntimeError):
    def __init__(self, message, bad_mask, certificate=None, diagnostic=None):
        super().__init__(message)
        self.bad_mask = bad_mask.detach()
        self.certificate = certificate or {}
        self.diagnostic = diagnostic or {}


def _pinverse_gram(a):
    gram = a.transpose(-1, -2) @ a
    eigen, basis = torch.linalg.eigh(gram)
    reciprocal = torch.where(eigen > 1e-9, eigen.clamp_min(1e-9).reciprocal(), 0.)
    return (basis * reciprocal[..., None, :]) @ basis.transpose(-1, -2)


def _constraints(projector, x, past, anchor, rotation, lower, upper, margin):
    base = torch.block_diag(projector.matrix, projector.matrix, projector.matrix).to(x)
    ax = x @ base.T
    nonlinear = torch.func.vmap(projector.angular_constraints)(x, past, anchor, rotation) + margin
    return torch.cat((ax-upper, lower-ax, nonlinear), -1)


def _jacobian(projector, x, past, anchor, rotation):
    base = torch.block_diag(projector.matrix, projector.matrix, projector.matrix).to(x)
    angular = torch.func.vmap(torch.func.jacrev(projector.angular_constraints, argnums=0))(x, past, anchor, rotation)
    return torch.cat((base.expand(len(x), -1, -1), -base.expand(len(x), -1, -1), angular), 1)


def _multipliers(a, c, target, x):
    active = c > -2e-6
    for _ in range(5):
        aa = a * active[..., None]
        pinv = _pinverse_gram(aa)
        multiplier = (aa @ (pinv @ (target-x)[..., None])).squeeze(-1)
        next_active = active & (multiplier > -1e-7)
        if bool((next_active == active).all()):
            break
        active = next_active
    multiplier = multiplier.clamp_min(0.) * active
    return multiplier, active


def _solve(projector, initial, target, past, anchor, rotation, lower, upper, margin):
    from .reference_constraints import _CoupledProjection
    x = initial.clone()
    base = torch.block_diag(projector.matrix, projector.matrix, projector.matrix).to(x).expand(len(x), -1, -1)
    fn = torch.func.vmap(projector.angular_constraints)
    jac_fn = torch.func.vmap(torch.func.jacrev(projector.angular_constraints, argnums=0))
    counts = torch.zeros(len(x), device=x.device, dtype=torch.long)
    finished = torch.zeros(len(x), device=x.device, dtype=torch.bool)
    step_error = x.new_full((len(x),), float('inf'))
    for iteration in range(60):
        value = fn(x, past, anchor, rotation) + margin
        jac = jac_fn(x, past, anchor, rotation)
        boundary = (jac @ x[..., None]).squeeze(-1) - value
        matrix = torch.cat((base, jac), 1)
        scale = matrix.norm(dim=-1).clamp_min(1.)
        with torch.profiler.record_function("projection.SO3_QP"):
            candidate = _CoupledProjection.apply(target,
                torch.cat((lower, torch.full_like(boundary, -1e4)), -1)/scale,
                torch.cat((upper, boundary), -1)/scale, matrix/scale[..., None], projector.coupled_iterations)
        step_error = (candidate-x).abs().amax(-1)
        converged = (value.amax(-1) <= 1e-8) & (step_error < 2e-6)
        finished |= converged
        if bool(finished.all()):
            break
        fractions = x.new_tensor([1., .5, .25, .125, .0625, .03125, .015625, 0.])
        trial = x[:, None] + fractions[None, :, None]*(candidate-x)[:, None]
        g = fn(trial.flatten(0, 1), past.repeat_interleave(len(fractions), 0),
               anchor.repeat_interleave(len(fractions), 0), rotation.repeat_interleave(len(fractions), 0)) + margin
        # 平滑L1可行性罚函数仅用于SQP步长，不对跨周期参考滤波。
        merit = .5*(trial-target[:, None]).square().sum(-1) + 1000*torch.relu(g).sum(-1).reshape(len(x), -1)
        choice = merit.argmin(-1)
        nxt = trial[torch.arange(len(x), device=x.device), choice]
        counts += (~finished).long()
        x = torch.where(finished[:, None], x, nxt)
    c = _constraints(projector, x, past, anchor, rotation, lower, upper, margin)
    a = _jacobian(projector, x, past, anchor, rotation)
    multiplier, active = _multipliers(a, c, target, x)
    stationarity = x-target+(a.transpose(-1, -2)@multiplier[..., None]).squeeze(-1)
    diagnostics = dict(sqp_iterations=counts, sqp_constraint_residual64=c.amax(-1),
                       sqp_stationarity_residual64=stationarity.abs().amax(-1),
                       sqp_step_residual64=step_error, sqp_active_constraints=active.sum(-1))
    return x, multiplier, active, diagnostics


class _ImplicitSO3(torch.autograd.Function):
    @staticmethod
    def forward(ctx, initial, target, past, anchor, rotation, lower, upper, margin, projector):
        with torch.profiler.record_function('projection.SO3_SQP_numeric_forward'):
            x, multiplier, active, diagnostic = _solve(projector, initial, target, past, anchor, rotation, lower, upper, margin)
        ctx.projector = projector
        ctx.save_for_backward(x, target, past, anchor, rotation, lower, upper, margin, multiplier, active)
        projector.last_statistics.update({k: v.detach() for k, v in diagnostic.items()})
        projector.last_solution64 = x.detach()
        return x

    @staticmethod
    def backward(ctx, gradient):
        x, target, past, anchor, rotation, lower, upper, margin, multiplier, active = ctx.saved_tensors
        projector = ctx.projector
        with torch.profiler.record_function('projection.SO3_KKT_implicit_backward'), torch.no_grad():
            # 最终非线性KKT导数含Hessian，不能用末次线性QP切空间近似替代。
            def lagrangian(value, old, current, nominal, lam):
                angular = projector.angular_constraints(value, old, current, nominal)
                return (angular*lam).sum()
            nlinear = lower.shape[-1]*2
            hessian_fn = torch.func.jacrev(torch.func.grad(lagrangian, argnums=0), argnums=0)
            hessian = torch.func.vmap(hessian_fn)(x, past, anchor, rotation, multiplier[:, nlinear:])
            eye = torch.eye(x.shape[-1], device=x.device, dtype=x.dtype)
            hessian = eye + hessian
            a = _jacobian(projector, x, past, anchor, rotation)*active[..., None]
            pinv = _pinverse_gram(a)
            tangent = eye-pinv@(a.transpose(-1, -2)@a)
            lhs = tangent@hessian@tangent + eye-tangent
            rhs = (tangent@gradient[..., None]).squeeze(-1)
            u = torch.linalg.solve_ex(lhs, rhs[..., None]).result.squeeze(-1)
            v = (a@(pinv@(gradient-(hessian@u[..., None]).squeeze(-1))[..., None])).squeeze(-1)
            # 对KKT残差作精确参数VJP，覆盖边界、历史与名义姿态依赖。
            with torch.enable_grad():
                values = [t.detach().requires_grad_(True) for t in (x, target, past, anchor, rotation, lower, upper)]
                xx, yy, pp, cc, rr, ll, hh = values
                constraints = _constraints(projector, xx, pp, cc, rr, ll, hh, margin)
                lag = (constraints*multiplier.detach()).sum()
                stationarity = xx-yy+torch.autograd.grad(lag, xx, create_graph=True)[0]
                objective = -(stationarity*u.detach()).sum()-(constraints*v.detach()).sum()
                gradients = torch.autograd.grad(objective, values[1:], allow_unused=True)
        return (None, *gradients, None, None)


def project_rotation_implicit(projector, proposal, projected, left, anchor, nominal_rotation, lower, upper):
    unit = projector.velocity[3:6]*.02
    x = (projected[..., 3:6]/unit).transpose(1, 2).flatten(1)
    fn = torch.func.vmap(projector.angular_constraints)
    with torch.no_grad():
        values = fn(x, left[:, 3:6], anchor[:, 3:6], nominal_rotation)
        selected = (values.amax(-1) > 0).nonzero().flatten()
    projector.last_statistics.update(sqp_active_count=selected.new_tensor(len(selected)),
        sqp_batch_count=selected.new_tensor(len(x)))
    if not len(selected):
        return projected
    initial = x[selected].double()
    target = (proposal[selected, :, 3:6].double()/unit.double()).transpose(1, 2).flatten(1)
    past, current = left[selected, 3:6].double(), anchor[selected, 3:6].double()
    rotation = nominal_rotation[selected].double()
    lo, hi = lower[selected, 3:6].double().flatten(1), upper[selected, 3:6].double().flatten(1)
    # 内部加速度余量0.01rad/s²用于两次20ms差分的FP32舍入；外部预算仍3。
    margin = values.new_full((values.shape[-1],), 1e-6, dtype=torch.double)
    margin[:11] = 1e-4/projector.rotation_velocity
    if projector.constrain_root_angular_acceleration:
        margin[11:22] = .01/projector.rotation_acceleration
    for attempt in range(3):
        solution = _ImplicitSO3.apply(initial, target, past, current, rotation, lo, hi, margin, projector)
        result = projected.clone()
        result[selected, :, 3:6] = (solution.reshape(-1, 3, 10).transpose(1, 2)*unit.double()).to(projected)
        with torch.no_grad():
            c64 = fn(solution, past, current, rotation)
            castx = (result[selected, :, 3:6]/unit).transpose(1, 2).flatten(1)
            castc = fn(castx, left[selected, 3:6], anchor[selected, 3:6], nominal_rotation[selected])
            projector.last_statistics.update(so3_float64_excess=c64.amax(-1),
                so3_float32_excess=castc.amax(-1), so3_resolve_attempts=selected.new_tensor(attempt+1))
            # 在实际输出精度下严格不超过预算，不使用certificate的容差通过内部重解。
            good = castc.amax(-1) <= 0
        if bool(good.all()):
            return result
        initial = solution.detach()
        margin = margin*2
    bad = torch.zeros(len(projected), device=projected.device, dtype=torch.bool)
    bad[selected[~good]] = True
    certificate = projector.certificate(result.detach(), left.detach(), anchor.detach(), nominal_rotation.detach())
    raise ProjectionNumericalFailure('SO(3)投影在实际输出精度下仍不可行；拒绝失败样本', bad, certificate, projector.last_statistics)
