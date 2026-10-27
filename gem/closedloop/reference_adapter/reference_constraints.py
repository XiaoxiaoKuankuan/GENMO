"""十帧参考的最小距离有限时域投影；不再以低通削弱全部网络提案。

输入为270维网络的独立有界提案，以及只读的前一帧/当前已提交残差。对每个通道
求解严格凸二次规划 min 0.5||d-proposal||²，约束幅值、附加速度、附加加速度，
包含当前锚点接缝。第11点仍是2*d10-d9，末段速度因此与第10段相等，附加加速度
为零；名义halo必须来自真实缓存。批处理ADMM定位活跃面，再解十维KKT投影；反向
使用该活跃面的隐式导数，对提案和历史锚点均保留梯度。没有逐环境Python循环。

根旋转还必须检查复合SO(3)后的真实角速度/角加速度，不能把旋转向量导数冒充它。
数值不收敛或SO(3)预算不满足时返回明确不可执行证据，调用方不得发送该动作。
零提案、零已提交历史严格保持零；非零历史下零提案仍须遵守接缝，不承诺数学上
不可能同时满足的“历史非零但未来无条件全零”。旧beta仅供失败基线对照。
"""

import torch
from supermimic.rotations import so3_exp, so3_log


class _EuclideanProjection(torch.autograd.Function):
    @staticmethod
    def forward(ctx, proposal, lower, upper, matrix, inverse, iterations):
        rho = 10.
        if proposal.is_cuda:
            from .projection_kernel import solve
            x, dual = solve(proposal, lower, upper, matrix, inverse, iterations)
        else:
            x = proposal
            ax = x@matrix.T
            z = ax.clamp(lower, upper)
            dual = torch.zeros_like(z)
            for _ in range(iterations):
                x = (proposal+rho*(z-dual)@matrix)@inverse
                ax = x@matrix.T
                z = (ax+dual).clamp(lower, upper)
                dual = dual+ax-z
        active = dual.abs() > 2e-5
        a = matrix*active[..., None]
        # 使用双精度小矩阵消除冗余活跃约束的秩判定误差；前向控制精度仍由输入决定。
        gram = a.transpose(-1, -2).double()@a.double()
        eigen, basis = torch.linalg.eigh(gram)
        reciprocal = torch.where(eigen > 1e-7, eigen.clamp_min(1e-7).reciprocal(), 0.)
        pinv = ((basis*reciprocal[..., None, :])@basis.transpose(-1, -2)).to(proposal)
        boundary = torch.where(dual > 0, upper, lower)
        rhs = ((proposal@matrix.T-boundary)*active)@matrix
        polished = proposal-(pinv@rhs[..., None]).squeeze(-1)
        p_ax = polished@matrix.T
        violation = torch.maximum(p_ax-upper, lower-p_ax).amax(-1)
        # 解在活跃面交点上且全约束可行时使用KKT结果，否则保留有证据的ADMM近似解。
        use_polish = violation <= 2e-5
        x = torch.where(use_polish[..., None], polished, x)
        identity = torch.eye(10, device=x.device, dtype=x.dtype)
        tangent = identity-pinv@(a.transpose(-1, -2)@a)
        ctx.save_for_backward(tangent, pinv, a, dual > 0)
        return x

    @staticmethod
    def backward(ctx, gradient):
        tangent, pinv, active_matrix, upper_side = ctx.saved_tensors
        proposal = (tangent@gradient[..., None]).squeeze(-1)
        boundary = (active_matrix@(pinv@gradient[..., None])).squeeze(-1)
        return proposal, boundary*(~upper_side), boundary*upper_side, None, None, None


class _CoupledProjection(torch.autograd.Function):
    """SO(3)线性化半空间QP；包含对约束矩阵的隐式导数，不能以直通梯度替代。"""
    @staticmethod
    def forward(ctx, proposal, lower, upper, matrix, iterations):
        # 近乎平行的SO(3)面使固定罚参数ADMM长期留下不可行残差；这里用双精度
        # 可行性恢复的原始-对偶内点法求同一严格凸QP，不改变目标或放宽半空间。
        # 全部环境批量30维线性代数；最多40个Newton步，反向仍为活跃面的隐式导数。
        a64, y = matrix.double(), proposal.double()
        g = torch.cat((a64, -a64), 1)
        h = torch.cat((upper.double(), -lower.double()), 1)
        eye64 = torch.eye(y.shape[-1], device=y.device, dtype=y.dtype)
        x64 = y.clone()
        slack = (h-(g@x64[..., None]).squeeze(-1)).abs().clamp_min(1.)
        multipliers = slack.reciprocal()

        def positive_step(value, direction):
            ratios = torch.where(direction < 0, -value/direction.clamp_max(-1e-100), torch.full_like(value, float('inf')))
            return ratios.amin(-1).clamp_max(1.)

        for step in range(iterations):
            dual_residual = x64-y+(g.transpose(-1, -2)@multipliers[..., None]).squeeze(-1)
            primal_residual = (g@x64[..., None]).squeeze(-1)+slack-h
            mu = (slack*multipliers).mean(-1)
            finished = ((torch.maximum(primal_residual.abs().amax(-1), dual_residual.abs().amax(-1)) < 1e-6)
                        & (mu < 1e-8)) if step >= 8 else torch.zeros_like(mu, dtype=torch.bool)
            if bool(finished.all()):
                break
            ratio = multipliers/slack
            lhs = eye64+g.transpose(-1, -2)@(ratio[..., None]*g)

            def newton(central):
                rhs = -dual_residual+(g.transpose(-1, -2)@((central-multipliers*primal_residual)/slack)[..., None]).squeeze(-1)
                dx = torch.linalg.solve_ex(lhs, rhs[..., None]).result.squeeze(-1)
                ds = -primal_residual-(g@dx[..., None]).squeeze(-1)
                dl = (-central-multipliers*ds)/slack
                return dx, ds, dl

            _, ds_aff, dl_aff = newton(slack*multipliers)
            ap, ad = positive_step(slack, ds_aff), positive_step(multipliers, dl_aff)
            mu_aff = ((slack+ap[:, None]*ds_aff)*(multipliers+ad[:, None]*dl_aff)).mean(-1)
            sigma = (mu_aff/mu.clamp_min(1e-30)).clamp(0, 1).pow(3)
            dx, ds, dl = newton(slack*multipliers+ds_aff*dl_aff-sigma[:, None]*mu[:, None])
            ap, ad = .995*positive_step(slack, ds), .995*positive_step(multipliers, dl)
            x64 = torch.where(finished[:, None], x64, x64+ap[:, None]*dx)
            slack = torch.where(finished[:, None], slack, (slack+ap[:, None]*ds).clamp_min(1e-30))
            multipliers = torch.where(finished[:, None], multipliers, (multipliers+ad[:, None]*dl).clamp_min(1e-30))
        x = x64.to(proposal)
        dual = (multipliers[:, :lower.shape[1]]-multipliers[:, lower.shape[1]:]).to(proposal)
        eye = eye64.to(proposal)
        active = dual.abs() > 2e-5
        a = matrix*active[..., None]
        gram = a.transpose(-1, -2).double()@a.double()
        eigen, basis = torch.linalg.eigh(gram)
        reciprocal = torch.where(eigen > 1e-7, eigen.clamp_min(1e-7).reciprocal(), 0.)
        pinv = ((basis*reciprocal[..., None, :])@basis.transpose(-1, -2)).to(proposal)
        boundary = torch.where(dual > 0, upper, lower)
        rhs = (a.transpose(-1, -2)@(((matrix@proposal[..., None]).squeeze(-1)-boundary)*active)[..., None])
        polished = proposal-(pinv@rhs).squeeze(-1)
        ax = (matrix@polished[..., None]).squeeze(-1)
        use = torch.maximum(ax-upper, lower-ax).amax(-1) < 2e-5
        x = torch.where(use[..., None], polished, x)
        tangent = eye-pinv@(a.transpose(-1, -2)@a)
        lagrange = (a@(pinv@(proposal-x)[..., None])).squeeze(-1)
        ctx.save_for_backward(tangent, pinv, a, dual > 0, x, lagrange)
        return x

    @staticmethod
    def backward(ctx, gradient):
        tangent, pinv, matrix, upper, x, lagrange = ctx.saved_tensors
        gy = (tangent@gradient[..., None]).squeeze(-1)
        gb = (matrix@(pinv@gradient[..., None])).squeeze(-1)
        ga = -lagrange[..., None]*gy[..., None, :]-gb[..., None]*x[..., None, :]
        return gy, gb*(~upper), gb*upper, ga, None


class ReferenceProjection(torch.nn.Module):
    """附加预算为算法候选范围，不是机器人硬件极限；SO(3)须另行实际检查。"""
    def __init__(self, spec, bounds, *, joint_velocity=1., joint_acceleration=10.,
                 ankle_velocity=.5, ankle_acceleration=5., height_velocity=.12,
                 height_acceleration=1.2, rotation_velocity=.3, rotation_acceleration=3.,
                 iterations=3000, coupled_iterations=40, tolerance=1e-6):
        super().__init__()
        self.iterations, self.tolerance = iterations, tolerance
        self.coupled_iterations = coupled_iterations
        self.rotation_velocity, self.rotation_acceleration = rotation_velocity, rotation_acceleration
        # 旋转三个分量分配欧氏球预算；最终角速度始终以SO(3)复合后结果核验。
        velocity = [.12, .12, height_velocity]+[rotation_velocity/(3**.5)]*3
        acceleration = [1.2, 1.2, height_acceleration]+[rotation_acceleration/(3**.5)]*3
        velocity += [ankle_velocity if 'ankle' in j else joint_velocity for j in spec.joint_names]
        acceleration += [ankle_acceleration if 'ankle' in j else joint_acceleration for j in spec.joint_names]
        dtype, device = bounds.dtype, bounds.device
        self.register_buffer('velocity', torch.tensor(velocity, dtype=dtype, device=device))
        self.register_buffer('acceleration', torch.tensor(acceleration, dtype=dtype, device=device))
        self.register_buffer('bounds', bounds.detach().clone())
        eye = torch.eye(10, device=device, dtype=dtype)
        difference = eye-torch.diag(torch.ones(9, device=device, dtype=dtype), -1)
        matrix = torch.cat((eye, difference, difference@difference))
        self.register_buffer('matrix', matrix)
        self.register_buffer('inverse', torch.linalg.inv(eye+10*matrix.T@matrix))

    def forward(self, proposal, left, anchor, nominal_rotation=None):
        unit = self.velocity*.02
        y = proposal.transpose(1, 2)/unit[None, :, None]
        l, c = left/unit, anchor/unit
        zero = c[..., None]*0
        v_boundary = torch.cat((c[..., None], zero.expand(-1, -1, 9)), -1)
        a_boundary = torch.cat(((2*c-l)[..., None], -c[..., None], zero.expand(-1, -1, 8)), -1)
        boundary = torch.cat((zero.expand(-1, -1, 10), v_boundary, a_boundary), -1)
        limits = torch.cat(((self.bounds/unit)[:, None].expand(-1, 10),
            torch.ones_like(self.bounds)[:, None].expand(-1, 10),
            (self.acceleration*.02/self.velocity)[:, None].expand(-1, 10)), -1)
        lower, upper = boundary-limits, boundary+limits
        # 已可行的提案原样通过，零严格为零，未激活约束时输入梯度严格为单位矩阵。
        feasible = ((y@self.matrix.T >= lower) & (y@self.matrix.T <= upper)).all(-1)
        indices = (~feasible).flatten().nonzero().flatten()
        if len(indices) == 0:
            result = proposal
        else:
            flat_y, flat_l, flat_u = y.reshape(-1, 10), lower.reshape(-1, 30), upper.reshape(-1, 30)
            solved = _EuclideanProjection.apply(flat_y[indices], flat_l[indices], flat_u[indices],
                                                self.matrix, self.inverse, self.iterations)
            x = flat_y.index_copy(0, indices, solved).reshape_as(y)
            projected = (x*unit[None, :, None]).transpose(1, 2)
            result = torch.where(feasible[:, None], proposal, projected)
        if nominal_rotation is not None:
            result = self.project_rotation(proposal, result, left, anchor, nominal_rotation, lower, upper)
        return result

    def angular_constraints(self, x, left, anchor, nominal_rotation):
        unit = self.velocity[3:6]*.02
        plan = x.reshape(3, 10).T*unit
        halo = 2*plan[-1:]-plan[-2:-1]
        d = torch.cat((left[None], anchor[None], plan, halo))
        corrected = nominal_rotation@so3_exp(d)
        w = so3_log(corrected[1:]@corrected[:-1].transpose(-1, -2))/.02
        nominal_w = so3_log(nominal_rotation[1:]@nominal_rotation[:-1].transpose(-1, -2))/.02
        dw = w-nominal_w
        return torch.cat((dw[1:].norm(dim=-1)/self.rotation_velocity-1,
            (dw.diff(dim=0)/.02).norm(dim=-1)/self.rotation_acceleration-1,
            plan.norm(dim=-1)/.05-1))

    def project_rotation(self, proposal, projected, left, anchor, nominal_rotation, lower, upper):
        unit = self.velocity[3:6]*.02
        x = (projected[..., 3:6]/unit).transpose(1, 2).flatten(1)
        fn = torch.func.vmap(self.angular_constraints)
        values = fn(x, left[:, 3:6], anchor[:, 3:6], nominal_rotation)
        selected = (values.amax(-1) > 1e-5).nonzero().flatten()
        if not len(selected):
            return projected
        # 角加速度要两次除以20ms；仅内部QP使用double仍不足，float32的SO(3)
        # 线性化/merit会在真实样本停滞。仅需修正的旋转子问题全程double，输出和
        # Actor仍是原float32；类型转换保留梯度，不降低最终执行证书精度。
        x = x[selected].double()
        target = (proposal[selected, :, 3:6].double()/unit.double()).transpose(1, 2).flatten(1)
        past, current, rotation = left[selected, 3:6].double(), anchor[selected, 3:6].double(), nominal_rotation[selected].double()
        base = torch.block_diag(self.matrix, self.matrix, self.matrix).double().expand(len(selected), -1, -1)
        lo, hi = lower[selected, 3:6].double().flatten(1), upper[selected, 3:6].double().flatten(1)
        jacobian = torch.func.vmap(torch.func.jacrev(self.angular_constraints, argnums=0))
        # SQP只修正真实SO(3)约束不满足的旋转通道；每次仍最小化与原提案的距离。
        for iteration in range(60):
            value = fn(x, past, current, rotation)
            finished = value.amax(-1) <= 1e-5 if iteration >= 4 else torch.zeros(len(x), device=x.device, dtype=torch.bool)
            if bool(finished.all()):
                break
            jac = jacobian(x, past, current, rotation)
            bound = (jac@x[..., None]).squeeze(-1)-value
            matrix = torch.cat((base, jac), 1)
            # 同一半空间按法向长度缩放，避免角加速度Jacobian把ADMM条件数放大；
            # 可行集合及最近点目标严格不变，缩放仍参与autograd。
            scale = matrix.norm(dim=-1).clamp_min(1.)
            candidate = _CoupledProjection.apply(target, torch.cat((lo, torch.full_like(bound, -1e4)), -1)/scale,
                torch.cat((hi, bound), -1)/scale, matrix/scale[..., None], self.coupled_iterations)
            # 非线性SQP采用本次优化内部的merit回溯，不对跨控制周期的参考做滤波。
            # 原线性约束在凸组合下保持可行，实际SO(3)约束逐候选重新计算。
            fractions = x.new_tensor([1., .5, .25, .125, .0625, .03125, .015625, 0.])
            trial = x[:, None]+fractions[None, :, None]*(candidate-x)[:, None]
            g = fn(trial.flatten(0, 1), past.repeat_interleave(len(fractions), 0),
                   current.repeat_interleave(len(fractions), 0), rotation.repeat_interleave(len(fractions), 0))
            merit = .5*(trial-target[:, None]).square().sum(-1)+1000*torch.relu(g).sum(-1).reshape(len(x), -1)
            chosen = merit.argmin(-1)
            x = torch.where(finished[:, None], x, trial[torch.arange(len(x), device=x.device), chosen])
        corrected = projected.clone()
        corrected[selected, :, 3:6] = (x.reshape(-1, 3, 10).transpose(1, 2)*unit).to(projected)
        return corrected

    def certificate(self, proposal, left, anchor, nominal_rotation):
        halo = 2*proposal[:, -1:]-proposal[:, -2:-1]
        d = torch.cat((left[:, None], anchor[:, None], proposal, halo), 1)
        velocity, acceleration = d.diff(dim=1)/.02, d.diff(dim=1).diff(dim=1)/.02**2
        excess_v = (velocity.abs()-self.velocity).amax((1, 2))
        excess_a = (acceleration.abs()-self.acceleration).amax((1, 2))
        corrected = nominal_rotation@so3_exp(d[..., 3:6])
        w = so3_log(corrected[:, 1:]@corrected[:, :-1].transpose(-1, -2))/.02
        nominal_w = so3_log(nominal_rotation[:, 1:]@nominal_rotation[:, :-1].transpose(-1, -2))/.02
        # 不限制GENMO本身合法速度；限制修正带来的真实世界角速度/角加速度偏离。
        dw = w-nominal_w
        excess_w = dw.norm(dim=-1).amax(-1)-self.rotation_velocity
        excess_alpha = (dw.diff(dim=1)/.02).norm(dim=-1).amax(-1)-self.rotation_acceleration
        excess_bound = (proposal.abs()-self.bounds).amax((1, 2))
        excess_rotation = proposal[..., 3:6].norm(dim=-1).amax(-1)-.05
        # 位移数值容差统一换算到导数，报告原量；不把不收敛解当可执行参考。
        good = ((excess_v <= self.tolerance/.02) & (excess_a <= self.tolerance/.02**2)
                & (excess_w <= self.tolerance/.02) & (excess_alpha <= self.tolerance/.02**2)
                & (excess_bound <= self.tolerance))
        good &= excess_rotation <= self.tolerance
        return dict(feasible=good, velocity_excess=excess_v, acceleration_excess=excess_a,
            angular_velocity_excess=excess_w, angular_acceleration_excess=excess_alpha,
            bound_excess=excess_bound, rotation_magnitude_excess=excess_rotation)


def constrain_plan(proposal, left, anchor, beta):
    """可微、线性且不detach的稳定约束；beta=1为原始独立十帧提案。"""
    if not 0 < beta <= 1:
        raise ValueError('参考滤波beta必须位于(0,1]')
    rho = 1-beta
    stage = (anchor-rho*left)/beta
    value = anchor
    result = []
    for target in proposal.unbind(1):
        stage = rho*stage+beta*target
        value = rho*value+beta*stage
        result.append(value)
    return torch.stack(result, 1)


def reference_kinematics(p, R, q, d, heading, dt=.02):
    """13个姿态覆盖t-1..t+11；返回包含锚点接缝和halo的真实离散导数。"""
    corrected_p = p+(heading[:, None]@d[..., :3, None]).squeeze(-1)
    corrected_R = R@so3_exp(d[..., 3:6])
    corrected_q = q+d[..., 6:]
    v = corrected_p.diff(dim=1)/dt
    w = so3_log(corrected_R[:, 1:]@corrected_R[:, :-1].transpose(-1, -2))/dt
    qd = corrected_q.diff(dim=1)/dt
    return dict(q=corrected_q[:, 1:-1], qd=qd, qdd=qd.diff(dim=1)/dt,
                root_v=v, root_a=v.diff(dim=1)/dt, root_w=w, root_alpha=w.diff(dim=1)/dt)


def quantiles(value):
    value = value.detach().float().reshape(-1).abs()
    q = torch.quantile(value, value.new_tensor([.5, .95, .99]))
    return dict(p50=float(q[0]), p95=float(q[1]), p99=float(q[2]), maximum=float(value.max()))
