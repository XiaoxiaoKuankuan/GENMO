"""根姿态投影对外部约束数据的只读数学复核。

除原提案与已提交历史外，隐式 KKT 还必须正确处理名义 SO(3) 姿态以及线性
上下界。这里用同一个活跃约束夹具，分别沿名义姿态的李代数方向和独立上下界
做中央有限差分；边界改变后每次都像真实入口一样先重算线性可行初值，不能
复用不满足新边界的旧 SQP 初值而得到虚假的导数反例。所有检查在 CPU float64
完成，不启动物理或训练，不把数值梯度通过等同于真实跟踪改善。
"""

import torch
from supermimic.rotations import so3_exp
from gem.closedloop.reference_adapter.so3_projection import _ImplicitSO3, _numerical_qp
from test_projection_repair import fixture


def test_nominal_rotation_manifold_direction_matches_implicit_gradient():
    projector, raw, left, anchor, rotation = fixture()
    perturbation = torch.zeros(1, 13, 3, dtype=raw.dtype, requires_grad=True)
    generator = torch.Generator().manual_seed(220)
    direction = torch.randn(perturbation.shape, dtype=raw.dtype, generator=generator)*.001
    probe = torch.linspace(.3, 1., raw.numel(), dtype=raw.dtype).reshape_as(raw)
    value = (projector(raw, left, anchor, rotation@so3_exp(perturbation))*probe).sum()
    gradient, = torch.autograd.grad(value, perturbation)
    epsilon = 1e-4
    plus = (projector(raw, left, anchor, rotation@so3_exp(epsilon*direction))*probe).sum()
    minus = (projector(raw, left, anchor, rotation@so3_exp(-epsilon*direction))*probe).sum()
    expected = (plus-minus)/(2*epsilon)
    torch.testing.assert_close((gradient*direction).sum(), expected, atol=3e-9, rtol=.003)
    assert float(projector.last_statistics['sqp_stationarity_residual64'].max()) < 1e-7


def test_independent_bounds_match_implicit_gradient_with_valid_initial_states():
    projector, raw, left, anchor, rotation = fixture()
    unit = projector.velocity[3:6]*.02
    target = (raw[..., 3:6]/unit).transpose(1, 2).flatten(1)
    # 本夹具left/anchor为零，所以线性边界完全由原候选预算给出。
    limits = torch.cat(((projector.bounds[3:6]/unit)[:, None].expand(-1, 10),
        torch.ones(3, 10, dtype=raw.dtype),
        (projector.acceleration[3:6]*.02/projector.velocity[3:6])[:, None].expand(-1, 10)), -1).flatten()[None]
    lower, upper = (-limits).clone().requires_grad_(), limits.clone().requires_grad_()
    base = torch.block_diag(projector.matrix, projector.matrix, projector.matrix)[None]
    margin = torch.full((32,), 1e-6, dtype=raw.dtype)
    margin[:11] = 1e-4/projector.rotation_velocity
    margin[11:22] = .01/projector.rotation_acceleration
    probe = torch.linspace(.3, 1., 30, dtype=raw.dtype)[None]

    def objective(lo, hi):
        # initial是数值搜索初值，不是优化目标中的可微依赖；最终梯度由KKT给出。
        with torch.no_grad():
            initial = _numerical_qp(target, lo, hi, base, 60)
        result = _ImplicitSO3.apply(initial, target, left[:, 3:6], anchor[:, 3:6], rotation,
                                   lo, hi, margin, projector)
        return (result*probe).sum()

    gradient_lower, gradient_upper = torch.autograd.grad(objective(lower, upper), (lower, upper))
    generator = torch.Generator().manual_seed(221)
    epsilon = .01
    for index, gradient in enumerate((gradient_lower, gradient_upper)):
        direction = torch.randn(lower.shape, dtype=raw.dtype, generator=generator)*.001
        plus, minus = [lower.detach(), upper.detach()], [lower.detach(), upper.detach()]
        plus[index] = plus[index]+epsilon*direction
        minus[index] = minus[index]-epsilon*direction
        expected = (objective(*plus)-objective(*minus))/(2*epsilon)
        torch.testing.assert_close((gradient*direction).sum(), expected, atol=1e-7, rtol=.003)
        assert float(projector.last_statistics['sqp_stationarity_residual64'].max()) < 1e-7
        assert float(projector.last_statistics['sqp_constraint_residual64'].max()) < 1e-7


def test_zero_and_plateau_angular_velocity_have_finite_lagrangian_curvature():
    projector, raw, left, anchor, _ = fixture()
    rotation = torch.eye(3, dtype=raw.dtype).expand(13, 3, 3)
    for plateau in (False, True):
        x = torch.zeros(3, 10, dtype=raw.dtype)
        if plateau:
            x[2] = torch.arange(1, 11, dtype=raw.dtype).clamp_max(4)*.1
        x = x.flatten()
        multiplier = torch.zeros_like(projector.angular_constraints(x, left[0, 3:6], anchor[0, 3:6], rotation))
        if plateau:
            multiplier[0] = 1.
        def lagrangian(value):
            return (projector.angular_constraints(value, left[0, 3:6], anchor[0, 3:6], rotation)*multiplier).sum()
        assert torch.func.hessian(lagrangian)(x).isfinite().all()
