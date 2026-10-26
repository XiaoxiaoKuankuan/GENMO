"""局部导数与有限幅度弦斜率判定的数学回归，不是物理质量测试。

三次函数的中心差分含epsilon平方截断误差，较大幅度不能强制等于导数；最小预声明
尺度仍须满足原阈值，并检查减半趋势。错误的最小尺度和不收敛数据必须被拒绝。
本测试不构造机器人、模型回报或任何可解锁正式训练的验收报告。
"""

from gem.closedloop.mb_control.short_audit import numerical_convergence


def test_local_cubic_derivative_converges_without_erasing_large_step_error():
    criteria = {"autograd_fd_atol": .0005}
    rows = [{"direction": "root_z", "epsilon": eps, "model_autograd": .001,
             "model_centered_difference": .001-30*eps**2,
             "autograd_fd_passed": abs(30*eps**2) <= .0005}
            for eps in (.005, .0025, .00125)]
    report = numerical_convergence(rows, criteria)["root_z"]
    assert report["passed"] and not report["all_scales_within_tolerance"]
    rows[-1]["autograd_fd_passed"] = False
    assert not numerical_convergence(rows, criteria)["root_z"]["passed"]
