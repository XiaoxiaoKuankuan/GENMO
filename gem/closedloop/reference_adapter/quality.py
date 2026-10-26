"""正式Adapter更新的数值与真实质量门槛。

消费独立真实MC校准、真实WM窗口误差和已执行物理审计的报告；阈值来自运行前配置，
缺失数据/非有限值一律失败，不能用训练loss或正则梯度替代。函数返回每项门槛及
失败原因，方便保存到JSONL/checkpoint。此模块不运行仿真，也不把报告转成物理证明。
"""

import math


def critic_gate(report, criteria, *, train_samples=None):
    required = ("min_train_samples", "min_validation_samples", "min_explained_variance",
                "min_mse_improvement", "min_near_failure_samples", "max_near_failure_mse_ratio")
    if not criteria or any(k not in criteria for k in required):
        raise ValueError("Critic质量阈值必须事前完整声明")
    finite = lambda x: x is not None and math.isfinite(x)
    constant = report.get("constant_baseline_mse")
    improvement = 1-report["mse"]/constant if finite(constant) and constant > 1e-12 else None
    near_base, near_mse = report.get("near_failure_constant_mse"), report.get("near_failure_mse")
    near_ratio = near_mse/near_base if finite(near_mse) and finite(near_base) and near_base > 1e-12 else None
    checks = dict(validation_samples=report["samples"] >= criteria["min_validation_samples"],
                  explained_variance=finite(report.get("explained_variance")) and report["explained_variance"] >= criteria["min_explained_variance"],
                  mse_improvement=finite(improvement) and improvement >= criteria["min_mse_improvement"],
                  near_failure_samples=report.get("near_failure_samples", 0) >= criteria["min_near_failure_samples"],
                  near_failure_quality=finite(near_ratio) and near_ratio <= criteria["max_near_failure_mse_ratio"])
    if train_samples is not None:
        checks["train_samples"] = train_samples >= criteria["min_train_samples"]
    return dict(passed=all(checks.values()), checks=checks, mse_improvement=improvement,
                near_failure_mse_ratio=near_ratio, thresholds=dict(criteria),
                failures=[k for k, v in checks.items() if not v])


def prediction_gate(report, criteria):
    required = ("root_position_m", "root_rotation_rad", "joint_position_rad", "joint_velocity_rad_s")
    if not criteria or any(k not in criteria for k in required):
        raise ValueError("WM真实验证阈值必须事前声明")
    checks = {f"h{h}_{k}": math.isfinite(report[str(h)][k]) and report[str(h)][k] <= criteria[k]
              for h in (1, 8, 16) for k in required}
    return dict(passed=all(checks.values()), checks=checks, thresholds=dict(criteria),
                failures=[k for k, v in checks.items() if not v])
