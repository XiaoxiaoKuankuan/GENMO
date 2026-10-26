"""多卡调度与不停训预算的CPU单元回归，不替代真实八卡Newton验收。

检查八个采集GPU互不重复、评价必须显式约定在同步边界复用GPU，并确保关闭资源
上限后，大步数、长运行时间及连续拒绝统计都不生成停止理由。只测试调度语义；
真实进程并行、冻结资产和正式有效更新由服务器上的独立证据证明。
"""

import pytest
from gem.closedloop.mb_control.parallel_layout import physical_gpu_layout, resource_stop_reason


def test_eight_real_collectors_share_evaluation_only_explicitly():
    result = physical_gpu_layout(0, list(range(1, 8)), list(range(8)), share_evaluation=True)
    assert result["collection_gpus"] == list(range(8))
    assert result["shared_evaluation_gpus"] == list(range(8))
    assert result["evaluation_runs_while_collection_paused"]
    with pytest.raises(ValueError):
        physical_gpu_layout(0, list(range(1, 8)), [1])


def test_duplicate_collection_devices_are_not_eight_gpus():
    with pytest.raises(ValueError):
        physical_gpu_layout(0, [0, 1, 2, 3, 4, 5, 6], [7])
    with pytest.raises(ValueError):
        physical_gpu_layout(0, [1], [2, 2])


def test_disabled_budgets_never_turn_rejections_or_bad_quality_into_exit():
    settings = {"max_control_steps": None, "max_wall_hours": None,
                "maximum_invalid_streak": 20, "degradation_count": 1000000}
    assert resource_stop_reason(settings, 10**15, 10**15) is None
    assert resource_stop_reason({"max_control_steps": 0, "max_wall_hours": -1}, 10**15, 10**15) is None


def test_explicit_optional_resource_budget_still_has_honest_reason():
    assert resource_stop_reason({"max_control_steps": 100}, 100, 1) == "explicit_control_step_budget"
    assert resource_stop_reason({"max_wall_hours": 1}, 0, 3600) == "explicit_wall_time_budget"
    assert resource_stop_reason({"max_wall_hours": 1}, 0, 3599) is None
