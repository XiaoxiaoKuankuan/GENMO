"""真实MC在零控制步请求边界终止的数学合同回归。

构造控制区间与终止事件，验证参考耗尽只关闭已有最后一步、不虚构转移或奖励；
行政截断和错位时间必须拒绝。这些是回报语义单元测试，不能代表真实物理性能。
"""

from types import SimpleNamespace

import pytest
import torch

from gem.closedloop.mb_control.returns import DecisionInterval, actor_return, close_real_mc_episode


def test_zero_step_real_terminal_closes_last_control_without_new_reward():
    start, end = SimpleNamespace(tick=0), SimpleNamespace(tick=12)
    row = DecisionInterval(start, end, torch.tensor([.2]), torch.ones(1), 1, .99, False, "control_step", {})
    event = DecisionInterval(end, end, torch.zeros(1), torch.zeros(1), 0, .99, True,
                             "reference_exhausted:no_supported_future", {})
    rows = close_real_mc_episode([row], event)
    assert len(rows) == 1 and rows[0].steps == 1 and rows[0].terminal
    assert torch.equal(actor_return(rows, torch.tensor([99.])), row.reward)


@pytest.mark.parametrize("reason,tick", [("administrative_truncation", 12), ("music_end", 24)])
def test_mc_rejects_administrative_or_misaligned_terminal(reason, tick):
    row = DecisionInterval(SimpleNamespace(tick=0), SimpleNamespace(tick=12), torch.ones(1),
                           torch.ones(1), 1, .99, False, "control_step", {})
    event = DecisionInterval(row.end, SimpleNamespace(tick=tick), torch.zeros(1), torch.zeros(1),
                             0, .99, True, reason, {})
    with pytest.raises(RuntimeError):
        close_real_mc_episode([row], event)
