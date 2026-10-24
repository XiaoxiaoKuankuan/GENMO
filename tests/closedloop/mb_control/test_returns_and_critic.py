"""变长折扣、真终止与Critic输入导数测试源，本轮未执行。

控制步奖励已经含dt，测试逐步求和与两区间递推的一致性。独立Critic参数冻结时，
状态和缓存仍应获得梯度；回归targets应无梯度。这些测试不评价实际控制质量。
"""

from dataclasses import replace

import torch

from gem.closedloop.mb_control.critic import PlanAwareCritic
from gem.closedloop.mb_control.returns import DecisionInterval, IntervalAccumulator, actor_return, lambda_targets
from gem.closedloop.mb_control.scheduler import Scheduler
from gem.closedloop.mb_control.trainer import frozen_parameters


def test_variable_intervals_equal_control_step_sum(tensor_state):
    rewards = [.02, .04, .01, .08, .03]
    alive = [.9, 1., .8, .7, 1.]
    gamma = .99
    intervals = []
    for begin, end in ((0, 2), (2, 5)):
        accumulator = IntervalAccumulator(torch.zeros(1), gamma)
        for r, c in zip(rewards[begin:end], alive[begin:end]):
            accumulator.add(torch.tensor([r]), torch.tensor([c]))
        intervals.append(DecisionInterval(tensor_state, tensor_state, accumulator.reward,
                                          accumulator.survival, accumulator.steps, gamma, False, "boundary", {}))
    expected, discount = 0., 1.
    for r, c in zip(rewards, alive):
        expected += discount * r
        discount *= gamma * c
    expected += discount * .5
    torch.testing.assert_close(actor_return(intervals, torch.tensor([.5])), torch.tensor([expected]))
    terminated = [*intervals[:-1], replace(intervals[-1], terminal=True)]
    torch.testing.assert_close(actor_return(terminated, torch.tensor([100.])), actor_return(terminated, torch.tensor([0.])))
    targets = lambda_targets(intervals, [torch.ones(1, requires_grad=True)] * 2)
    assert all(not t.requires_grad for t in targets)


def test_frozen_value_parameters_keep_state_and_cache_derivatives(tensor_state):
    critic = PlanAwareCritic(Scheduler(120), width=32, layers=1, heads=4)
    state = tensor_state.state.clone().requires_grad_(True)
    reference = tensor_state.reference.world_qpos.clone().requires_grad_(True)
    z = replace(tensor_state, state=state, reference=replace(tensor_state.reference, world_qpos=reference))
    with frozen_parameters(critic):
        critic(z).sum().backward()
    assert state.grad is not None and state.grad.abs().sum() > 0
    assert reference.grad is not None and reference.grad.abs().sum() > 0
    assert all(p.grad is None for p in critic.parameters())
