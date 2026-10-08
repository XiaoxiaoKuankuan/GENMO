"""第二阶段性能改造的计时和数值探针验收。

使用 CPU 小型真实 Actor 与可控计时替身，验证记录器不改变随机数或梯度、嵌套区间
不会被伪装为可加总关键路径、候选自身误差与旧标量路径误差分开保存。探针必须覆盖
全部去噪步骤，并在结束后恢复模型模式和已有梯度。临时产物使用外部 pytest basetemp，
本文件不连接服务器、不启动 GMT，也不读取或覆盖正式训练目录。
"""
import torch

from gem.closedloop.dppo.execution_profile import probe_profiles
from gem.closedloop.dppo.performance import PhaseProfiler, activate, deactivate, measure
from gem.closedloop.dppo.policy import DPPODiffusionPolicy
from tests.closedloop.test_stage1_actor import _activate_branches, _conditions
from tests.closedloop.test_stage1_actor import actor_factory as actor_factory


def test_performance_counts_and_preserves_rng_and_gradients():
    rng = torch.get_rng_state().clone()
    parameter = torch.tensor(2., requires_grad=True)
    profiler = PhaseProfiler()
    token = activate(profiler)
    try:
        with measure('outer'):
            for _ in range(3):
                with measure('inner', gpu=True):
                    (parameter.square()/3).backward()
    finally:
        deactivate(token)
    report = profiler.report()
    assert report['stages']['outer']['calls'] == 1
    assert report['stages']['inner']['calls'] == 3
    assert report['intervals'] == 'inclusive_nested_do_not_sum'
    assert report['stages']['inner']['cuda_seconds'] == 0
    torch.testing.assert_close(parameter.grad, torch.tensor(4.))
    assert torch.equal(rng, torch.get_rng_state())


def test_profile_reports_self_and_cross_all_twenty_steps(actor_factory):
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        actor, batch = actor_factory(starts=(45,))
        _activate_branches(actor)
        actor.train()
        parameter = next(p for p in actor.parameters() if p.requires_grad)
        parameter.grad = torch.ones_like(parameter)
        saved = parameter.grad
        policy = DPPODiffusionPolicy(actor, steps=20)
        actor.train()
        reports = probe_profiles(policy, _conditions(batch), maximum_microbatch=1)
        assert len(reports) == 2
        for row in reports:
            assert row['backward_denoising_steps'] == 20
            assert len(row['self_consistency']['per_step_logprob_error']) == 20
            assert len(row['cross_execution']['layer_max_abs_error']['mean']) == 20
            assert row['independent_gaussian_error'] <= 1e-8
            assert row['finite_gradients']
        assert actor.training
        assert parameter.grad is saved
        assert not policy.cfg_batch
    finally:
        torch.set_num_threads(previous_threads)
