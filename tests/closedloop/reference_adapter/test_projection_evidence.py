"""部分想象拒样的独立CPU证据与完整checkpoint回归。

使用现有真实SM权重数值夹具，在第二个想象时刻注入明确的数值失败，检查实际
失败输入而非仅初始状态被保存；其余microbatch仍执行一次Adam。证据经常规完整
checkpoint写入并读回后须无计算图、可重建同一投影器和调用输入。合成异常只测试
保存及恢复语义，不宣称真实物理方向或模拟器已通过，也不触碰任何正式运行文件。
"""

import torch
from supermimic.networks import Critic
from test_alternating import setup
from gem.closedloop.reference_adapter.alternating_learning import policy_update
from gem.closedloop.reference_adapter.reference_constraints import ReferenceProjection
from gem.closedloop.reference_adapter.so3_projection import ProjectionNumericalFailure
from gem.closedloop.reference_adapter.projection_evidence import (
    checkpoint_projection_evidence, restore_projection)
from gem.closedloop.reference_adapter.vector_checkpoint import save_vector
from gem.closedloop.reference_adapter.vector_collector import VectorCollector
from gem.closedloop.reference_adapter.vector_replay import VectorReplay
from gem.closedloop.reference_adapter.vector_learning import TangentReplay


def test_partial_rejection_saved_in_regular_full_checkpoint_and_reproducible(system, tmp_path, monkeypatch):
    from gem.closedloop.reference_adapter import so3_projection
    engine, control, z = setup(system)
    control.projection = ReferenceProjection(control.bundle.spec, control.adapter.bounds)
    critic = Critic(2835)
    optimizers = dict(adapter=torch.optim.Adam(control.adapter.parameters(), lr=3e-5),
        wm=torch.optim.Adam(engine.dynamics.model.net.parameters(), lr=3e-5),
        critic=torch.optim.Adam(critic.parameters(), lr=1e-4))
    original, calls = so3_projection.project_rotation_implicit, []
    def fail_second(projector, proposal, projected, left, anchor, rotation, lower, upper):
        calls.append(1)
        if len(calls) == 2:
            raise ProjectionNumericalFailure('fixture_actual_step1_projection_failure',
                torch.tensor([True]), certificate={'feasible': torch.tensor([False])},
                diagnostic={'sqp_stationarity_residual64': proposal.new_tensor([3e-6])})
        return original(projector, proposal, projected, left, anchor, rotation, lower, upper)
    monkeypatch.setattr(so3_projection, 'project_rotation_implicit', fail_second)
    report, _ = policy_update(control, engine.dynamics, critic, optimizers['adapter'], z, 2,
        terminal_value=False, microbatch=1, failure_context=dict(source='fixture_source',
            iteration=101, adapter_updates_before=0, candidate_adapter_update=1))
    assert report['accepted_samples'] == 1 and report['rejected_samples'] == 1
    assert report['optimizer_step_completed'] and all(int(v['step']) == 1 for v in optimizers['adapter'].state.values())
    extra = {}
    checkpoint_projection_evidence(extra, control)
    evidence = extra['imagination_failure']
    assert evidence['imagination_step'] == 1 and evidence['offset'] == 0
    assert evidence['context']['candidate_adapter_update'] == 1
    assert evidence['context']['source'] == 'fixture_source'
    assert len(evidence['context']['adapter_sha256']) == len(evidence['context']['wm_sha256']) == 64
    assert evidence['accepted_sample_indices'] == [1]
    torch.testing.assert_close(evidence['failure_state']['frame'], z['frame'][:1]+1)
    torch.testing.assert_close(evidence['batch_identity']['episode'], z['episode'])
    class Env:
        N = 2
    collector = VectorCollector(Env(), control)
    collector.state = z
    replay = VectorReplay(80, 2, 'cpu', compute_mc=False)
    replay.add(z, z['action'], z['state'], torch.ones(2), torch.ones(2),
        torch.zeros(2, dtype=torch.bool), torch.zeros(2, dtype=torch.bool), 1)
    path = tmp_path/'partial_rejection_regular_checkpoint.pt'
    save_vector(path, engine, critic, optimizers, replay, TangentReplay(1), collector,
        dict(accepted_updates=1, wm_updates=engine.dynamics.model_version), {}, extra_state=extra)
    saved = torch.load(path, map_location='cpu', weights_only=False)['extra_state']['imagination_failure']
    def check(value):
        if isinstance(value, torch.Tensor):
            assert value.device.type == 'cpu' and value.grad_fn is None and not value.requires_grad
        elif isinstance(value, dict):
            for child in value.values():
                check(child)
        elif isinstance(value, (tuple, list)):
            for child in value:
                check(child)
    check(saved)
    projector, inputs = restore_projection(saved['projection_inputs'])
    for key, value in control.projection.state_dict().items():
        torch.testing.assert_close(projector.state_dict()[key], value)
    assert inputs['raw'].requires_grad
    actual = projector(inputs['raw'], inputs['left'], inputs['anchor'], inputs['nominal_rotation'])
    expected = control.projection(inputs['raw'], inputs['left'], inputs['anchor'], inputs['nominal_rotation'])
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    # 恢复后尚无新失败时不能抹掉checkpoint带来的历史证据。
    del control.last_imagination_failure
    checkpoint_projection_evidence(extra, control)
    assert extra['imagination_failure'] is evidence


def test_certificate_rejection_retains_projection_grad_mode_outside_no_grad(system, monkeypatch):
    engine, control, z = setup(system)
    control.projection = ReferenceProjection(control.bundle.spec, control.adapter.bounds)
    original, calls = control.constraint_certificate, []
    def certificate(state, proposal):
        result = original(state, proposal)
        calls.append(1)
        if len(calls) == 1:
            result['feasible'] = torch.tensor([False])
        return result
    monkeypatch.setattr(control, 'constraint_certificate', certificate)
    report, _ = policy_update(control, engine.dynamics, Critic(2835),
        torch.optim.Adam(control.adapter.parameters(), lr=3e-5), z, 1,
        terminal_value=False, microbatch=1)
    assert report['accepted_samples'] == report['rejected_samples'] == 1
    evidence = control.last_imagination_failure['projection_inputs']
    assert evidence['gradient_enabled'] and evidence['requires_grad']['raw']
    assert not evidence['inputs']['raw'].requires_grad
    _, inputs = restore_projection(evidence)
    assert inputs['raw'].requires_grad and not evidence['inputs']['raw'].requires_grad


def test_sqp_and_kkt_polish_statistics_are_separate_and_do_not_retain_graph():
    from gem.closedloop.reference_adapter.alternating_learning import (
        _accumulate_projection_statistics, _merge_projection_statistics)
    first, second = {}, {}
    _accumulate_projection_statistics(first, dict(sqp_iterations=torch.tensor([2., 3.], requires_grad=True),
        sqp_kkt_polish_iterations=torch.tensor([2., 0., 3.], requires_grad=True)))
    _accumulate_projection_statistics(second, dict(sqp_kkt_polish_iterations=torch.tensor([1., 5.])))
    _merge_projection_statistics(first, second)
    assert first['sqp_iterations_sum'] == 5 and first['sqp_iterations_count'] == 2
    assert first['sqp_iterations_max'] == 3
    assert first['sqp_kkt_polish_iterations_sum'] == 11 and first['sqp_kkt_polish_iterations_count'] == 5
    assert first['sqp_kkt_polish_iterations_max'] == 5
    assert all(value.grad_fn is None and not value.requires_grad for value in first.values())
