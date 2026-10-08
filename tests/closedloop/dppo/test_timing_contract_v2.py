"""第二阶段部署关键延迟与训练证据耗时分离的CPU合同验收。

使用确定性虚拟时钟和现有执行替身，验证训练journal必须先可靠落盘再ACK，
仅journal耗时从部署关键路径排除，execute/ACK仍计入真实RPC耗时；环境v2保留
必要条件构造、Actor和参考prepare时间，同时排除原始证据IO，但推进物理前
仍保存完整trace。旧v1继续包含全部开销，到达tick的两个版本显式区分。
测试只创建pytest tmp_path下的小型审计文件，不启动GPU、Isaac或正式训练。
"""
from types import SimpleNamespace

import pytest
import torch

from gem.closedloop.dppo import env_adapter as env_module
from gem.closedloop.dppo import rpc as rpc_module
from gem.closedloop.dppo.rpc import AcknowledgedBackend
from tests.closedloop.dppo.test_budget_rpc import Client, Journal
from tests.closedloop.dppo.test_data_learning import context
from tests.closedloop.dppo.test_env_adapter import adapter


class Clock:
    def __init__(self):
        self.value = 0.
    def now(self):
        return self.value
    def advance(self, seconds):
        self.value += seconds


def test_rpc_journal_is_excluded_but_ack_is_required(monkeypatch):
    clock = Clock()
    monkeypatch.setattr(rpc_module.time, 'perf_counter', clock.now)
    calls = []
    class TimedClient(Client):
        def call(self, method, **payload):
            calls.append(method)
            if method == 'execute':
                clock.advance(.02)
            if method == 'ack':
                clock.advance(.005)
            return super().call(method, **payload)
    class TimedJournal(Journal):
        def append_result(self, value):
            calls.append('durable')
            clock.advance(.2)
            return super().append_result(value)
    backend = AcknowledgedBackend(TimedClient(), TimedJournal())
    backend.call('reset_episode', seed=42)
    assert calls == ['hello', 'execute', 'durable', 'ack']
    assert backend.last_call_timing['journal_seconds'] == pytest.approx(.2)
    assert backend.last_call_timing['ack_seconds'] == pytest.approx(.005)
    assert backend.last_call_timing['critical_seconds'] == pytest.approx(.025)
    assert backend.last_call_timing['total_seconds'] == pytest.approx(.225)


def make_generate_env(tmp_path, monkeypatch, clock, version, *, rank=None):
    env, backend = adapter(tmp_path)
    del env.generate
    env.timing_contract = version
    env.rank = rank
    original_call = backend.call
    def call(method, **payload):
        if method in ('reserve_prefix', 'prepare_plan'):
            total, journal = (.03, .02) if method == 'reserve_prefix' else (.08, .06)
            clock.advance(total)
            backend.last_call_timing = dict(method=method, journal_seconds=journal)
            return payload['request'] if method == 'reserve_prefix' else {'prepared_plan_id': 'ticket'}
        return original_call(method, **payload)
    backend.call = call
    def build(snapshot, reservation, *args, **kwargs):
        clock.advance(.01)
        return context(), {**reservation, 'parent_plan_id': 'old', 'world_anchor': [0., 0., 0., 0.]}
    env.builder = SimpleNamespace(build=build)
    actor = torch.nn.Linear(1, 1)
    actor.endecoder = SimpleNamespace(codec=SimpleNamespace(apply_world_anchor=lambda q, anchor: q))
    def sample(batch, *, generator):
        clock.advance(.05)
        return dict(qpos=torch.zeros(1, 120, 28), qpos30=torch.randn(1, 120, 30, generator=generator),
                    contact=torch.zeros(1, 120, 2), chain=torch.zeros(1, 3, 120, 30))
    env.policy = SimpleNamespace(actor=actor, sample_rollout=sample)
    save = env._save_evidence
    def timed_save(value, path):
        clock.advance(.4)
        return save(value, path)
    monkeypatch.setattr(env, '_save_evidence', timed_save)
    return env


@pytest.mark.parametrize('version,critical', [('legacy_audit_inclusive.v1', .57),
                                             ('deployment_critical.v2', .09)])
def test_generate_timing_keeps_trace_durable_and_versions_distinct(tmp_path, monkeypatch, version, critical):
    clock = Clock()
    monkeypatch.setattr(env_module.time, 'perf_counter', clock.now)
    env = make_generate_env(tmp_path, monkeypatch, clock, version)
    generated = env.generate()
    assert generated['elapsed'] == pytest.approx(.57)
    assert generated['critical_ready_seconds'] == pytest.approx(critical)
    assert generated['timing']['journal_seconds'] == pytest.approx(.08)
    assert generated['timing']['prepare_rpc_seconds'] == pytest.approx(.08)
    assert generated['timing']['raw_evidence_seconds'] == pytest.approx(.4)
    saved = torch.load(generated['raw_path'], weights_only=False)
    assert torch.equal(saved['trace']['chain'], generated['trace']['chain'])
    assert env.backend.tick == 600  # 审计完成前没有提前推进真实物理。


def test_v2_arrival_uses_critical_delay_and_rank_seeds_do_not_collide(tmp_path, monkeypatch):
    env, backend = adapter(tmp_path/'arrival', elapsed=.75)
    env.timing_contract = 'deployment_critical.v2'
    generate = env.generate
    env.generate = lambda **kwargs: {**generate(**kwargs), 'critical_ready_seconds': .05}
    result = env.step()
    assert result.executed_control_steps == 25
    assert next(payload for method, payload in backend.calls if method == 'commit_plan')['expected_control_tick'] == 636
    clock = Clock()
    monkeypatch.setattr(env_module.time, 'perf_counter', clock.now)
    first = make_generate_env(tmp_path/'rank0', monkeypatch, clock, 'deployment_critical.v2', rank=0).generate()
    second = make_generate_env(tmp_path/'rank1', monkeypatch, clock, 'deployment_critical.v2', rank=1).generate()
    assert first['seed'] != second['seed']
