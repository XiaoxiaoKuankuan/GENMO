"""Stage10 v3 评估与完整保存边界的集成回归。

此测试针对真实八卡入口在初始评估完成后发生的接口错误：选择报告含有 eligible
等诊断字段，而保存绑定函数仅接受 best_observed 与 best_saved 两个历史指针。
测试调用真实 v3 包装函数和实际选择/校验逻辑，只替换耗时的仿真评估及模型快照，
在 pytest 独立临时目录核对实际文件 SHA、模型身份、best 指针与报告发布一致性。
错误模型指纹必须拒绝，不能因修复关键字传递而削弱重要评估模型的身份检查。
本测试不运行物理环境，不把小型集成回归解释为 GPU 性能或强化学习质量验收。
"""
from __future__ import annotations

import copy
import json
from types import SimpleNamespace

import pytest

from gem.closedloop.dppo import parallel_training, training_v3
from gem.closedloop.dppo.checkpoint_v3 import file_sha256
from gem.closedloop.dppo.periodic_monitor import choose_best


@pytest.mark.parametrize('matching_actor', [True, False])
def test_evaluation_binds_only_best_pointers_and_validates_actor(tmp_path, monkeypatch, matching_actor):
    session = tmp_path / 'sessions' / 'finite'
    artifact = session / 'evaluations' / 'initial.json'
    artifact.parent.mkdir(parents=True)
    checkpoint = tmp_path / 'checkpoints' / 'evaluated.pt'
    checkpoint.parent.mkdir()
    checkpoint.write_bytes(b'independent finite checkpoint fixture')
    result = dict(iteration=0, plan_sha256='fixed-plan', actor_identity=dict(model_fingerprint='actor'),
        source_balanced_reward=2., physical_failure_count=0, mean_executed_seconds=10.,
        episode_metrics=[dict(task_id='one', dataset='AIST', seed=42, reward=2.,
                             executed_seconds=10., physical_failure=0)])
    result['selection'] = choose_best(result, result, iteration=0)
    artifact.write_text(json.dumps(result))
    report = dict(evaluation_artifacts=[dict(path=str(artifact.relative_to(tmp_path)),
                                            sha256=file_sha256(artifact))])
    publication = dict(path=str(checkpoint.relative_to(tmp_path)), sha256=file_sha256(checkpoint),
                       iteration=0, metadata=dict(actor_model_fingerprint='actor'))
    context = SimpleNamespace(distributed=None, actor=object(), output=tmp_path, session=session, state={})
    monkeypatch.setattr(parallel_training, '_evaluate_and_record', lambda *args: copy.deepcopy(result))
    monkeypatch.setattr(training_v3, '_snapshot', lambda *args, **kwargs: dict(publication=publication))
    monkeypatch.setattr(training_v3, 'root_call', lambda distributed, call: call())
    monkeypatch.setattr(training_v3, '_model_fingerprint', lambda actor: 'actor' if matching_actor else 'wrong')
    if not matching_actor:
        with pytest.raises(ValueError, match='Actor differs'):
            training_v3._evaluate(context, report, 'initial')
        assert not (tmp_path / 'best_checkpoint.json').exists()
        return
    saved = training_v3._evaluate(context, report, 'initial')
    assert saved['selection']['eligible'] is True
    assert saved['selection']['best_saved']['checkpoint']['sha256'] == file_sha256(checkpoint)
    assert set(context.state) == {'best_observed', 'best_saved'}
    assert not (tmp_path / 'eligible.json').exists()
    assert json.loads((tmp_path / 'best_checkpoint.json').read_text()) == publication
    assert json.loads(artifact.read_text()) == saved
    assert report['evaluation_artifacts'][0]['sha256'] == file_sha256(artifact)
