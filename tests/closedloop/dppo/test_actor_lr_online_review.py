"""在线实验报告的有效分母与真实权重绑定回归，不启动训练或物理仿真。

小型记录只检查A/B目标各自的全局链分母、四步访问、BC总量及回溯成本。小型
PyTorch模块包含参数和buffer，用于证明保存快照必须与评估指纹对应；夹具不是
实际Actor，也不能用于声称高学习率具有物理收益。临时checkpoint仅写pytest临时目录。
"""
import copy
import pytest
import torch

from gem.closedloop.dppo.evaluation import _model_fingerprint
from gem.closedloop.dppo.training_checkpoint import SCHEMA
from tools.eval.analyze_stage2_actor_lr_online import optimizer_rows, verify_evaluation_weights


def row(objective):
    transaction = dict(accepted=True, attempt_count=3, attempts=[{}, {}, {}])
    step = dict(global_chain_count=7, denominator=28 if objective == 'coordinate_dppo' else 7,
        internal_sample_visits=28, bc_global_samples=1024, objective=objective, backtracking=transaction)
    return dict(mode='dppo_bc', actor=dict(steps=[step], optimizer_steps=1, advantage_diagnostics={}),
        state=dict(iteration=1, training_control_steps=100), physics_paused_during_optimization=True,
        trainable4_kl={}, collection_by_rank=[], wall_seconds=1., actor_seconds=.5, critic_seconds=.1,
        critic={}, memory_by_rank=[], performance_by_rank=[], overhead_since_previous_record={})


@pytest.mark.parametrize('objective', ['coordinate_dppo', 'joint_score_pg'])
def test_online_review_uses_real_chain_denominator_and_attempts(objective):
    value = row(objective)
    result = optimizer_rows([value])
    assert result[0]['backtracking']['attempt_count'] == 3
    assert result[0]['gradient']['global_chain_count'] == 7
    wrong = copy.deepcopy(value)
    wrong['actor']['steps'][0]['denominator'] *= 8
    with pytest.raises(ValueError, match='denominator'):
        optimizer_rows([wrong])
    wrong = copy.deepcopy(value)
    wrong['actor']['steps'][0]['backtracking']['attempts'].pop()
    with pytest.raises(ValueError, match='attempt count'):
        optimizer_rows([wrong])
    wrong = copy.deepcopy(value)
    wrong['physics_paused_during_optimization'] = False
    with pytest.raises(ValueError, match='physical pause'):
        optimizer_rows([wrong])


def test_online_review_reloads_parameters_and_buffers(tmp_path):
    actor = torch.nn.Linear(2, 2)
    actor.register_buffer('frozen_buffer', torch.tensor([3.]))
    fingerprint = _model_fingerprint(actor)
    identity = dict(test='only')
    payload = dict(schema=SCHEMA, state=dict(iteration=4), identity=identity,
                   actor=copy.deepcopy(actor.state_dict()), actor_fingerprint=fingerprint)
    checkpoint = tmp_path / 'checkpoints' / 'unit_evaluated_eval4.pt'
    checkpoint.parent.mkdir()
    torch.save(payload, checkpoint)
    artifact = dict(label='eval4', iteration=4)
    report = dict(evaluation_identity=dict(training_identity=identity),
                  actor_identity=dict(model_fingerprint=fingerprint))
    actor.weight.data.zero_()
    result = verify_evaluation_weights(tmp_path, artifact, report, actor)
    assert result['actor_fingerprint'] == fingerprint and result['actual_weights_reloaded_on_cpu']
    payload['actor']['frozen_buffer'] += 1
    torch.save(payload, checkpoint)
    with pytest.raises(ValueError, match='parameters differ'):
        verify_evaluation_weights(tmp_path, artifact, report, actor)
