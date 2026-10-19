"""离线概率候选的来源与新鲜物理评估身份分离验收。

小型真实state_dict验证显式实测延迟样本池迁移：保留旧rollout身份，目标评估绑定
当前源码/物理/奖励/数据；只有样本池和来源可显式变化，时钟、前缀预算、采样核及
奖励变化必须拒绝。候选必须完成全部计划更新并通过真实联合KL，不可作为训练恢复。
不运行Actor前向或物理世界，测试文件只写pytest独立临时目录。
"""
import copy
import pytest
import torch
from gem.closedloop.dppo.training_checkpoint import probability_candidate_binding,load_evaluation_actor


def identities():
    profile=dict(schema='profile',prefix_budget_seconds=.32,samples_seconds=[.25],provenance=[{'path':'old','sha256':'a'*64}])
    source=dict(assets={'stage1':'a'},kernel={'actual_std':[.04,.06,.08,.1]},reward={'tracking':1.},
        physical_configuration={'physics_dt':.005},data_catalog={'sha':'b'},generation_horizon={'frames':60},
        timing={'control_hz':50,'physics_hz':200,'replan_seconds':.5,'deployment_profile':profile},
        source_files={'old':'sha'})
    target=copy.deepcopy(source);target['timing']['deployment_profile']['samples_seconds']=[.26,.27]
    target['timing']['deployment_profile']['provenance']=[{'path':'current','sha256':'c'*64}]
    target['source_files']={'current':'sha'}
    return source,target


def test_explicit_profile_transfer_preserves_behavior_identity_and_clock():
    source,target=identities()
    with pytest.raises(ValueError,match='Explicit measured'):probability_candidate_binding(source,target)
    result=probability_candidate_binding(source,target,allow_measured_profile_transfer=True)
    assert result['measured_delay_profile_transfer'] and not result['training_resume']
    assert source['timing']['deployment_profile']['samples_seconds']==[.25]
    changed=copy.deepcopy(target);changed['timing']['deployment_profile']['prefix_budget_seconds']=.34
    with pytest.raises(ValueError,match='prefix budget'):probability_candidate_binding(source,changed,allow_measured_profile_transfer=True)
    changed=copy.deepcopy(target);changed['timing']['physics_hz']=100
    with pytest.raises(ValueError,match='clock differs'):probability_candidate_binding(source,changed,allow_measured_profile_transfer=True)
    changed=copy.deepcopy(target);changed['reward']['tracking']=2.
    with pytest.raises(ValueError,match='reward'):probability_candidate_binding(source,changed,allow_measured_profile_transfer=True)


def test_probability_candidate_only_loads_for_complete_matching_fresh_evaluation(tmp_path):
    source,target=identities();actor=torch.nn.Linear(2,1).float()
    payload=dict(schema='genmo.stage2.probability_candidate_evaluation.v1',evaluation_only=True,
        source_identity=source,evaluation_identity=target,
        evaluation_transfer=probability_candidate_binding(source,target,allow_measured_profile_transfer=True),
        actor=copy.deepcopy(actor.state_dict()),hard_kl={'passed':True},source_case_sha256='d'*64,
        update={'optimizer_steps':3,'planned_optimizer_steps':3})
    path=tmp_path/'candidate.pt';torch.save(payload,path)
    binding={k:target[k] for k in ('source_files','reward','timing','physical_configuration','data_catalog','generation_horizon')}
    with torch.no_grad():actor.weight.zero_()
    result=load_evaluation_actor(path,actor=actor,kernel=target['kernel'],assets=target['assets'],execution_binding=binding)
    assert not result['training_resume'] and result['evaluation_transfer']['measured_delay_profile_transfer']
    torch.testing.assert_close(actor.weight,payload['actor']['weight'])
    payload['update']['optimizer_steps']=2;torch.save(payload,path)
    with pytest.raises(ValueError,match='complete joint-KL'):load_evaluation_actor(path,actor=actor,
        kernel=target['kernel'],assets=target['assets'],execution_binding=binding)
