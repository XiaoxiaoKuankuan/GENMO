"""四步KL证据、评估权重绑定及物理结果诊断的CPU回归测试。

本文件仅在pytest临时目录构造明确的合成JSON，不采集物理数据、不发布训练配置。
验证软/硬阈值、四步覆盖、冻结身份和实际评估权重SHA不能被空证据或旧20步统计
替代；新增物理失败仍被保留为诊断，但不阻止发布标定或启动训练。
另验证Actor初始化和Critic训练来源可以不同，而物理源码与奖励
必须完全相同。这些合成门禁测试不代表真实KL标定通过。
"""
import copy
import json
import pytest
from gem.closedloop.dppo.kl_calibration import digest,validate_calibration
from tools.eval.compare_closedloop_stage10 import paired_execution_identity


def evidence(tmp_path,fault=None):
    kernel={'steps':[15,16,17,18]};reward={'version':'synthetic_new_reward'}
    limits=dict(mean_coordinate=.001,maximum_step_mean_coordinate=.004,p95_coordinate=.003)
    kl=dict(gaussian_kl_steps=[15,16,17,18],covered_internal_transitions=8,covered_chains=2,
        frozen_identity={'checked':True},mean_coordinate_kl=.0001,maximum_step_mean_coordinate_kl=.0002,
        coordinate_kl={'p95':.0003})
    candidate=dict(actor_lr=1e-7,trainable4_kl=kl,hard_kl={'passed':True},evaluation_actor={'sha256':'a'*64},
        update=dict(optimizer_steps=2,planned_optimizer_steps=2,steps=[{'pre_step_mean_coordinate_kl':.0001}]))
    fixed=dict(schema='genmo.stage2.real_data_four_step_actor_calibration.v1',kernel_sha256=digest(kernel),
        reward_sha256=digest(reward),candidates=[candidate])
    physical=dict(schema='genmo.closedloop.stage10.paired_comparison.v1',status='passed',paired_task_count=32,
        all_planned_tasks_included=True,execution_kernel_sha256=digest(kernel),execution_reward_sha256=digest(reward),
        evaluated_actor_initialization={'sha256':'a'*64},overall={'failure_pair_counts':{'new_failure':0}})
    if fault=='twenty':kl['gaussian_kl_steps']=list(range(20))
    if fault=='coverage':kl['covered_internal_transitions']=7
    if fault=='frozen':kl['frozen_identity']['checked']=False
    if fault=='hard':kl['mean_coordinate_kl']=.002
    if fault=='partial':candidate['update']['optimizer_steps']=1
    if fault=='physical_actor':physical['evaluated_actor_initialization']['sha256']='b'*64
    if fault=='failure':physical['overall']['failure_pair_counts']['new_failure']=1
    if fault=='reward':physical['execution_reward_sha256']=digest({'other':1})
    def write(name,value):
        import hashlib
        p=tmp_path/name;p.write_text(json.dumps(value))
        return {'path':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
    fixed_ref=write('fixed.json',fixed)
    report=dict(schema='genmo.stage2.four_step_kl_calibration.v1',status='passed',kernel_sha256=digest(kernel),
        reward_sha256=digest(reward),hard_limits=copy.deepcopy(limits),soft_limit=.0005,
        fixed_rollout_trials=[dict(evidence=fixed_ref,actor_lr=1e-7)],physical_comparison=write('physical.json',physical),
        selection_reason='Synthetic fixture only: test that every required measured field is bound, not a physical acceptance.')
    limits['calibration']=dict(status='validated',evidence=write('calibration.json',report))
    if fault=='sha':(tmp_path/'fixed.json').write_text('{}')
    return limits,kernel,reward


def test_four_step_evidence_binds_soft_hard_and_physical_actor(tmp_path):
    limits,kernel,reward=evidence(tmp_path)
    assert validate_calibration(limits,kernel,reward,.0005)['status']=='passed'
    with pytest.raises(ValueError,match='thresholds'):
        validate_calibration(limits,kernel,reward,.0006)
    limits['mean_coordinate']=.00001
    with pytest.raises(ValueError,match='thresholds'):
        validate_calibration(limits,kernel,reward,.0005)


def test_new_physical_failure_is_diagnostic_not_training_admission(tmp_path):
    limits,kernel,reward=evidence(tmp_path,'failure')
    report=validate_calibration(limits,kernel,reward,.0005)
    physical=json.loads((tmp_path/'physical.json').read_text())
    assert report['status']=='passed'
    assert physical['overall']['failure_pair_counts']['new_failure']==1


@pytest.mark.parametrize('fault',['twenty','coverage','frozen','hard','partial','physical_actor','reward','sha'])
def test_invalid_calibration_cannot_enable_actor_learning(tmp_path,fault):
    limits,kernel,reward=evidence(tmp_path,fault)
    with pytest.raises(ValueError):validate_calibration(limits,kernel,reward,.0005)


def test_paired_comparison_ignores_only_learning_state():
    keys=('schema','assets','kernel','reward','runtime','timing','physical_episode','ppo_rollout',
        'generation_horizon','physical_configuration','source_files','data_catalog','source_stage1_step','torch_version','recovery')
    first={'training_identity':{key:{'identity':key} for key in keys},'critical_latency_budgets':[.32]}
    first['training_identity'].update(actor_initialization={'kind':'stage1'},critic={'artifact':None})
    second=copy.deepcopy(first);second['training_identity'].update(actor_initialization={'kind':'updated'},critic={'artifact':'new_mc'})
    assert paired_execution_identity(first)==paired_execution_identity(second)
    second['training_identity']['source_files']={'changed':True}
    assert paired_execution_identity(first)!=paired_execution_identity(second)


@pytest.mark.parametrize('fault', [None, 'actor_lr', 'ppo_epochs', 'global_minibatch_chains', 'bc_samples'])
def test_published_calibration_binds_actual_optimizer_and_can_be_read_by_training(tmp_path, fault):
    """真实执行发布器与训练读取函数，合成证据只验证身份门禁，不冒充物理标定。"""
    import hashlib
    from pathlib import Path
    import torch
    import yaml
    from gem.closedloop.dppo.actor_experiment import optimization_binding
    from gem.closedloop.dppo.kl_calibration import publish_calibration
    from gem.closedloop.dppo.training_config import configuration
    limits, kernel, _ = evidence(tmp_path)
    root = Path(__file__).resolve().parents[3]
    config = yaml.safe_load((root/'configs/closedloop/stage10_server1.yaml').read_text())
    config['stage10']['kl'].update({key: limits[key] for key in (
        'mean_coordinate', 'maximum_step_mean_coordinate', 'p95_coordinate')})
    config['stage10']['training'].update(actor_lr=1e-7, kl_soft_stop_coordinate=.0005)
    path = tmp_path/'configuration.yaml'; path.write_text(yaml.safe_dump(config))
    resolved = configuration(path)
    optimization = optimization_binding(resolved['stage10'])
    reward = resolved['stage9']['reward']
    fixed = json.loads((tmp_path/'fixed.json').read_text())
    fixed['reward_sha256'] = digest(reward)
    fixed['candidates'][0]['optimization'] = copy.deepcopy(optimization)
    if fault:
        fixed['candidates'][0]['optimization'][fault] *= 2
    case = tmp_path/'case'; case.mkdir()
    torch.save({'kernel': kernel}, case/'learning_state.pt')
    (case/'case.json').write_text(json.dumps(dict(learning_state_sha256=hashlib.sha256(
        (case/'learning_state.pt').read_bytes()).hexdigest())))
    fixed.update(source_case=str(case), source_case_sha256=hashlib.sha256((case/'case.json').read_bytes()).hexdigest())
    (tmp_path/'fixed.json').write_text(json.dumps(fixed))
    physical = json.loads((tmp_path/'physical.json').read_text())
    physical['execution_reward_sha256'] = digest(reward)
    physical['overall']['failure_pair_counts']['new_failure'] = 1
    (tmp_path/'physical.json').write_text(json.dumps(physical))
    args = (path, tmp_path/'fixed.json', tmp_path/'physical.json', 1e-7,
            'Synthetic publication regression: bind measured optimization and preserve physical failures as diagnostics.',
            tmp_path/'published.json')
    if fault:
        with pytest.raises(ValueError, match='Measured candidate optimization'):
            publish_calibration(*args)
        assert not (tmp_path/'published.json').exists()
        assert not (tmp_path/'published.json.pending').exists()
    else:
        reference = publish_calibration(*args)
        limits['calibration'] = dict(status='validated', evidence=reference)
        result = validate_calibration(limits, kernel, reward, .0005, optimization=optimization)
        assert result['schema'] == 'genmo.stage2.four_step_kl_calibration.v2'
        assert result['physical_outcome_policy']=='diagnostic_only_not_training_admission'
        assert result['physical_failure_pair_counts']['new_failure']==1
