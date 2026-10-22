"""独立物理评价的模型来源与配置隔离回归。

验证评价配置相对正式配置仅关闭训练父模型初始化，并强制观察入口拒绝可能在
显式评价Actor加载后覆盖模型的配置；同时检查禁止恢复训练和覆盖已有运行目录。
这些测试保护物理对照的模型身份，不运行Isaac、不加载真实checkpoint或修改训练。
所有临时文件由pytest的显式独立basetemp及外层TemporaryDirectory负责清理。
"""
from pathlib import Path

import pytest
import yaml
import numpy as np

from tools.evaluate_stage2_dppo_consumed import validate_arguments
from tools.report_stage2_dppo_suspicions import current_consumption_delta

FORMAL=Path('configs/closedloop/stage2_dppo_server2_formal.yaml')
EVALUATION=Path('configs/closedloop/stage2_dppo_server2_diagnostic_eval.yaml')


def test_evaluation_configuration_changes_only_parent_initialization():
    formal=yaml.safe_load(FORMAL.read_text())
    evaluation=yaml.safe_load(EVALUATION.read_text())
    assert formal['stage10']['actor_initialization']['mode']=='actor_adam_new_reward'
    formal['stage10']['actor_initialization']=None
    assert formal==evaluation


def test_rejects_configuration_that_would_replace_explicit_actor(tmp_path):
    with pytest.raises(ValueError,match='覆盖显式评价Actor'):
        validate_arguments(['--output-dir',str(tmp_path/'new'),'--evaluation-actor','candidate.pt',
                            '--config',str(FORMAL),'--evaluate-only'])


def test_evaluation_only_and_no_output_overwrite(tmp_path):
    output=tmp_path/'new'
    argv=['--output-dir',str(output),'--evaluation-actor','candidate.pt','--config',str(EVALUATION)]
    assert validate_arguments(argv+['--evaluate-only'])==output
    with pytest.raises(ValueError):validate_arguments(argv)
    with pytest.raises(ValueError):validate_arguments(argv+['--evaluate-only','--resume','latest'])
    output.mkdir()
    with pytest.raises(ValueError):validate_arguments(argv+['--evaluate-only'])


def test_consumption_aligns_exact_current_tick_and_plan_not_future_observation():
    window=np.full((4,21),'future',dtype='<U16')
    window[:,10]=['bootstrap:0','planA','planA','planB']
    before=dict(control_begin_tick=np.array([600,612,624,636]),
        reward_reference_tick=np.array([612,624,636,648]),
        gmt_window_plan_ids=window,reward_center_plan_id=np.array(['planA','planA','planA','planB']),
        reward_center_joint_pos=np.array([[10.],[20.],[30.],[40.]]))
    after={k:v.copy() for k,v in before.items()}
    after['reward_center_joint_pos']=np.array([[11.],[21.],[99.],[41.]])
    result=current_consumption_delta(before,after,np.arange(4),np.arange(4))
    assert result['verified_controls']==2
    assert result['rms_rad']==1.
    # planB仅在前瞻出现，且当前点的计划身份不匹配时，不能把69rad差异记为真实当前消费。
    after['gmt_window_plan_ids'][:,10]='bootstrap:0'
    assert current_consumption_delta(before,after,np.arange(4),np.arange(4))['verified_controls']==0


def test_physics_observer_preserves_transition_and_publishes_numeric_arrays(tmp_path,monkeypatch):
    import hashlib,json,sys
    from types import SimpleNamespace
    from gem.closedloop.dppo import evaluation
    from tools import evaluate_stage2_dppo_consumed as observer
    from tools import train_closedloop_stage10 as entry
    target=tmp_path/'independent'
    controls=[]
    for begin in [600,612]:
        controls.append(dict(control_tick_begin=begin,reference_tick=begin+12,
            consumed_reference_ticks=np.arange(begin-120,begin+121,12),
            consumed_plan_ids=np.full(21,'planA'),reference_plan_id='planA',
            reference={'joint_pos':np.ones(21)},actual_joint_pos_gmt=np.ones(21)*2))
    transition=SimpleNamespace(metadata=dict(execution_trace=controls,
        generated=dict(plan_id='planA',decision_tick=np.int64(600),seed=np.int64(31)),
        published=dict(commit_tick=np.int64(792)),rejection=None))
    task=dict(task_id='fixed_task',dataset='Mine',sample_id='song',seed=42)
    seen=[]

    def flow(env,task,index,plan,**kwargs):
        kwargs['transition_sink'](task,transition)
        if False:yield None
        return 'preserved'

    def run():
        iterator=evaluation.evaluation_episode_flow(None,task,0,{'plan_sha256':'fixed'},
            catalog=None,limit=None,deterministic=False,transition_sink=lambda t,r:seen.append(r),progress=None)
        with pytest.raises(StopIteration) as end:next(iterator)
        assert end.value.value=='preserved'

    monkeypatch.setattr(evaluation,'evaluation_episode_flow',flow)
    monkeypatch.setattr(entry,'main',run)
    monkeypatch.setattr(sys,'argv',['observe','--output-dir',str(target),'--evaluation-actor','candidate.pt',
                                  '--config',str(EVALUATION),'--evaluate-only'])
    monkeypatch.setenv('RANK','0')
    observer.main()
    assert seen==[transition]
    path=target/'consumed/rank00/fixed_task.npz'
    header=json.loads(path.with_suffix('.json').read_text())
    assert header['arrays_sha256']==hashlib.sha256(path.read_bytes()).hexdigest()
    assert header['controls']==2
    assert header['decisions'][0]['published']['commit_tick']==792
    with np.load(path,allow_pickle=False) as data:
        np.testing.assert_array_equal(data['control_begin_tick'],[600,612])
        assert data['gmt_window_plan_ids'].shape==(2,21)
        np.testing.assert_array_equal(data['actual_joint_pos'],np.ones((2,21))*2)
