"""连续验证汇总的来源权重、音乐cluster与共同执行前缀验收。

合成两个来源且任务数不同的数据，验证来源等权与按种子聚合；另一例令候选提前
失败，确保共同前缀只统计真实对齐控制，同时保留原失败和完整时长，不能用删尾部
把失败说成成功。这些是统计工具测试，不涉及Actor或物理采样算法。
"""
import pytest
import json
from tools.summarize_stage2_continuous_validation import balanced_rate_and_interval,matched_control_prefix,ERRORS


def test_robofft_summary_preserves_proxy_boundary_and_execution_evidence(tmp_path):
    from tools.summarize_stage2_continuous_validation import training_summary
    folder=tmp_path/'metrics';folder.mkdir()
    row=dict(state=dict(iteration=1),label='train_000001',mode='robofft_d',efficiency={'total_seconds':12},
        actor=dict(algorithm='robofft',eligible_actions=20,epochs=10,accepted_optimizer_steps=10,stop_reason=None,
            initial_surrogate={'loss':-.2},final_surrogate={'loss':-.25},steps=[{'learning_rate':1e-5}],
            reference_change=[{'near_joint_change_rms_rad':.002}],execution_weight_audit=[{'near_free_frames':120}]))
    (folder/'one.jsonl').write_text(json.dumps(row)+'\n')
    result=training_summary(tmp_path)[0]
    assert result['surrogate_loss_decrease']==pytest.approx(.05) and result['proxy_is_not_physical_gain']
    assert result['reference_change']==row['actor']['reference_change'] and 'chain_joint_kl' not in result


def test_sources_are_equal_weight_and_seeds_are_one_music_cluster():
    pairs=[]
    for source,count,before,after in [('Mine',1,1.,2.),('FineDance',3,3.,2.)]:
        for music in range(count):
            for seed in (42,1729):
                pairs.append(dict(dataset=source,sample_id=str(music),seed=seed,
                    initial={'reward_per_executed_second':before},final={'reward_per_executed_second':after}))
    result=balanced_rate_and_interval(pairs,draws=20)
    assert result['initial']==2. and result['final']==2.
    assert result['music_clusters']==4
    assert result['music_cluster_bootstrap_95']==[0.,0.]


def test_common_prefix_keeps_failure_and_does_not_use_unexecuted_tail():
    def make(errors,failure):
        return dict(task_id='same',dataset='Mine',physical_failure=failure,
            decisions=[dict(controls=[dict(dt_s=.02,raw_errors={k+'_squared_error':e for k in ERRORS}) for e in errors])])
    result=matched_control_prefix(make([1.,1.,100.],False),make([4.,4.],True))
    assert result['controls']==2 and result['seconds']==pytest.approx(.04)
    assert result['squared_error_integrals']['initial']['leg_joint_pos']==pytest.approx(.04)
    assert result['squared_error_integrals']['final']['leg_joint_pos']==pytest.approx(.16)
    assert result['full_task_failures']==[False,True]
    assert (result['initial_controls'],result['final_controls'])==(3,2)
