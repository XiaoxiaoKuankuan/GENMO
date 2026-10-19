"""概率归因、异方差可加和性、尾部权重与候选训练范围的服务器回归。

独立Normal公式覆盖新旧方差不同、已知前缀/padding、冻结末步排除与带符号相消。
可解析权重验证logsumexp ESS及最高权重身份，无溢出裁剪或样本删除。真实小型Stage1
网络验证缩小训练范围后零更新概率不变、流式梯度仅进入指定输出层；分布式异常
测试确认可恢复数值异常保持类型，而结构/设备异常继续作为致命错误传播。
所有资产由已有临时夹具创建，不写生产模型或数据，不将数学通过解释为物理收益。
"""
import math
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
import numpy as np
import pytest
import torch

from gem.closedloop.dppo.probability_attribution import coordinate_statistics,coordinate_report,importance_report,feature_groups
from gem.closedloop.dppo.actor_scope import configure_actor_scope
from gem.closedloop.dppo.chain_ppo import evaluate_chain_records,backward_chain_surrogate
from gem.closedloop.dppo.execution_checks import policy_phase
from gem.closedloop.dppo.tensor_cache import RolloutTensorCache
from gem.closedloop.dppo.optimization_support import _local_phase
from gem.closedloop.dppo.actor_backtracking import backtracked_step
from gem.closedloop.dppo.gradient_merge import BCGradientController
from tests.closedloop.dppo.test_clipped_dppo import case
from tests.closedloop.test_stage1_actor import actor_factory

JOINTS=['l_leg_pitch_joint','r_leg_pitch_joint','waist_yaw_joint','l_leg_roll_joint','r_leg_roll_joint',
    'l_arm_pitch_joint','r_arm_pitch_joint','l_leg_yaw_joint','r_leg_yaw_joint','l_arm_roll_joint','r_arm_roll_joint',
    'l_knee_pitch_joint','r_knee_pitch_joint','l_arm_yaw_joint','r_arm_yaw_joint','l_ankle_pitch_joint',
    'r_ankle_pitch_joint','l_elbow_pitch_joint','r_elbow_pitch_joint','l_ankle_roll_joint','r_ankle_roll_joint']


def test_coordinate_full_unequal_variance_formula_and_partition_sums():
    g=torch.Generator().manual_seed(916)
    old=torch.randn(12,7,30,generator=g,dtype=torch.float64)
    sigma=torch.rand(12,1,1,generator=g,dtype=torch.float64)+.2
    new=old+.03*torch.randn(old.shape,generator=g,dtype=torch.float64)
    new_sigma=sigma*1.13
    value=old+sigma*torch.randn(old.shape,generator=g,dtype=torch.float64)
    mask=torch.ones_like(old,dtype=torch.bool);mask[:,:2]=False;mask[:,-1]=False
    detail=coordinate_statistics(value,old,sigma,new,new_sigma,mask)
    a=torch.distributions.Normal(old,sigma);b=torch.distributions.Normal(new,new_sigma)
    torch.testing.assert_close(detail['kl'],torch.distributions.kl_divergence(a,b).masked_fill(~mask,0),rtol=1e-10,atol=1e-12)
    torch.testing.assert_close(detail['log_ratio'],(b.log_prob(value)-a.log_prob(value)).masked_fill(~mask,0),rtol=1e-10,atol=1e-12)
    reshaped={k:v.reshape(3,4,7,30) for k,v in detail.items()}
    metadata=[dict(prefix_frames=2,source='A',music_time_bin=str(i)) for i in range(3)]
    report=coordinate_report(reshaped,metadata,JOINTS)
    for partition in ('by_step','by_frame','by_feature','by_prefix','by_source','by_music_time_bin'):
        assert sum(v['total_kl'] for v in report[partition].values())==pytest.approx(report['total']['total_kl'],rel=1e-12)
        assert sum(v['free_coordinates'] for v in report[partition].values())==report['total']['free_coordinates']
    assert report['by_frame']['0']['total_kl']==0
    assert report['by_frame']['6']['total_kl']==0
    assert len(feature_groups(JOINTS)['leg_joints'])==12
    same=coordinate_statistics(value,old,sigma,old,sigma,mask)
    assert torch.count_nonzero(same['kl'])==torch.count_nonzero(same['log_ratio'])==0
    changed=new.clone();changed[~mask]=float('nan')
    ignored=coordinate_statistics(value,old,sigma,changed,new_sigma,mask)
    assert torch.equal(ignored['log_ratio'],detail['log_ratio'])


def test_importance_logsumexp_exact_identity_and_extreme_tail_identity():
    metadata=[dict(identity={'chain':i}) for i in range(4)]
    log=torch.tensor([0.,0.,0.,math.log(1000.)],dtype=torch.float64)
    report=importance_report(log,torch.tensor([0.,0.,0.,.01]),torch.tensor([1.,-1.,1.,1.]),metadata)
    weights=np.array([1.,1.,1.,1000.])
    assert report['effective_chains']==pytest.approx(weights.sum()**2/np.square(weights).sum())
    assert report['highest_weight_chains'][0]['identity']=={'chain':3}
    assert report['top_weight_shares']['0.001']==pytest.approx(1000/1003)
    assert report['active_clip_fraction']==.25
    same=importance_report(torch.zeros(4),torch.zeros(4),torch.ones(4),metadata)
    assert same['effective_chains']==pytest.approx(4.) and same['ess_fraction']==pytest.approx(1.)
    extreme=importance_report(log+1000,torch.ones(4),torch.ones(4),metadata)
    assert extreme['effective_chains']==pytest.approx(report['effective_chains'])
    assert extreme['ratio'] is None and extreme['ratio_unrepresentable_count']==4


@pytest.mark.parametrize('scope,prefix',[('qpos_output','denoiser.final_layer.'),('qpos_projection','denoiser.final_layer.fc2.')])
def test_restricted_actor_keeps_kernel_and_real_streamed_score(scope,prefix,actor_factory):
    policy,rows,targets=case(actor_factory)
    original={n:p.detach().clone() for n,p in policy.actor.named_parameters()}
    kernel=dict(policy.kernel_config)
    cache=RolloutTensorCache(rows,targets,'cpu')
    with policy_phase(policy):baseline=evaluate_chain_records(policy,rows,cache,microbatch=3)
    report=configure_actor_scope(policy.actor,scope)
    assert all(name.startswith(prefix) for name in report['parameter_names'])
    try:
        with policy_phase(policy):
            records=evaluate_chain_records(policy,rows,cache,microbatch=3)
            # CPU微批重放和原采样批量可有FP32差异；范围切换本身必须逐位不变。
            torch.testing.assert_close(records,baseline,rtol=0,atol=0)
            backward_chain_surrogate(policy,rows,cache,records,clip=.1,global_chain_count=len(rows),microbatch=3)
        opt=torch.optim.AdamW([p for p in policy.actor.parameters() if p.requires_grad],lr=1e-6)
        opt.step()
        assert any(not torch.equal(original[n],p) for n,p in policy.actor.named_parameters() if n.startswith(prefix))
        assert all(torch.equal(original[n],p) for n,p in policy.actor.named_parameters() if not n.startswith(prefix))
        assert policy.kernel_config==kernel and policy.assert_frozen_identity()['checked']
        configure_actor_scope(policy.actor,'all')
        assert any(p.requires_grad for n,p in policy.actor.named_parameters() if not n.startswith(prefix))
    finally:cache.close()


def test_cooperative_numerical_type_reaches_finite_backtracking():
    class Collective:
        rank=0
        def all_gather_object(self,value):return [value]
    collective=Collective();actor=torch.nn.Linear(2,1).float();opt=torch.optim.AdamW(actor.parameters(),lr=1e-6)
    for p in actor.parameters():p.grad=torch.ones_like(p)
    count=[]
    def evaluate():
        count.append(1)
        with _local_phase(collective,'test_candidate'):
            if len(count)==1:raise FloatingPointError('nonfinite candidate')
        return {}
    result=backtracked_step(actor,opt,BCGradientController(),evaluate=evaluate,limits={},distributed=collective)
    assert result['accepted'] and result['attempt_count']==2 and result['accepted_lr']==5e-7
    with pytest.raises(RuntimeError,match='device identity'):
        with _local_phase(collective,'fatal'):raise ValueError('device identity')


def test_candidate_configurations_bind_scope_and_200_iteration_cap():
    from gem.closedloop.dppo.training_config import configuration
    from gem.closedloop.dppo.actor_optimization import optimization_binding
    root=Path(__file__).resolve().parents[3]
    candidates=[]
    for scope in ('qpos_output','qpos_projection'):
        c=configuration(root/f'configs/closedloop/stage2_{scope}.yaml')
        assert c['stage10']['run_control']['max_training_iterations']==200
        assert c['stage10']['limits']['accepted_iterations']==200
        assert c['stage9']['actor_trainable_scope']==scope
        assert c['runtime']['record_compact_execution_provenance']
        candidates.append(optimization_binding(c['stage10']))
    assert candidates[0]!=candidates[1]
    subprocess.run([sys.executable,'-B',str(root/'tools/analyze_stage2_probability.py'),'--help'],check=True,capture_output=True,text=True,timeout=30)
