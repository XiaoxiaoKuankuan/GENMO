"""实际60帧生成、Stage1监督、20步行为和冻结Critic的贯通CPU检查。

复用真实Stage1 Dataset/codec/Transformer的小型夹具，保留max_len120与参数形状，
用forward hook检查每个Transformer块实际只收到T帧，禁止120帧计算后裁剪。
同时验证真实60帧监督halo、随机链与FP64联合概率、前缀掩码和Critic时序特征。
GPU真实s595000、GMT前缀保护与物理质量必须另外实测，不以本夹具代替。
"""
import torch
import pytest
from tests.closedloop.test_stage1_actor import actor_factory,_conditions
from gem.closedloop.contracts import validate_stage1_training_batch
from gem.closedloop.dppo.dense_policy import DenseDiffusionPolicy
from gem.closedloop.dppo.pretrained_critic import PretrainedConditionCritic


@pytest.mark.parametrize('frames',[60,120])
def test_actual_horizon_keeps_pretrained_capacity_and_full_twenty_step_chain(actor_factory,frames):
    actor,batch=actor_factory(starts=(45,),motion_frames=frames)
    validate_stage1_training_batch(batch)
    assert actor.denoiser.max_len==120 and batch['target_qpos30'].shape==(1,frames,30)
    shapes=[]
    handles=[b.register_forward_pre_hook(lambda _,a:shapes.append(a[0].shape[1])) for b in actor.denoiser.blocks]
    critic=PretrainedConditionCritic(actor,dimension=32,layers=2,heads=4)
    policy=DenseDiffusionPolicy(actor,std_schedule=[.01]*20,generation_frames=frames)
    context=_conditions(batch)
    trace=policy.sample_rollout(context,generator=torch.Generator().manual_seed(42))
    assert trace['chain'].shape==(1,21,frames,30) and trace['old_means'].shape==(1,20,frames,30)
    assert trace['old_log_probs'].dtype==torch.float64 and not trace['chain'].requires_grad
    actor.train();prediction=actor.training_forward(batch)
    prediction['pred_x_start'].square().mean().backward();actor.eval()
    for handle in handles:handle.remove()
    assert shapes and set(shapes)=={frames}
    with torch.no_grad():
        features=critic.encode_frozen(context);value=critic.forward_encoded(features,torch.tensor([20.]))
    assert value.shape==(1,) and features['music'].shape[1]==frames
    assert policy.kernel_config['generation_horizon_frames']==frames
    wrong=dict(context,future_valid=torch.ones(1,180,dtype=torch.bool))
    with pytest.raises(ValueError):policy.prepare_conditions(wrong)
