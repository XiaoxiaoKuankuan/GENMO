"""Stage10 的原Stage1监督保留支路及辅助损失成熟度继承。

复用原SupervisedAnchor的数据、前缀分布、逐样本等权loss与独立RNG，不添加
新的正则项。Stage1 s595000的辅助损失已进入成熟阶段，BC不能把loss的warmup
重新归零而暂时关闭contact/物理约束；本适配只为loss传入source_global_step+
bc_update_steps，BC自身计数、采样RNG和优化器依然属于新的Stage2运行。
该偏移明确写入训练身份，恢复时核验；不把它冒充恢复Stage1优化器或训练进度。

周期能力保持诊断使用四来源val中的固定窗口和固定扩散噪声/时间步，沿用原损失
及源checkpoint成熟度。不同rank按样本索引分片，编码/预测每次评估重算；只缓存
只读监督数据，不缓存可训练模型输出。该诊断不做backward、不消费BC采样状态，
也不将运动学演示监督损失当作实际机器人奖励；样本与演示关节速度范围明确记录。
"""
from pathlib import Path
import random

import numpy as np
import torch
from torch import nn
from .trainer import SupervisedAnchor


class _MatureStage1Loss(nn.Module):
    def __init__(self,loss,source_step):super().__init__();self.loss,self.source_step=loss,int(source_step)
    def forward(self,*args,global_step=0,**kwargs):
        return self.loss(*args,global_step=self.source_step+global_step,**kwargs)


class Stage1RetentionAnchor(SupervisedAnchor):
    def __init__(self,config,actor,train_config,*,source_global_step):
        super().__init__(config,actor,train_config)
        self.losses=_MatureStage1Loss(self.losses,source_global_step)
        self.source_global_step=int(source_global_step)
        self.maximum_batch_size=config['stage9']['bc_samples_per_rollout']
        # 配对SUM以后已经统计真实全局BC梯度，省去每个监督反向的重复全模型FP64 hook。
        self.micro_gradient_diagnostics=False
    def backward(self,actor,weight):
        if self.batch_size==0:
            return dict(loss=0.,batch_size=0,samples=[],bc_update_steps=self.bc_update_steps,
                weight=weight,empty_local_BC_shard=True)
        report=super().backward(actor,weight)
        report.update(loss_maturity_step=self.source_global_step+self.bc_update_steps-1,
            warmup_origin='source_stage1_step_plus_independent_bc_updates')
        return report
    def state_dict(self):return dict(super().state_dict(),source_global_step=self.source_global_step,
        batch_contract='actual_optimizer_step_allocation.v1')
    def validate_state_dict(self,state):
        if state.get('source_global_step')!=self.source_global_step:
            raise ValueError('Stage1 BC mature-loss origin differs from checkpoint')
        if (state.get('batch_contract')!='actual_optimizer_step_allocation.v1' or type(state['batch_size'])is not int
                or not 0<=state['batch_size']<=self.maximum_batch_size):raise ValueError('Invalid dynamic BC batch identity')
        # 父类的固定batch检查只适用于Stage9；这里验证RNG和计数，保存实际尾批分配。
        return super().validate_state_dict(dict(state,batch_size=self.batch_size))
    def load_state_dict(self,state):
        super().load_state_dict(state)
        self.batch_size=state['batch_size']

    @torch.no_grad()
    def validation_report(self,actor,config,*,rank,world):
        from omegaconf import OmegaConf
        from gem.closedloop.stage1_dataset import BumiClosedLoopStage1Dataset,collate_stage1_training_samples
        from gem.closedloop.training import batch_to_device
        if not hasattr(self,'_validation_samples'):
            data=OmegaConf.load(Path(config['paths']['genmo_repo'])/'configs/closedloop/stage1_dataset_server1_fourset_90505_v1.yaml')
            data.server1_example.root=config['stage9']['bc_data_root']
            data.qpos30_stats.path=config['paths']['stats'];data.dataset_defaults.kinematics_path=config['paths']['kinematics']
            data.sample_contract.update(prefix_min_frames=self.prefix_contract['min_frames'],
                prefix_max_frames=self.prefix_contract['max_frames'],prefix_zero_probability=self.prefix_contract['zero_probability'])
            selection_rng=torch.Generator().manual_seed(20261012)
            noise_rng=torch.Generator().manual_seed(20261012+1000003*rank);items=[]
            # 数据加载即使增加了随机处理，也不能污染训练CPU/Python/NumPy状态。
            numpy_state,python_state=np.random.get_state(),random.getstate()
            try:
                with torch.random.fork_rng(devices=[]):
                    torch.manual_seed(20261012);np.random.seed(20261012);random.seed(20261012)
                    for entry in data.datasets.val.values():
                        options=OmegaConf.to_container(entry,resolve=True);options.pop('_target_',None)
                        dataset=BumiClosedLoopStage1Dataset(**options)
                        indices=torch.randperm(len(dataset),generator=selection_rng)[:min(len(dataset),2*world)].tolist()
                        for ordinal,index in enumerate(indices):
                            if ordinal%world!=rank:continue
                            sample=dataset[index]
                            items.append((collate_stage1_training_samples([sample]),
                                torch.randint(actor.diffusion_steps,(1,),generator=noise_rng),
                                torch.randn(1,120,30,generator=noise_rng)))
            finally:np.random.set_state(numpy_state);random.setstate(python_state)
            self._validation_samples=items
        modes={name:child.training for name,child in actor.named_modules()};actor.eval();results=[]
        try:
            for host,timestep,noise in self._validation_samples:
                batch=batch_to_device(host,next(actor.parameters()).device)
                prediction=actor.training_forward(batch,timesteps=timestep.to(next(actor.parameters()).device),noise=noise)
                loss,details=self.losses(batch,prediction['pred_x_start'],prediction['static_conf_logits'],global_step=0)
                if not torch.isfinite(loss):raise FloatingPointError('Nonfinite fixed Stage1 validation loss')
                q=host['target_qpos30'][0,:,9:30];mask=host['target_qpos30_valid'][0,:,9:30]
                pairs=mask[1:]&mask[:-1]
                speed=float(((q[1:]-q[:-1])*30).square()[pairs].mean().sqrt()) if pairs.any() else None
                results.append(dict(meta=host['meta'][0],loss=float(loss),
                    details={key:float(value) for key,value in details.items() if torch.is_tensor(value) and value.numel()==1},
                    diffusion_timestep=int(timestep[0]),demo_joint_speed_rms_rad_s=speed))
        finally:
            for name,child in actor.named_modules():child.training=modes[name]
        return dict(rank=rank,samples=results,source_loss_maturity=self.source_global_step,
            scope='fixed_val_original_Stage1_supervised_loss_not_physical_reward',
            BC_training_rng_unchanged=True,trainable_outputs_recomputed=True)
