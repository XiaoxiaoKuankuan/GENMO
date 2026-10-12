"""Stage10 v3 的原Stage1监督保留支路及辅助损失成熟度继承。

复用原SupervisedAnchor的数据、前缀分布、逐样本等权loss与独立RNG，不添加
新的正则项。Stage1 s595000的辅助损失已进入成熟阶段，BC不能把loss的warmup
重新归零而暂时关闭contact/物理约束；本适配只为loss传入source_global_step+
bc_update_steps，BC自身计数、采样RNG和优化器依然属于新的Stage2运行。
该偏移明确写入训练身份，恢复时核验；不把它冒充恢复Stage1优化器或训练进度。
"""
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
        # 配对SUM以后已经统计真实全局BC梯度，省去每个监督反向的重复全模型FP64 hook。
        self.micro_gradient_diagnostics=False
    def state_dict(self):return dict(super().state_dict(),source_global_step=self.source_global_step)
    def validate_state_dict(self,state):
        if state.get('source_global_step')!=self.source_global_step:
            raise ValueError('Stage1 BC mature-loss origin differs from checkpoint')
        return super().validate_state_dict(state)
