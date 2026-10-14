"""完整20步行为扩散中的固定Stage1前段与可训练末段。

初始化时从Stage1 Actor深拷贝独立、永久无梯度的前段网络。前16个去噪步骤使用
此副本，末4步使用原Actor；不共享Parameter，不把前段重新生成结果当训练数据。
条件编码各自来自对应网络，采样、混合步骤概率复算和最终20步KL使用同一分派。
条件缓存显式保存冻结特征，训练末段的条件编码仍正常反向；缓存不可跨参数更新。
冻结权重由已绑定SHA的Stage1资产重建，其完整指纹进入kernel/训练/checkpoint身份；
恢复只覆盖可训练Actor，不能把恢复后的Actor误复制成冻结前段。物理时钟与噪声不变。
"""
from __future__ import annotations

import copy
import torch

from .dense_policy import DenseDiffusionPolicy


class FrozenPrefixPolicy(DenseDiffusionPolicy):
    condition_feature_names=('conditional','unconditional','frozen_conditional','frozen_unconditional')

    def __init__(self, actor, *, frozen_prefix_steps=16, frozen_actor=None, **kwargs):
        if frozen_prefix_steps!=16:
            raise ValueError('The accepted frozen-prefix design fixes the first 16 of 20 steps')
        frozen=copy.deepcopy(actor) if frozen_actor is None else frozen_actor
        if {id(p) for p in actor.parameters()} & {id(p) for p in frozen.parameters()}:
            raise ValueError('Frozen prefix cannot share Actor parameters')
        frozen.requires_grad_(False).eval()
        super().__init__(actor,**kwargs)
        self.frozen_policy=DenseDiffusionPolicy(frozen,**kwargs)
        self.frozen_prefix_steps=frozen_prefix_steps
        from .evaluation import _model_fingerprint
        self.kernel_config.update(frozen_prefix=dict(schema='genmo.stage10.frozen_stage1_prefix.v1',
            steps=frozen_prefix_steps,model_fingerprint=_model_fingerprint(frozen),
            initialization='bound_Stage1_before_trainable_Actor_restore',full20_probability_diagnostics=True))

    def _parameter_signature(self):
        if self._phase_signature is not None:return self._phase_signature
        current=super()._parameter_signature()
        frozen=getattr(self,'frozen_policy',None)
        return current if frozen is None else current+frozen._parameter_signature()

    def prepare_conditions(self, conditions):
        prepared=super().prepare_conditions(conditions)
        # 有梯度调用只训练末段；冻结编码器只在采样/完整KL的无梯度阶段计算。
        frozen=None
        if not torch.is_grad_enabled():
            frozen=self.frozen_policy.prepare_conditions(conditions)
        for name in ('conditional','unconditional'):
            prepared['frozen_'+name]=None if frozen is None else frozen[name]
        return prepared

    @staticmethod
    def _select(prepared, index):
        selected={k:v for k,v in prepared.items() if k!='_cfg_inputs'}
        selected['adapted']={k:v.index_select(0,index) for k,v in prepared['adapted'].items()}
        for name in FrozenPrefixPolicy.condition_feature_names:
            value=prepared.get(name)
            selected[name]=None if value is None else value.index_select(0,index)
        return selected

    def _frozen_transition(self, prepared, state, steps, diagnostics):
        if torch.is_grad_enabled():
            raise ValueError('Frozen prefix steps are diagnostic only; PPO must select the trainable terminal steps')
        if prepared.get('frozen_conditional') is None:
            raise ValueError('Frozen prefix condition features are missing')
        selected=dict(prepared,owner=id(self.frozen_policy),parameters=self.frozen_policy._parameter_signature(),
            grad_enabled=False,conditional=prepared['frozen_conditional'],unconditional=prepared['frozen_unconditional'])
        selected.pop('_cfg_inputs',None)
        return self.frozen_policy._transition_prepared(selected,state,steps,diagnostics=diagnostics)

    def _transition_prepared(self, prepared, x_k, step_index, *, diagnostics=False):
        if (prepared.get('owner')!=id(self) or prepared.get('parameters')!=self._parameter_signature()
                or prepared.get('grad_enabled')!=torch.is_grad_enabled()):
            raise ValueError('Frozen-prefix cache differs from Actor version or gradient mode')
        if isinstance(step_index,int):
            if step_index<self.frozen_prefix_steps:
                return self._frozen_transition(prepared,x_k,step_index,diagnostics)
            return super()._transition_prepared(prepared,x_k,step_index,diagnostics=diagnostics)
        steps=self._step_indices(step_index,x_k)
        early=torch.nonzero(steps<self.frozen_prefix_steps,as_tuple=False).flatten()
        late=torch.nonzero(steps>=self.frozen_prefix_steps,as_tuple=False).flatten()
        if not len(early):return super()._transition_prepared(prepared,x_k,steps,diagnostics=diagnostics)
        if not len(late):return self._frozen_transition(prepared,x_k,steps,diagnostics)
        before=self._frozen_transition(self._select(prepared,early),x_k[early],steps[early],diagnostics)
        after=super()._transition_prepared(self._select(prepared,late),x_k[late],steps[late],diagnostics=diagnostics)
        result={}
        for key,value in before.items():
            result[key]=(None if value is None else value.new_empty((len(x_k),*value.shape[1:]))
                .index_copy(0,early,value).index_copy(0,late,after[key]))
        return result
