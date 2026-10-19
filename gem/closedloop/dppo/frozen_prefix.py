"""完整20步行为扩散中的固定Stage1前段与可训练末段。

初始化时从Stage1 Actor深拷贝独立、永久无梯度的前段网络。前15个去噪步骤使用
此副本，索引15至18使用原Actor；不共享Parameter，不把前段重新生成结果当训练数据。
条件编码各自来自对应网络；采样执行20步，概率只含19个随机步，普通KL只覆盖15..18。
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

    def __init__(self, actor, *, frozen_prefix_steps=15, frozen_actor=None, **kwargs):
        if frozen_prefix_steps!=15:
            raise ValueError('The accepted frozen-prefix design fixes the first 15 and deterministic final step of 20 steps')
        frozen=copy.deepcopy(actor) if frozen_actor is None else frozen_actor
        if ({id(p) for p in actor.parameters()} & {id(p) for p in frozen.parameters()} or
                {p.data_ptr() for p in actor.parameters()} & {p.data_ptr() for p in frozen.parameters()}):
            raise ValueError('Frozen prefix cannot share Actor parameters')
        frozen.requires_grad_(False).eval()
        kwargs["deterministic_steps"]=(19,)
        super().__init__(actor,**kwargs)
        self.frozen_policy=DenseDiffusionPolicy(frozen,**kwargs)
        self.frozen_prefix_steps=frozen_prefix_steps
        self._frozen_coefficients=self.frozen_policy._coefficients(next(frozen.parameters()).device)
        from .evaluation import _model_fingerprint
        self.kernel_config.update(frozen_prefix=dict(schema='genmo.stage2.frozen15_train4_deterministic1.v1',
            steps=frozen_prefix_steps,model_fingerprint=_model_fingerprint(frozen),
            initialization='bound_Stage1_before_trainable_Actor_restore',gaussian_kl_steps=[15,16,17,18],deterministic_final_step=19))
        import hashlib
        digest=hashlib.sha256()
        for name,value in list(frozen.named_parameters())+list(frozen.named_buffers())+list(self._frozen_coefficients.items()):
            digest.update((name+str(value.dtype)+str(tuple(value.shape))).encode())
            digest.update(value.detach().cpu().contiguous().numpy().tobytes())
        self._bound_frozen_settings=self._frozen_settings()
        digest.update(self._bound_frozen_settings.encode())
        self.kernel_config['frozen_prefix']['content_sha256']=digest.hexdigest()
        self._bound_frozen_signature=self._frozen_signature()

    trainable_steps=(15,16,17,18)

    def _frozen_signature(self):
        model=self.frozen_policy.actor
        return tuple((name,id(value),value.data_ptr(),value._version,tuple(value.shape),str(value.dtype),
                      bool(value.requires_grad)) for name,value in
                     list(model.named_parameters())+list(model.named_buffers())+
                     [(f'diffusion_cache/{name}',value) for name,value in self._frozen_coefficients.items()])

    def _frozen_settings(self):
        # 精度scope在每次真实前向都会设置并恢复；身份绑定scope内实际设置，不绑定外部环境默认值。
        from .execution_checks import precision_scope
        with precision_scope(self.frozen_policy.numerical_execution):
            return self._frozen_settings_in_scope()

    def _frozen_settings_in_scope(self):
        import json
        model=self.frozen_policy.actor
        attributes=('execution_backend','forward_backend','weight_reduction','dense_compute_dtype',
                    'dense_precise_output','dense_linear_backend','dense_weight_gradient','attention_compute_dtype')
        settings=dict(kernel=self.frozen_policy.kernel_config,timesteps=self.frozen_policy.timestep_map,
            std_schedule=self.frozen_policy.std_schedule,eta=self.frozen_policy.eta,
            guidance_scale=self.frozen_policy.guidance_scale,
            alpha=self.frozen_policy.diffusion.alphas_cumprod.tolist(),
            previous_alpha=self.frozen_policy.diffusion.alphas_cumprod_prev.tolist(),
            actual_timestep_map=list(self.frozen_policy.diffusion.timestep_map),
            live_matmul_tf32=torch.backends.cuda.matmul.allow_tf32,
            live_cudnn_tf32=torch.backends.cudnn.allow_tf32,
            matmul_precision=torch.get_float32_matmul_precision(),
            modules={name:dict(training=module.training,**{key:str(getattr(module,key))
                for key in attributes if hasattr(module,key)}) for name,module in model.named_modules()})
        return json.dumps(settings,sort_keys=True,separators=(',',':'))

    def assert_frozen_identity(self):
        if self._frozen_signature()!=self._bound_frozen_signature or self._frozen_settings()!=self._bound_frozen_settings:
            raise RuntimeError('Frozen prior parameters/buffers/numerical kernel or condition encoder changed')
        return dict(schema='genmo.stage2.frozen_identity_check.v1',checked=True,
            content_sha256=self.kernel_config['frozen_prefix']['content_sha256'],
            stochastic_frozen_steps=list(range(15)),deterministic_frozen_steps=[19],
            verification='immutable_tensor_versions_storage_runtime_settings_not_gaussian_replay')

    def _parameter_signature(self):
        if self._phase_signature is not None:return self._phase_signature
        current=super()._parameter_signature()
        frozen=getattr(self,'frozen_policy',None)
        return current if frozen is None else current+frozen._parameter_signature()

    def prepare_conditions(self, conditions, *, include_frozen=True):
        if self._phase_signature is None:self.assert_frozen_identity()
        prepared=super().prepare_conditions(conditions)
        # 有梯度调用只训练末段；冻结编码器仅在采样/周期概率回放/最终参考诊断计算。
        frozen=None
        if include_frozen and not torch.is_grad_enabled():
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
        if isinstance(step_index,int) and step_index==19:
            if torch.is_grad_enabled():raise ValueError('Final Stage1 mapping must be frozen and no-grad')
            result=self._frozen_transition(prepared,x_k,step_index,diagnostics)
            result['std']=None
            result['base_std']=None
            return result
        if isinstance(step_index,int):
            if step_index<self.frozen_prefix_steps:
                return self._frozen_transition(prepared,x_k,step_index,diagnostics)
            return super()._transition_prepared(prepared,x_k,step_index,diagnostics=diagnostics)
        steps=self._step_indices(step_index,x_k)
        if bool((steps==19).any()):raise ValueError('Deterministic step has no Gaussian probability or KL')
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
