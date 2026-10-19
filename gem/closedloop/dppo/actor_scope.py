"""Stage2独立候选的可训练参数范围，保留Stage1结构、初始化及真实采样核。

策略在复制冻结Stage1前段/确定性末步之后调用本模块。只从原架构允许训练的参数中
选取全部参数、qpos两层输出MLP或其最终线性投影，不新增网络、不修改state_dict键。
每个候选仍使用单组AdamW，因此共享原联合KL回溯、Adam恢复和完整链PPO数学目标。
范围名称进入优化身份；不能将旧全参数Adam状态透明加载到参数子集。参数量和确切
名称返回日志，冻结接触头时BC仍保留原监督，只能更新候选范围中的参数。
"""
SCOPES=('all','qpos_output','qpos_projection')


def configure_actor_scope(actor,scope='all'):
    if scope not in SCOPES:raise ValueError('Unknown explicit Stage2 Actor trainable scope')
    allowed=getattr(actor,'_stage2_architecture_trainable_names',None)
    if allowed is None:
        allowed=frozenset(name for name,p in actor.named_parameters() if p.requires_grad)
        actor._stage2_architecture_trainable_names=allowed
    prefix={'qpos_output':'denoiser.final_layer.','qpos_projection':'denoiser.final_layer.fc2.'}.get(scope)
    names=[];count=0
    for name,p in actor.named_parameters():
        trainable=name in allowed and (scope=='all' or name.startswith(prefix))
        p.requires_grad_(trainable);p.grad=None
        if trainable:names.append(name);count+=p.numel()
    if not names:raise ValueError('Requested Actor scope has no architecture-approved trainable parameters')
    return dict(scope=scope,parameter_count=count,parameter_names=names,
        optimizer_groups=1,stage1_architecture_and_sampling_kernel_unchanged=True)
