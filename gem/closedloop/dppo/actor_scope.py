"""Stage2独立候选的可训练参数范围，保留Stage1结构、初始化及真实采样核。

策略在复制冻结Stage1前段/确定性末步之后调用本模块。只从原架构允许训练的参数中
选取全部参数、qpos两层输出MLP、最终线性投影或去噪输入/时间映射，不新增网络、
不修改state_dict键。输入范围冻结Transformer主干、条件编码及输出头，仍可反传输入梯度。
腿部投影范围按真实joint_order屏蔽输出层非腿行的PPO/BC梯度；实际Gaussian仍覆盖
全部自由坐标，其他行的概率因子因参数未变自然抵消。新鲜零衰减Adam保证其矩保持零。
参数分组由actor_optimizer独立构造，共享联合KL回溯、Adam恢复和完整链PPO数学目标。
范围名称进入优化身份；不能将旧全参数Adam状态透明加载到参数子集。参数量和确切
名称返回日志，冻结接触头时BC仍保留原监督，只能更新候选范围中的参数。
"""
SCOPES=('all','qpos_output','qpos_projection','denoiser_inputs','leg_projection')


def configure_actor_scope(actor,scope='all'):
    if scope not in SCOPES:raise ValueError('Unknown explicit Stage2 Actor trainable scope')
    for handle in getattr(actor,'_stage2_row_gradient_hooks',[]):handle.remove()
    actor._stage2_row_gradient_hooks=[]
    allowed=getattr(actor,'_stage2_architecture_trainable_names',None)
    if allowed is None:
        allowed=frozenset(name for name,p in actor.named_parameters() if p.requires_grad)
        actor._stage2_architecture_trainable_names=allowed
    prefix={'qpos_output':'denoiser.final_layer.','qpos_projection':'denoiser.final_layer.fc2.',
        'leg_projection':'denoiser.final_layer.fc2.'}.get(scope)
    from .gradient_merge import _group
    names=[];count=0
    for name,p in actor.named_parameters():
        selected=(scope=='all' or
            (scope=='denoiser_inputs' and name.startswith('denoiser.') and _group(name)=='denoiser') or
            (prefix is not None and name.startswith(prefix)))
        trainable=name in allowed and selected
        p.requires_grad_(trainable);p.grad=None
        if trainable:names.append(name);count+=p.numel()
    if not names:raise ValueError('Requested Actor scope has no architecture-approved trainable parameters')
    masked=None;effective=count
    if scope=='leg_projection':
        from .probability_attribution import feature_groups
        masked=feature_groups(actor.endecoder.kinematics.joint_order)['leg_joints'];effective=0
        for name,p in actor.named_parameters():
            if name not in names:continue
            if p.shape[0]!=30:raise ValueError('Leg projection requires the unchanged qpos30 output layout')
            mask=p.new_zeros(p.shape);mask[masked]=1
            # 同时约束PPO和原BC梯度；AdamW必须保持既有weight_decay=0，旧非兼容状态不迁移。
            actor._stage2_row_gradient_hooks.append(p.register_hook(lambda grad,mask=mask:grad*mask))
            effective+=int(mask.sum())
    return dict(scope=scope,parameter_count=count,parameter_names=names,
        effective_parameter_count=effective,row_masked_output_features=masked,
        stage1_architecture_and_sampling_kernel_unchanged=True)
