"""记录实际FP32参数更新与低精度可见性，避免把非零梯度误称为有效学习。

本模块在optimizer.step前保存一份设备端参数快照与非零梯度掩码；step后逐参数
累计变化元素数、非零梯度却舍入不变的元素数及BF16骨干实际可见的权重变化。
统计在GPU归约后一次回传小向量，不复制模型到CPU，不计算哈希/ULP分布。
它是明确计入优化器阶段的诊断成本；完整快照及时释放，不参与恢复或改写Adam。

权重可见性不能代替策略输出：最终全量KL另外记录旧rollout状态上真实执行合同
下的均值变化。两者都不证明闭环效果改善，仍需固定验证任务比较。
"""
import torch


class ParameterUpdateObservation:
    @torch.no_grad()
    def __init__(self, actor, *, bf16_backbone=False):
        self.records = [(name, parameter, parameter.detach().clone(), parameter.grad.detach().ne(0))
            for name,parameter in actor.named_parameters() if parameter.requires_grad and parameter.grad is not None]
        self.bf16_backbone = bf16_backbone

    @torch.no_grad()
    def finish(self):
        if not self.records: return dict(checked_elements=0, scope='all_parameters_with_gradients')
        total = torch.zeros(6, dtype=torch.int64, device=self.records[0][1].device)
        for name, parameter, before, nonzero in self.records:
            changed = parameter != before
            total += torch.stack((changed.new_tensor(parameter.numel(), dtype=torch.int64), changed.sum(),
                nonzero.sum(), (nonzero & ~changed).sum(),
                (parameter.bfloat16() != before.bfloat16()).sum() if self.bf16_backbone and name.startswith('denoiser.blocks.') else total.new_zeros(()),
                changed.new_tensor(parameter.numel() if self.bf16_backbone and name.startswith('denoiser.blocks.') else 0, dtype=torch.int64)))
        self.records.clear()
        values = total.cpu().tolist()
        return dict(zip(('checked_elements','fp32_changed_elements','nonzero_gradient_elements',
            'nonzero_gradient_unchanged_elements','bf16_visible_backbone_changed_elements',
            'bf16_visible_backbone_checked_elements'), values), scope='all_parameters_with_gradients',
            bf16_is_weight_visibility_not_policy_output=self.bf16_backbone)
