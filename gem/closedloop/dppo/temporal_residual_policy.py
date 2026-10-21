"""独立的物理时间支持优先残差均值候选，不覆盖完整60帧PPO。

输入仍是Stage1已有音乐、50帧历史及已知前缀；从known_mask推导前缀边界，按显式
实验配置保留一段自由帧残差，再用余弦平滑降至零。第15至18步的真实高斯均值为
冻结Stage1均值加门控后的可训练均值残差，同一入口同时供真实采样及概率重算。
标准差、全部自由坐标密度、前15步与确定性末步均保持原定义；不存在丢弃远期概率。

零门控只保证该步、该旧状态条件下的均值等于冻结先验。末步Transformer存在跨帧
耦合，不能据此声称最终远期轨迹不变。此候选额外计算冻结网络，须独立测量开销和
固定噪声输出及物理效果；不默认启用，也不把旧采样核的数据静默改成新核。
"""
from __future__ import annotations

import math
import torch

from .frozen_prefix import FrozenPrefixPolicy


def residual_gate(known_mask,future_valid,*,focus_free_frames,taper_frames):
    if (type(focus_free_frames) is not int or focus_free_frames<1 or
            type(taper_frames) is not int or taper_frames<1):
        raise ValueError('时间门控必须显式给出正整数有效支持和过渡长度')
    # 姿态前缀18帧时，最后一帧的两个outgoing根位移仍是自由坐标：它们影响
    # 下一帧位置，不改变该已知姿态。因此从首个实际自由坐标帧构造门控，不能
    # 把“17帧全坐标已知”误当作机器人只保留17帧前缀；原constrain仍保护已知坐标。
    known=known_mask.all(-1)
    prefix=known.long().cumprod(1).sum(1)
    if bool((known & (torch.arange(known.shape[1],device=known.device)[None]>=prefix[:,None])).any()):
        raise ValueError('时间残差候选仅接受连续已知前缀，不猜测稀疏已知帧')
    offset=torch.arange(known.shape[1],device=known.device)[None]-prefix[:,None]
    phase=((offset-focus_free_frames+1).float()/taper_frames).clamp(0.,1.)
    gate=.5*(1.+torch.cos(math.pi*phase))
    gate=torch.where((offset>=0)&future_valid,gate,torch.zeros_like(gate))
    return gate[...,None]


class TemporalResidualPolicy(FrozenPrefixPolicy):
    def __init__(self,actor,*,temporal_residual,**kwargs):
        if set(temporal_residual)!={'focus_free_frames','taper_frames','evidence'}:
            raise ValueError('独立门控需明确参数与真实消费证据来源')
        if not isinstance(temporal_residual['evidence'],str) or not temporal_residual['evidence']:
            raise ValueError('缺少消费证据标识')
        self.temporal_residual=dict(temporal_residual)
        super().__init__(actor,**kwargs)
        self.kernel_config['temporal_residual']=dict(schema='stage2.mean_residual_time_gate.v1',
            **temporal_residual,probability='all_free_coordinates_true_joint_no_factor_removed')

    def prepare_conditions(self,conditions,*,include_frozen=True):
        # PPO原本不需要冻结条件；此候选四个可训练步也需冻结均值，故必须显式编码。
        prepared=super().prepare_conditions(conditions,include_frozen=False)
        with torch.no_grad():frozen=self.frozen_policy.prepare_conditions(conditions)
        for name in ('conditional','unconditional'):prepared['frozen_'+name]=frozen[name]
        return prepared

    def _transition_prepared(self,prepared,x_k,step_index,*,diagnostics=False):
        result=super()._transition_prepared(prepared,x_k,step_index,diagnostics=diagnostics)
        if isinstance(step_index,int):
            if step_index not in self.trainable_steps:return result
            late=torch.arange(len(x_k),device=x_k.device)
            steps=step_index
        else:
            steps=self._step_indices(step_index,x_k)
            late=torch.nonzero((steps>=15)&(steps<=18),as_tuple=False).flatten()
            if not len(late):return result
            steps=steps[late]
        selected=self._select(prepared,late)
        with torch.no_grad():prior=self._frozen_transition(selected,x_k[late],steps,False)['mean']
        gate=residual_gate(selected['adapted']['known_mask'],selected['adapted']['future_valid'],
            focus_free_frames=self.temporal_residual['focus_free_frames'],taper_frames=self.temporal_residual['taper_frames'])
        trainable=result['mean'][late]
        mean=prior+gate*(trainable-prior)
        # 边界逐位保持，不依赖浮点相消获得原值。
        mean=torch.where(gate==1.,trainable,torch.where(gate==0.,prior,mean))
        result['mean']=result['mean'].index_copy(0,late,mean)
        return result
