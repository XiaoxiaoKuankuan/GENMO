"""从 Stage1 条件编码器独立复制的时序价值网络与冻结特征缓存接口。

音乐 MLP、存在标记、带相对时间的掩码历史 GRU、前缀编码器和物理归一化均从
已核验的 Stage1 Actor 复制。模块和存储与 Actor 完全隔离；首版冻结这些副本，
训练只更新低维投影、两层 Temporal Transformer 和 Value Head。120帧音乐和
前缀保持顺序，不做平均池化；相对时间、音乐/未来/前缀 mask 和音乐剩余时长
明确参与融合。Value token通过注意力读取完整条件，输出当前条件的 V(s)。

encode_frozen()仅处理决策前已有的十个条件字段，拒绝动作、奖励和未来状态。
冻结特征可跨 Critic epoch 缓存，forward_encoded()的可训练部分每步重新执行。
train()不会启用冻结编码器的 dropout。实验性解冻只作用于 Critic 自己的副本，
此时禁止使用冻结特征缓存，必须新建实验身份。全空历史继续输出严格零特征。
"""
from __future__ import annotations

import copy
import math

import torch
from torch import nn

from gem.closedloop.contracts import STAGE1_CONDITION_KEYS, validate_stage1_condition_batch


class PretrainedConditionCritic(nn.Module):
    def __init__(self, actor, *, dimension=256, layers=2, heads=4, freeze_encoders=True):
        super().__init__()
        if dimension < 8 or dimension%heads or dimension%2 or layers < 1:
            raise ValueError('Invalid temporal Critic dimensions')
        self.dimension, self.freeze_encoders = dimension, bool(freeze_encoders)
        for name in ('music_embedder','cond_exists_embedder','history_encoder','prefix_encoder','proprio_normalizer'):
            setattr(self,name,copy.deepcopy(getattr(actor,name)))
        self.register_buffer('qpos_mean',actor.endecoder.mean.detach().clone())
        self.register_buffer('qpos_std',actor.endecoder.std.detach().clone())
        width=actor.denoiser.latent_dim
        self.music_projection=nn.Linear(width,dimension)
        self.prefix_projection=nn.Linear(width,dimension)
        self.history_projection=nn.Linear(width,dimension)
        self.mask_projection=nn.Linear(3,dimension,bias=False)
        self.remaining_projection=nn.Sequential(nn.Linear(1,dimension),nn.SiLU(),nn.Linear(dimension,dimension))
        self.value_token=nn.Parameter(torch.zeros(1,1,dimension))
        layer=nn.TransformerEncoderLayer(dimension,heads,dimension*2,dropout=0.,activation='gelu',
            batch_first=True,norm_first=True)
        self.temporal=nn.TransformerEncoder(layer,layers,enable_nested_tensor=False)
        self.value_head=nn.Sequential(nn.LayerNorm(dimension),nn.Linear(dimension,dimension//2),nn.SiLU(),nn.Linear(dimension//2,1))
        frequency=torch.exp(torch.arange(0,dimension,2,dtype=torch.float32)*(-math.log(10000.)/dimension))
        self.register_buffer('time_frequency',frequency)
        if self.freeze_encoders:
            for module in self.encoder_modules():
                module.requires_grad_(False);module.eval()
        actor_parameters={id(p) for p in actor.parameters()}
        actor_storage={p.data_ptr() for p in actor.parameters()}
        if any(id(p) in actor_parameters or p.data_ptr() in actor_storage for p in self.parameters()):
            raise RuntimeError('Critic must own independent parameter storage')

    def encoder_modules(self):
        return tuple(m for m in (self.music_embedder,self.cond_exists_embedder,self.history_encoder,
            self.prefix_encoder,self.proprio_normalizer) if m is not None)

    def train(self, mode=True):
        super().train(mode)
        if self.freeze_encoders:
            for module in self.encoder_modules():module.eval()
        return self

    def _encode(self, context):
        if set(context)!=set(STAGE1_CONDITION_KEYS):
            raise ValueError('Critic accepts only causal Stage1 conditions')
        validate_stage1_condition_batch(context,history_steps=50)
        device=self.qpos_mean.device
        c={k:v.to(device) for k,v in context.items()}
        future=c['future_valid'];music_valid=c['music_valid']&future
        mask=c['known_qpos30_mask']&future[...,None]
        physical=torch.where(mask,c['known_qpos30'].float(),0.)
        known=torch.where(mask,(physical-self.qpos_mean)/self.qpos_std,0.)
        music=self.music_embedder(torch.where(music_valid[...,None],c['music_features'].float(),0.))
        music=torch.where(music_valid[...,None],music,0.)
        if self.cond_exists_embedder is not None:
            music=self.cond_exists_embedder(torch.cat((music,music_valid[...,None].to(music)),-1))
        music=torch.where(future[...,None],music,0.)
        valid=c['proprio_history_valid']
        history=self.proprio_normalizer(c['proprio_history'].float(),valid)
        relative=torch.where(valid,c['proprio_history_times']-c['decision_time'][:,None],0.).float()
        h=self.history_encoder(history,valid,relative)
        prefix=self.prefix_encoder(known,mask)
        return dict(music=music,prefix=prefix,history=h,future_valid=future,
            music_valid=music_valid,prefix_valid=mask.any(-1),
            relative_times=torch.where(future,c['future_times']-c['decision_time'][:,None],0.).float())

    @torch.no_grad()
    def encode_frozen(self, context):
        if not self.freeze_encoders:
            raise RuntimeError('Trainable Critic encoders cannot use detached feature caches')
        return self._encode(context)

    def forward_encoded(self, features, remaining_music_seconds):
        f=features;batch=len(f['music'])
        remaining=torch.as_tensor(remaining_music_seconds,device=self.qpos_mean.device,dtype=torch.float32).reshape(-1)
        if remaining.shape!=(batch,) or not torch.isfinite(remaining).all() or (remaining < 0).any():
            raise ValueError('Music remaining time must be finite nonnegative [B]')
        angle=f['relative_times'][...,None]*self.time_frequency
        times=torch.stack((angle.sin(),angle.cos()),-1).flatten(-2)
        flags=torch.stack((f['future_valid'],f['music_valid'],f['prefix_valid']),-1).float()
        temporal=self.music_projection(f['music'])+self.prefix_projection(f['prefix'])+times+self.mask_projection(flags)
        temporal=temporal+self.history_projection(f['history'])[:,None]
        temporal=torch.where(f['future_valid'][...,None],temporal,0.)
        token=self.value_token.expand(batch,-1,-1)+self.remaining_projection((remaining/30.)[:,None])[:,None]
        sequence=torch.cat((token,temporal),1)
        blocked=torch.cat((torch.zeros(batch,1,dtype=torch.bool,device=sequence.device),~f['future_valid']),1)
        value=self.value_head(self.temporal(sequence,src_key_padding_mask=blocked)[:,0]).squeeze(-1)
        if not torch.isfinite(value).all():raise FloatingPointError('Temporal Critic produced a nonfinite value')
        return value

    def forward(self, context, remaining_music_seconds):
        f=self.encode_frozen(context) if self.freeze_encoders else self._encode(context)
        return self.forward_encoded(f,remaining_music_seconds)
