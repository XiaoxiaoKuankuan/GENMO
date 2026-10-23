"""非对称三编码器Critic：冻结音乐/前缀，可训练的Stage1初始化历史跟踪GRU。

Actor原有条件完全不变。本网络持有独立参数和存储，以50×90真实跟踪历史及1维相对
时间替代原49维GRU输入。输入权重前48列、最后时间列、循环权重、bias和输出投影
从Stage1复制，新增42列严格零初始化。音乐/已承诺前缀编码冻结且可跨epoch缓存；
历史GRU每次forward_encoded重新计算梯度图，绝不缓存可训练历史特征。
仍使用256维、2层、4头Temporal Transformer与Value Head，不增加第四个编码器。
"""
from __future__ import annotations

import copy
import math

import torch
from torch import nn

from gem.closedloop.contracts import STAGE1_CONDITION_KEYS
from .tracking_context import validate_tracking_context


class PretrainedConditionCritic(nn.Module):
    def __init__(self, actor, *, dimension=256, layers=2, heads=4, history_scales=(1.5,1.,1.,1.,.3,1.,1.5,3.14)):
        super().__init__()
        if dimension < 8 or dimension%heads or dimension%2 or layers < 1:
            raise ValueError('Invalid temporal Critic dimensions')
        self.dimension, self.cache_frozen_conditions = dimension, True
        for name in ('music_embedder','cond_exists_embedder','history_encoder','prefix_encoder','proprio_normalizer'):
            setattr(self,name,copy.deepcopy(getattr(actor,name)))
        original=self.history_encoder.cell
        expanded=nn.GRUCell(91,original.hidden_size,device=original.weight_ih.device,dtype=original.weight_ih.dtype)
        with torch.no_grad():
            expanded.weight_ih.zero_()
            expanded.weight_ih[:,:48].copy_(original.weight_ih[:,:48])
            expanded.weight_ih[:,90].copy_(original.weight_ih[:,48])
            for name in ('weight_hh','bias_ih','bias_hh'):
                getattr(expanded,name).copy_(getattr(original,name))
        self.history_encoder.cell=expanded
        self.history_encoder.execution_backend='native'
        self.history_encoder.requires_grad_(True)
        if len(history_scales)!=8 or any(not math.isfinite(v) or v<=0 for v in history_scales):
            raise ValueError('Eight explicit positive tracking physical scales required')
        self.register_buffer('tracking_scales',torch.cat([torch.full((n,),float(v)) for n,v in
            zip((3,1,21,2,3,6,3,3),history_scales)]))
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
        if self.cache_frozen_conditions:
            for module in self.encoder_modules():
                module.requires_grad_(False);module.eval()
        actor_parameters={id(p) for p in actor.parameters()}
        actor_storage={p.data_ptr() for p in actor.parameters()}
        if any(id(p) in actor_parameters or p.data_ptr() in actor_storage for p in self.parameters()):
            raise RuntimeError('Critic must own independent parameter storage')

    def encoder_modules(self):
        return tuple(m for m in (self.music_embedder,self.cond_exists_embedder,
            self.prefix_encoder,self.proprio_normalizer) if m is not None)

    def train(self, mode=True):
        super().train(mode)
        if self.cache_frozen_conditions:
            for module in self.encoder_modules():module.eval()
        return self

    def _encode(self, context):
        validate_tracking_context(context)
        device=self.qpos_mean.device
        c={k:v.to(device) for k,v in context.items()}
        future=c['future_valid'];extra=c['music_valid'].shape[1]-future.shape[1]
        if extra:
            # Critic保留完整音乐轴；新增尾部没有伪造运动或已知前缀。
            future=torch.cat((future,c['music_valid'][:,-extra:]),1)
            c['known_qpos30']=torch.nn.functional.pad(c['known_qpos30'],(0,0,0,extra))
            c['known_qpos30_mask']=torch.nn.functional.pad(c['known_qpos30_mask'],(0,0,0,extra))
            c['future_times']=c['future_times'][:,:1]+torch.arange(future.shape[1],device=device,dtype=torch.float64)[None]/30.
        music_valid=c['music_valid']&future
        mask=c['known_qpos30_mask']&future[...,None]
        physical=torch.where(mask,c['known_qpos30'].float(),0.)
        known=torch.where(mask,(physical-self.qpos_mean)/self.qpos_std,0.)
        music=self.music_embedder(torch.where(music_valid[...,None],c['music_features'].float(),0.))
        music=torch.where(music_valid[...,None],music,0.)
        if self.cond_exists_embedder is not None:
            music=self.cond_exists_embedder(torch.cat((music,music_valid[...,None].to(music)),-1))
        music=torch.where(future[...,None],music,0.)
        valid=c['proprio_history_valid']
        history=self.proprio_normalizer(c['tracking_history'][...,:48].float(),valid)
        extra=torch.where(valid[...,None],c['tracking_history'][...,48:].float(),0.)/self.tracking_scales
        relative=torch.where(valid,c['proprio_history_times']-c['decision_time'][:,None],0.).float()
        prefix=self.prefix_encoder(known,mask)
        return dict(music=music,prefix=prefix,history_input=torch.cat((history,extra),-1),
            history_valid=valid,history_relative=relative,future_valid=future,
            music_valid=music_valid,prefix_valid=mask.any(-1),
            relative_times=torch.where(future,c['future_times']-c['decision_time'][:,None],0.).float())

    @torch.no_grad()
    def encode_frozen(self, context):
        if not self.cache_frozen_conditions:
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
        history=self.history_encoder(f['history_input'],f['history_valid'],f['history_relative'])
        temporal=temporal+self.history_projection(history)[:,None]
        temporal=torch.where(f['future_valid'][...,None],temporal,0.)
        token=self.value_token.expand(batch,-1,-1)+self.remaining_projection((remaining/30.)[:,None])[:,None]
        sequence=torch.cat((token,temporal),1)
        blocked=torch.cat((torch.zeros(batch,1,dtype=torch.bool,device=sequence.device),~f['future_valid']),1)
        value=self.value_head(self.temporal(sequence,src_key_padding_mask=blocked)[:,0]).squeeze(-1)
        if not torch.isfinite(value).all():raise FloatingPointError('Temporal Critic produced a nonfinite value')
        return value

    def forward(self, context, remaining_music_seconds):
        f=self.encode_frozen(context)
        return self.forward_encoded(f,remaining_music_seconds)
