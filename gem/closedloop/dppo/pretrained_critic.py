"""非对称三编码器Critic：冻结音乐/前缀，可训练的Stage1初始化历史跟踪GRU。

Actor原有条件完全不变。本网络持有独立参数和存储，以50×90真实跟踪历史及1维相对
时间替代原49维GRU输入。输入权重前48列、最后时间列、循环权重、bias和输出投影
从Stage1复制，新增42列严格零初始化。音乐/已承诺前缀编码冻结且可跨epoch缓存；
历史GRU每次forward_encoded重新计算梯度图，绝不缓存可训练历史特征。
旧schema保持三编码器结构；显式future_reference启用第四个轻量已提交未来参考编码器。
新分支只读取因果缓存及有效mask，按时间注意力汇聚后接入原Temporal Transformer。
最后投影零初始化，显式继承旧Critic共享参数时初始值逐元素一致，新增信息的有效性
必须在相同完整MC的独立困难状态上验证，不能把初始化或训练loss下降当作物理收益。
"""
from __future__ import annotations

import copy
import math

import torch
from torch import nn

from gem.closedloop.contracts import STAGE1_CONDITION_KEYS
from .tracking_context import validate_tracking_context, SCHEMA, FUTURE_SCHEMA


class FutureCommittedReferenceEncoder(nn.Module):
    """逐帧MLP和带有效mask的时间注意力；不缓存任何可训练编码结果。"""
    def __init__(self,dimension):
        super().__init__()
        self.frames=nn.Sequential(nn.Linear(39,128),nn.LayerNorm(128),nn.SiLU(),nn.Linear(128,128),nn.SiLU())
        self.attention=nn.Linear(128,1)
        self.projection=nn.Linear(128,dimension,bias=False)
        self.metadata=nn.Sequential(nn.Linear(20,64),nn.SiLU(),nn.Linear(64,dimension,bias=False))
        self.register_buffer('scales',torch.tensor([1.]*9+[2.]*3+[4.]*3+[3.14]*21+[2.,2.]))
        self.register_buffer('metadata_scales',torch.tensor([2.,1.,2.,1.,1.,1.,1.,1.,5.,100.]))
        nn.init.zeros_(self.projection.weight);nn.init.zeros_(self.metadata[-1].weight)

    def forward(self,f):
        valid=f['committed_valid'];x=f['committed_reference']/self.scales
        times=f['committed_relative'][...,None]
        h=self.frames(torch.cat((x,times),-1))
        logits=self.attention(h).squeeze(-1).masked_fill(~valid,-1e4)
        weights=logits.softmax(1)*valid
        weights=weights/weights.sum(1,keepdim=True).clamp_min(1e-12)
        pooled=(h*weights[...,None]).sum(1)
        meta=f['committed_metadata']/self.metadata_scales
        meta=torch.cat((meta*f['committed_metadata_valid'],f['committed_metadata_valid'].float()),-1)
        return self.projection(pooled)+self.metadata(meta)


class PretrainedConditionCritic(nn.Module):
    def __init__(self, actor, *, dimension=256, layers=2, heads=4, history_scales=(1.5,1.,1.,1.,.3,1.,1.5,3.14),future_reference=False):
        super().__init__()
        if dimension < 8 or dimension%heads or dimension%2 or layers < 1:
            raise ValueError('Invalid temporal Critic dimensions')
        self.dimension, self.cache_frozen_conditions = dimension, True
        self.context_schema=FUTURE_SCHEMA if future_reference else SCHEMA
        self.future_reference=bool(future_reference)
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
        if self.future_reference:self.future_reference_encoder=FutureCommittedReferenceEncoder(dimension)
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
        history=self.proprio_normalizer(c['tracking_history'][...,:48].float(),valid)
        extra=torch.where(valid[...,None],c['tracking_history'][...,48:].float(),0.)/self.tracking_scales
        relative=torch.where(valid,c['proprio_history_times']-c['decision_time'][:,None],0.).float()
        prefix=self.prefix_encoder(known,mask)
        result=dict(music=music,prefix=prefix,history_input=torch.cat((history,extra),-1),
            history_valid=valid,history_relative=relative,future_valid=future,
            music_valid=music_valid,prefix_valid=mask.any(-1),
            relative_times=torch.where(future,c['future_times']-c['decision_time'][:,None],0.).float())
        if self.future_reference:
            if 'committed_reference' not in c:raise ValueError('Future Critic cannot use legacy or reconstructed-after-action cache')
            result.update({k:c[k] for k in ('committed_reference','committed_valid','committed_metadata','committed_metadata_valid')})
            result['committed_relative']=(c['committed_times']-c['decision_time'][:,None]).float()
        return result

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
        future_feature=self.future_reference_encoder(f) if self.future_reference else None
        if future_feature is not None:temporal=temporal+future_feature[:,None]
        temporal=torch.where(f['future_valid'][...,None],temporal,0.)
        token=self.value_token.expand(batch,-1,-1)+self.remaining_projection((remaining/30.)[:,None])[:,None]
        if future_feature is not None:token=token+future_feature[:,None]
        sequence=torch.cat((token,temporal),1)
        blocked=torch.cat((torch.zeros(batch,1,dtype=torch.bool,device=sequence.device),~f['future_valid']),1)
        value=self.value_head(self.temporal(sequence,src_key_padding_mask=blocked)[:,0]).squeeze(-1)
        if not torch.isfinite(value).all():raise FloatingPointError('Temporal Critic produced a nonfinite value')
        return value

    def forward(self, context, remaining_music_seconds):
        f=self.encode_frozen(context)
        return self.forward_encoded(f,remaining_music_seconds)


def initialize_future_critic_shared(critic,legacy_state):
    """显式新schema初始化，只复制旧共享权重；不恢复旧Adam或冒充旧checkpoint恢复。"""
    if not critic.future_reference:raise ValueError('Explicit new future Critic required')
    current=critic.state_dict();shared={k for k in current if not k.startswith('future_reference_encoder.')}
    if set(legacy_state)!=shared or any(current[k].shape!=legacy_state[k].shape for k in shared):
        raise ValueError('Legacy Critic parameter mapping differs')
    critic.load_state_dict({**current,**legacy_state},strict=True)
    return dict(source_schema=SCHEMA,target_schema=FUTURE_SCHEMA,optimizer_restored=False,new_projection_zero=True)
