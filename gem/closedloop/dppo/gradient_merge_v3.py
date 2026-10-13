"""PPO与BC的单次分桶SUM通信、真实全局梯度诊断及有界平滑BC标定。

两项局部loss都按全局样本分母计算。PPO和未加权BC梯度分别累计，在同一桶中
并列SUM一次，再得到全局范数、夹角和模块贡献，最后以统一系数组合、统一裁剪。
没有先同步PPO再清空非root副本，也没有默认平均梯度后的额外world_size除法。
诊断以真正SUM后的向量计算，不能把各rank局部范数的和冒充全局梯度范数。

BC比例标定只读取共同作用参数；contact等单独头单列。目标系数由范数比例得到，
用指数平滑、显式上下限及每步最大变化倍率控制，不按每个microbatch跳变。
该系数是梯度合成系数，不等于Adam参数更新贡献。控制器状态属于新checkpoint
与整轮回滚状态。实际通信字节、桶数和时间被报告，不隐瞒双向量诊断的额外流量。
"""
from __future__ import annotations

import math
import time

import torch
import torch.distributed as dist


class BCGradientController:
    def __init__(self, *, initial=.1,target_ratio=.1,minimum=1e-4,maximum=10.,smoothing=.9,max_change=2.,mode='bounded_ratio'):
        if not 0<=minimum<=initial<=maximum or target_ratio<=0 or not 0<=smoothing<1 or max_change<1:
            raise ValueError('Invalid bounded BC gradient calibration')
        if mode not in ('fixed','bounded_ratio'):raise ValueError('Unknown BC calibration mode')
        self.value,self.target_ratio,self.minimum,self.maximum=initial,target_ratio,minimum,maximum
        self.smoothing,self.max_change,self.mode=smoothing,max_change,mode
        self.updates=0

    def update(self,ppo_norm,bc_norm):
        old=self.value
        if self.mode=='bounded_ratio' and ppo_norm>0 and bc_norm>1e-20:
            target=min(self.maximum,max(self.minimum,self.target_ratio*ppo_norm/bc_norm))
            proposed=self.smoothing*old+(1-self.smoothing)*target
            self.value=min(self.maximum,old*self.max_change,max(self.minimum,old/self.max_change,proposed))
        self.updates+=1
        return self.value

    def state_dict(self):return dict(self.__dict__)
    def load_state_dict(self,state):
        for key in ['target_ratio','minimum','maximum','smoothing','max_change','mode']:
            if state[key]!=getattr(self,key):raise ValueError('BC controller contract differs from checkpoint')
        if not self.minimum<=state['value']<=self.maximum or state['updates']<0:
            raise ValueError('Invalid BC controller state')
        self.value,self.updates=state['value'],state['updates']


def take_gradients(module):
    """转移局部梯度所有权，不额外复制整模型；后续backward建立新的grad。"""
    result={}
    for name,p in module.named_parameters():
        if p.requires_grad and p.grad is not None:result[name]=p.grad.detach();p.grad=None
    return result


def _group(name):
    if name.startswith('denoiser.blocks.'):return 'transformer_backbone'
    if name.startswith('denoiser.final_layer.'):return 'qpos_output'
    if 'static_conf' in name or 'contact' in name:return 'contact_head'
    return name.split('.')[0]


def merge_global_gradients(actor,ppo,bc,controller,*,distributed=None,bucket_bytes=32*1024**2):
    parameters=[(n,p) for n,p in actor.named_parameters() if p.requires_grad]
    device=parameters[0][1].device
    presence=torch.tensor([[n in ppo,n in bc] for n,_ in parameters],dtype=torch.int32,device=device)
    if distributed is not None:dist.all_reduce(presence,op=dist.ReduceOp.SUM,group=distributed.tensor_group)
    used=presence.cpu().tolist()
    groups={name:torch.zeros(3,dtype=torch.float64,device=device) for name in ['all','shared',*sorted({_group(n) for n,_ in parameters})]}
    buckets=[];pending=[];size=0
    for (name,p),flags in zip(parameters,used):
        if not any(flags):p.grad=None;continue
        required=2*p.numel()*p.element_size()
        if pending and size+required>bucket_bytes:buckets.append(pending);pending=[];size=0
        pending.append((name,p,flags));size+=required
    if pending:buckets.append(pending)
    # 一轮只保留总共两项梯度，不再创建第三份完整未裁剪副本。
    start=time.perf_counter();traffic=0
    for bucket in buckets:
        count=sum(p.numel() for _,p,_ in bucket)
        packed=torch.empty(2,count,device=device,dtype=torch.float32)
        offset=0
        for name,p,flags in bucket:
            view=packed[:,offset:offset+p.numel()]
            view[0].zero_() if name not in ppo else view[0].copy_(ppo[name].reshape(-1))
            view[1].zero_() if name not in bc else view[1].copy_(bc[name].reshape(-1))
            offset+=p.numel()
        if distributed is not None:dist.all_reduce(packed,op=dist.ReduceOp.SUM,group=distributed.tensor_group)
        traffic+=packed.numel()*packed.element_size()
        offset=0
        for name,p,flags in bucket:
            a,b=packed[:,offset:offset+p.numel()]
            numbers=torch.stack((a.double().square().sum(),b.double().square().sum(),(a.double()*b.double()).sum()))
            groups['all']+=numbers;groups[_group(name)]+=numbers
            groups['shared']+=numbers*((numbers[0]>0)&(numbers[1]>0))
            # 明确保存全局未加权向量，BC控制只在全部桶统计完成后更新一次。
            ppo[name]=a.reshape_as(p).clone() if flags[0] else None
            bc[name]=b.reshape_as(p).clone() if flags[1] else None
            offset+=p.numel()
    reports={}
    for name,(a2,b2,dot) in zip(groups,torch.stack(list(groups.values())).cpu().tolist()):
        reports[name]=dict(ppo_norm=math.sqrt(a2),unweighted_bc_norm=math.sqrt(b2),
            cosine=None if a2<=0 or b2<=0 else max(-1.,min(1.,dot/math.sqrt(a2*b2))))
    shared=reports['shared'];weight=controller.update(shared['ppo_norm'],shared['unweighted_bc_norm'])
    for name,p in parameters:
        a,b=ppo.get(name),bc.get(name)
        if a is None and b is None:p.grad=None
        elif a is None:p.grad=b.mul_(weight)
        elif b is None:p.grad=a
        else:p.grad=a.add_(b,alpha=weight)
    for item in reports.values():item['weighted_bc_norm']=weight*item['unweighted_bc_norm']
    reports.update(bc_weight=weight,bc_weight_is_adam_contribution=False,
        communication=dict(bucket_count=len(buckets),local_reduced_bytes=traffic,
            wall_seconds=time.perf_counter()-start,gradient_reduction='single_paired_bucket_SUM_no_world_average'))
    ppo.clear();bc.clear()
    return reports
