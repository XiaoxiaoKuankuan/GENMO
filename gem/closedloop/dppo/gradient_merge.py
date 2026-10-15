"""PPO与BC的单次分桶SUM通信、真实全局梯度诊断及有界平滑BC标定。

两项局部loss都按全局样本分母计算。PPO和未加权BC梯度分别累计，在同一桶中
并列SUM一次，再得到全局范数、夹角和模块贡献，最后以统一系数组合、统一裁剪。
没有先同步PPO再清空非root副本，也没有默认平均梯度后的额外world_size除法。
诊断以真正SUM后的向量计算，不能把各rank局部范数的和冒充全局梯度范数。

BC比例标定只读取共同作用参数；contact等单独头单列。目标系数由范数比例得到，
用指数平滑、显式上下限及每步最大变化倍率控制，不按每个microbatch跳变。
该系数是梯度合成系数，不等于Adam参数更新贡献。控制器状态属于新checkpoint
与整轮回滚状态。实际通信字节、桶数和时间被报告，不隐瞒双向量诊断的额外流量。
归约后梯度直接持有本次bucket的独立视图，bucket不跨步复用；消除逐参数clone而
不改变SUM、FP64统计、合成顺序或参数更新，p.grad持有存储直到下一次zero_grad。
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


def merge_global_gradients(actor,ppo,bc,controller,*,distributed=None,bucket_bytes=32*1024**2,
                           half_split=False,sketch=None,sketch_budget=1<<20):
    """PPO与BC梯度一次SUM合并；可选梯度信噪比诊断不改变p.grad。

    half_split在同一bucket增加第三行，只放前一半rank的PPO梯度，归约后得到两半rank
    独立半批梯度的余弦。sketch按固定步长抽取全局PPO梯度坐标并跨optimizer步累加，
    供相邻两轮比较方向；抽样步长只由参数列表决定。
    """
    parameters=[(n,p) for n,p in actor.named_parameters() if p.requires_grad]
    device=parameters[0][1].device
    split=bool(half_split and distributed is not None and distributed.world_size>1)
    rows=3 if split else 2
    stride=max(1,math.ceil(sum(p.numel() for _,p in parameters)/sketch_budget))
    half=torch.zeros(3,dtype=torch.float64,device=device)
    presence=torch.tensor([[n in ppo,n in bc] for n,_ in parameters],dtype=torch.int32,device=device)
    if distributed is not None:dist.all_reduce(presence,op=dist.ReduceOp.SUM,group=distributed.tensor_group)
    used=presence.cpu().tolist()
    groups={name:torch.zeros(3,dtype=torch.float64,device=device) for name in ['all','shared',*sorted({_group(n) for n,_ in parameters})]}
    buckets=[];pending=[];size=0
    for (name,p),flags in zip(parameters,used):
        if not any(flags):p.grad=None;continue
        required=rows*p.numel()*p.element_size()
        if pending and size+required>bucket_bytes:buckets.append(pending);pending=[];size=0
        pending.append((name,p,flags));size+=required
    if pending:buckets.append(pending)
    # 默认只保留PPO/BC两项梯度；SNR诊断时第三行仅是bucket内的前半rank PPO副本。
    start=time.perf_counter();traffic=0
    for bucket in buckets:
        count=sum(p.numel() for _,p,_ in bucket)
        packed=torch.empty(rows,count,device=device,dtype=torch.float32)
        offset=0
        for name,p,flags in bucket:
            view=packed[:,offset:offset+p.numel()]
            view[0].zero_() if name not in ppo else view[0].copy_(ppo[name].reshape(-1))
            view[1].zero_() if name not in bc else view[1].copy_(bc[name].reshape(-1))
            if split:
                first=distributed.rank<distributed.world_size//2 and name in ppo
                view[2].copy_(ppo[name].reshape(-1)) if first else view[2].zero_()
            offset+=p.numel()
        if distributed is not None:dist.all_reduce(packed,op=dist.ReduceOp.SUM,group=distributed.tensor_group)
        traffic+=packed.numel()*packed.element_size()
        offset=0
        for name,p,flags in bucket:
            a,b=packed[:2,offset:offset+p.numel()]
            if split:
                first=packed[2,offset:offset+p.numel()].double();second=a.double()-first
                half+=torch.stack((first.square().sum(),second.square().sum(),(first*second).sum()))
            if sketch is not None and flags[0]:
                sample=a[::stride].double()
                sketch[name]=sketch[name]+sample if name in sketch else sample.clone()
            numbers=torch.stack((a.double().square().sum(),b.double().square().sum(),(a.double()*b.double()).sum()))
            groups['all']+=numbers;groups[_group(name)]+=numbers
            groups['shared']+=numbers*((numbers[0]>0)&(numbers[1]>0))
            # 明确保存全局未加权向量，BC控制只在全部桶统计完成后更新一次。
            ppo[name]=a.reshape_as(p) if flags[0] else None
            bc[name]=b.reshape_as(p) if flags[1] else None
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
    if split:
        first2,second2,dot=half.cpu().tolist()
        reports['half_split']=dict(first_half_norm=math.sqrt(first2),second_half_norm=math.sqrt(second2),dot=dot,
            first_half_square=first2,second_half_square=second2,
            cosine=None if first2<=0 or second2<=0 else max(-1.,min(1.,dot/math.sqrt(first2*second2))),
            split='ranks_below_half_world_vs_rest')
    reports.update(bc_weight=weight,bc_weight_is_adam_contribution=False,
        communication=dict(bucket_count=len(buckets),local_reduced_bytes=traffic,
            wall_seconds=time.perf_counter()-start,gradient_reduction='single_paired_bucket_SUM_no_world_average'))
    ppo.clear();bc.clear()
    return reports
