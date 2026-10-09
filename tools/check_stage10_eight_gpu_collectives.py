"""服务器1八卡通信最小复现，隔离CUDA/NCCL传输与Actor计算。

本工具只在显式八卡torchrun中运行，启动前确认显卡空闲。依次测试训练使用的
FP64 MAX、int32梯度presence SUM及FP32梯度bucket SUM，每次同步并验证精确值，
记录张量形状、设备、NCCL/PyTorch版本和环境开关。它不改变训练参数、协议或
概率阈值，不调用GMT，也不把通信故障误报为Actor概率不一致。
"""
import json
import os
from pathlib import Path
import sys
from datetime import timedelta
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
import torch.distributed as dist
from tools.train_closedloop_stage10_8gpu import _available_gpus


def main():
    rank,world,local=(int(os.environ.get(k,'-1')) for k in ('RANK','WORLD_SIZE','LOCAL_RANK'))
    if world!=8 or rank!=local:raise ValueError('Exactly eight local GPU ranks required')
    dist.init_process_group('gloo',timeout=timedelta(minutes=2))
    status=[None]
    if rank==0:
        try:status[0]=dict(devices=_available_gpus())
        except Exception as error:status[0]=dict(error=str(error))
    dist.broadcast_object_list(status,0)
    if 'error' in status[0]:raise RuntimeError(status[0]['error'])
    torch.cuda.set_device(rank)
    group=dist.new_group(backend='nccl',timeout=timedelta(minutes=2))
    if rank==0:print(json.dumps(dict(torch=torch.__version__,nccl=torch.cuda.nccl.version(),
        env={k:v for k,v in os.environ.items() if k.startswith('NCCL_')})),flush=True)
    for dtype,size,op in [(torch.float64,3,dist.ReduceOp.MAX),*(
            (torch.int32,n,dist.ReduceOp.SUM) for n in (1,32,128,256,512,1024)),
            (torch.float32,8*1024*1024,dist.ReduceOp.SUM)]:
        value=torch.full((size,),rank+1,device=f'cuda:{rank}',dtype=dtype)
        dist.all_reduce(value,op=op,group=group);torch.cuda.synchronize()
        expected=8 if op==dist.ReduceOp.MAX else 36
        if not bool((value==expected).all()):raise ValueError('Collective produced incorrect data')
        if rank==0:print(json.dumps(dict(dtype=str(dtype),elements=size,operation=str(op),passed=True)),flush=True)
    dist.destroy_process_group(group);dist.destroy_process_group()


if __name__=='__main__':main()
