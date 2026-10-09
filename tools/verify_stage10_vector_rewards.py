"""服务器1八卡对真实GPU物理证据做批量连续奖励数值与有限性能验收。

读取已有八个真实固定参考重放，不重新仿真、不改变任何物理数据。将不同重放中的
同一控制位置组成N1/2/4/8批次，在GPU计算七个连续奖励分项，逐字段对照原CPU
ExecutionReward的raw、normalized、score和valid，保留atol1e-8/rtol1e-7。
该工具只证明同输入公式等价，不把已有CPU/GPU轨迹差异或复制数据当成采样提速。
测速固定1200个真实记录，数据传输、算子执行和CPU证据展开区分；不发布训练模型。
"""
from __future__ import annotations
import argparse
from datetime import timedelta
import json
import os
from pathlib import Path
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
import torch.distributed as dist
from tools.train_closedloop_stage10 import configuration
from tools.train_closedloop_stage10_8gpu import _available_gpus
from gem.closedloop.dppo.distributed_runtime import DistributedCollectives
from gem.closedloop.dppo.parallel_support import root_call,local_call
from gem.closedloop.dppo.rewards import ExecutionReward
from gem.closedloop.dppo.vector_reward_math import VectorRewardMath


def stack_records(values, device, key=''):
    first=values[0]
    if isinstance(first,dict):return {name:stack_records([v[name] for v in values],device,name) for name in first}
    if isinstance(first,np.ndarray) or key in ('error','source_score','control_tick','reference_tick','physics_tick','env_index'):
        return torch.as_tensor(np.stack(values),device=device)
    if isinstance(first,(int,float,np.generic)) and any(value!=first for value in values):
        return torch.as_tensor(np.asarray(values),device=device)
    if any(str(value)!=str(first) for value in values):raise ValueError(f'Unexpected variable metadata {key}')
    return first


def compare(actual, expected, path, errors):
    if isinstance(expected,dict):
        if set(actual)!=set(expected):raise AssertionError(f'{path} fields differ')
        for key in expected:compare(actual[key],expected[key],f'{path}.{key}',errors)
    elif isinstance(expected,(list,tuple)) and expected and isinstance(expected[0],dict):
        if len(actual)!=len(expected):raise AssertionError(f'{path} length differs')
        for index,(a,b) in enumerate(zip(actual,expected)):compare(a,b,f'{path}.{index}',errors)
    elif expected is None or isinstance(expected,str) or (isinstance(expected,(list,tuple)) and expected and isinstance(expected[0],str)):
        if actual!=expected:raise AssertionError(f'{path} metadata differs')
    else:
        a,b=np.asarray(actual),np.asarray(expected)
        np.testing.assert_allclose(a,b,atol=1e-8,rtol=1e-7,err_msg=path)
        if a.size:errors[path]=max(errors.get(path,0.),float(np.max(np.abs(a.astype(float)-b.astype(float)))))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('config','replay','gmt-repo','output'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();rank=int(os.environ['RANK'])
    if int(os.environ['WORLD_SIZE'])!=8:raise ValueError('Eight Server1 GPUs required')
    dist.init_process_group('gloo',timeout=timedelta(minutes=15));group=DistributedCollectives(rank,8,device='cpu')
    root_call(group,_available_gpus)
    root_call(group,lambda:None if not a.output.exists() else (_ for _ in ()).throw(FileExistsError(a.output)))
    torch.cuda.set_device(rank);torch.set_num_threads(1)
    sys.path.insert(0,str(a.gmt_repo/'source/NoetixRobot/NoetixRobot/tasks/mimic/mimic_noetix_bumi4340_mha_sonic/closedloop'))
    from vector_diagnostics import to_host,select_env
    cfg=configuration(a.config)['stage9']['reward'];device=f'cuda:{rank}'
    sequences=[]
    for index in range(8):
        records=torch.load(a.replay/f'rank{(rank+index)%8:02d}/gpu_requests_and_full_replies.pt',map_location='cpu',weights_only=False)
        sequences.append([row for record in records if record['method']=='advance' for row in record['result']['trace']])
    count=min(map(len,sequences));report=dict(rank=rank,status='running',controls_per_sequence=count,cases=[])
    def run():
        for batch in (1,2,4,8):
            engine=VectorRewardMath(cfg,batch,device);scalar=[ExecutionReward(cfg) for _ in range(batch)]
            errors={};gpu_seconds=cpu_seconds=transfer_seconds=0.
            for step in range(count):
                rows=[seq[step] for seq in sequences[:batch]]
                began=time.perf_counter()
                tracking=stack_records([r['motion_tracking'] for r in rows],device)
                substeps=[stack_records([r['physics_substeps'][s]['physical_diagnostics'] for r in rows],device) for s in range(4)]
                for d in substeps:d['env_index']=torch.arange(batch,device=device)
                arguments=dict(tracking=tracking,substeps=substeps,
                    errors={key:torch.tensor([r['errors'][key] for r in rows],device=device,dtype=torch.float64) for key in rows[0]['errors']},
                    actual_joint_pos=torch.as_tensor(np.stack([r['actual_joint_pos_gmt'] for r in rows]),device=device),
                    reference_joint_pos=torch.as_tensor(np.stack([r['reference']['joint_pos'] for r in rows]),device=device),
                    terminated=torch.tensor([r['terminated'] for r in rows],device=device),
                    active=torch.ones(batch,dtype=torch.bool,device=device),ticks=torch.tensor([r['tick'] for r in rows],device=device))
                torch.cuda.synchronize();transfer_seconds+=time.perf_counter()-began
                began=time.perf_counter();value=engine.compute(**arguments);torch.cuda.synchronize()
                gpu_seconds+=time.perf_counter()-began
                value=to_host(value)
                for index,row in enumerate(rows):
                    result=select_env(value,index)['components']
                    if result['cmd']['raw']['first_step']:result['cmd']['raw']['previous_joint_position_target_rad']=None
                    began=time.perf_counter()
                    original={name:getattr(scalar[index],'_'+name)(row) for name in result}
                    cpu_seconds+=time.perf_counter()-began
                    for name,(score,valid,raw,normalized) in original.items():
                        compare(result[name],dict(score=score,valid=valid,raw=raw,normalized=normalized),name,errors)
            report['cases'].append(dict(batch=batch,controls=count*batch,passed=True,maximum_field_error=max(errors.values()),
                field_errors=errors,gpu_math_seconds=gpu_seconds,cpu_scalar_seconds=cpu_seconds,
                input_transfer_seconds=transfer_seconds,scope='same_saved_physics_seven_continuous_terms_not_training_throughput'))
    try:local_call(group,run);report['status']='passed'
    except BaseException as error:report.update(status='failed',error=str(error));raise
    finally:
        records=group.all_gather_object(report)
        if rank==0:a.output.write_text(json.dumps(dict(status='passed' if all(r['status']=='passed' for r in records) else 'failed',ranks=records),ensure_ascii=False,indent=2)+'\n')
        dist.destroy_process_group()


if __name__=='__main__':main()
