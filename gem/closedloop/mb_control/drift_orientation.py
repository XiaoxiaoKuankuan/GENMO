"""身体姿态是否相对根部的严格指标审计，原奖励不作任何改写。

从完整真实轨迹读取原55维状态和原始在线教师目标，使用真实机器人FK分别计算
世界身体姿态误差与根部坐标系身体姿态误差。报告只替换这一分组后的对照数值，
用于识别根yaw是否重复进入根部与身体两组。对照不作为已采用的新训练目标，
不能拿较低的重定义指标冒充原目标改善；当前正式及独立训练合同均保持原值。
"""
import argparse
import json
from pathlib import Path
import sys
import gc
import torch
from .drift_diagnostics import write_json


@torch.no_grad()
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--run',required=True);parser.add_argument('--output',required=True)
    args=parser.parse_args();run=Path(args.run);out=Path(args.output)
    cfg=json.loads((run/'manifest.json').read_text())['config'];sys.path.insert(0,cfg['supermimic']['repo'])
    from .application import build_components
    from .dance_losses import robust
    from supermimic.mdp import unflatten_state
    from supermimic.rotations import so3_log
    cfg['teacher']['enabled']=False;cfg['model']['compile_networks']=False
    actor,adapter,*_=build_components(cfg,'cpu');del actor
    kin=adapter.runtime.kin;torch.set_num_threads(2);rows=[]
    for version in [0,100,300,600,700]:
        folder=run/'evaluations/fixed_delay_replay'/('initial_genmo_baseline' if version==0 else f'actor_{version:06}')
        for source in folder.glob('shard_*/*_real_controls.pt'):
            payload=torch.load(source,map_location='cpu',weights_only=False,mmap=True)
            controls=[x['data'] for x in payload['controls']['data']]
            first=controls[0]['start']['data'];states=torch.cat([x['end']['data']['state'] for x in controls])
            ticks=torch.tensor([x['end']['data']['tick'] for x in controls]);teacher=first['teacher_reference']
            target=teacher['reference_state'][0,(ticks-int(teacher['ticks'][0]))//12]
            world=[];relative=[]
            for start in range(0,len(states),256):
                a=unflatten_state(states[start:start+256]);b=unflatten_state(target[start:start+256])
                _,ar,_,_=kin.forward(*a,all_bodies=True);_,br,_,_=kin.forward(*b,all_bodies=True)
                al=a[1].transpose(-1,-2)[:,None]@ar;bl=b[1].transpose(-1,-2)[:,None]@br
                world.append(robust(so3_log(br.transpose(-1,-2)@ar),cfg['rewards']['dance_scales']['body_orientation']).flatten(1).mean(-1))
                relative.append(robust(so3_log(bl.transpose(-1,-2)@al),cfg['rewards']['dance_scales']['body_orientation']).flatten(1).mean(-1))
            w=float(torch.cat(world).mean());r=float(torch.cat(relative).mean())
            summary=json.loads((source.parent/'physical_summary.json').read_text())
            original=next(x['physical_metrics_mean']['dance_loss'] for x in summary['physical_results'] if x['task']==first['task_id'])
            rows.append({'version':version,'task':first['task_id'],'world_body_orientation_group':w,
                'root_relative_body_orientation_group':r,'original_music_loss':original,
                'diagnostic_loss_replacing_only_body_orientation':original+(r-w)/8,
                'reward_contract_changed':False,'quality_improvement_claim':False})
            del payload,controls,states,target;gc.collect()
        write_json(out,rows)
    print(json.dumps({str(v):{k:sum(r[k] for r in rows if r['version']==v)/8 for k in
        ['world_body_orientation_group','root_relative_body_orientation_group','original_music_loss','diagnostic_loss_replacing_only_body_orientation']} for v in [0,100,300,600,700]},indent=2))


if __name__=='__main__':main()
