"""按接管阶段只读比较完整MC价值误差，检查全局MSE是否掩盖启动状态低估。

输入为旧/新完整checkpoint、当前配置、已经采到真实终点的MC训练/验证bank，以及
可选的三窗口优势明细。重建同结构Critic并加载真实权重，不执行优化；同一条物理
轨迹分别按旧gamma=.99和新gamma=.95重算MC，以各自正确目标评估两个Critic。
当前gamma标签必须与保存bank逐元素相同，独立验证的全局MSE还与原预热报告核对。

统计区分首次生成、接管0～2秒、2～20秒及20秒以后，并报告各组在MC监督与普通
20秒采集中的占比。分组用于揭示采样/误差分布，不重加权训练、不裁剪优势、不改
奖励和模型，也不把一次轨迹的MC噪声当作精确期望价值或同状态动作排序的真值。
本工具仅需一张空闲GPU做已保存条件的前向，所有输出独立保存，原数据只读。
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import gzip
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import torch

from gem.closedloop.dppo.budget import atomic_json
from gem.closedloop.dppo.complete_episode import complete_mc_returns
from gem.closedloop.dppo.pretrained_critic import PretrainedConditionCritic
from gem.closedloop.dppo.trainer import load_actor
from gem.closedloop.dppo.training_config import configuration


def phases(tick):
    elapsed=(tick-600)/600
    result=['all','takeover_0_2s' if elapsed<2 else 'takeover_2_20s' if elapsed<20 else 'takeover_20s_plus']
    if tick==600:result.append('first_generation')
    return result


def metrics(records):
    result=dict(states=len(records))
    for prefix,gamma in [('old',.99),('new',.95)]:
        error=torch.tensor([r[prefix+'_prediction']-r[prefix+'_mc'] for r in records],dtype=torch.float64)
        if not len(error):result[prefix]=None;continue
        result[prefix]=dict(gamma=gamma,mse=float(error.square().mean()),bias=float(error.mean()),
            rmse=float(error.square().mean().sqrt()),minimum_error=float(error.min()),maximum_error=float(error.max()))
    return result


@torch.no_grad()
def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--parent',type=Path,required=True)
    parser.add_argument('--current',type=Path,required=True)
    parser.add_argument('--session',type=Path,required=True)
    parser.add_argument('--advantage-states',type=Path)
    parser.add_argument('--output-dir',type=Path,required=True)
    args=parser.parse_args()
    busy=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
    if busy:raise RuntimeError('Value phase forward needs an unoccupied GPU allocation')
    args.output_dir.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(2);torch.cuda.set_device(0)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    config=configuration(args.config,genmo_repo=ROOT);config['runtime']['genmo_device']='cpu'
    actor,_,_=load_actor(config);options=config['stage10']['critic'];models={};payloads={}
    for key,path in [('old',args.parent),('new',args.current)]:
        payload=torch.load(path,map_location='cpu',weights_only=False,mmap=True);payloads[key]=payload
        model=PretrainedConditionCritic(actor,dimension=options['dimension'],layers=options['layers'],
            heads=options['heads'],history_scales=options['history_scales'])
        model.load_state_dict(payload['critic'],strict=True);models[key]=model.eval().cuda()
    if payloads['old']['actor_fingerprint']!=payloads['new']['actor_fingerprint']:
        raise ValueError('Phase comparison requires the same unchanged behavior Actor')
    del actor
    records=[]
    for bank in ['preheat_train','preheat_val']:
        for path in sorted((args.session/'mc_calibration'/bank).glob('rank*/complete_mc.pt')):
            data=torch.load(path,map_location='cpu',weights_only=False,mmap=True)
            rows=data['rows'];targets={};offset=0
            for episode in data['episodes']:
                part=rows[offset:offset+episode['states']]
                objects=[SimpleNamespace(**{k:v for k,v in r.items() if k not in ('begin','end')},
                    control_tick_begin=r['begin'],control_tick_end=r['end'],transition_valid=True) for r in part]
                old=complete_mc_returns(objects,gamma_upper=.99)
                new=complete_mc_returns(objects,gamma_upper=.95)
                if not torch.equal(new,data['returns'][offset:offset+len(part)]):
                    raise ValueError('Reconstructed current MC differs from saved actual labels')
                for i,row in enumerate(part):targets[offset+i]=(float(old[i]),float(new[i]),episode['task']['dataset'],episode['task']['task_id'])
                offset+=len(part)
            if offset!=len(rows):raise ValueError('Complete bank row boundaries differ')
            for start in range(0,len(rows),128):
                part=rows[start:start+128]
                context={k:torch.cat([r['metadata']['critic_context'][k] for r in part]).cuda()
                    for k in part[0]['metadata']['critic_context']}
                remaining=torch.tensor([r['metadata']['remaining_music_seconds'] for r in part],device='cuda')
                values={key:model(context,remaining).cpu().tolist() for key,model in models.items()}
                for j,row in enumerate(part):
                    old,new,source,task=targets[start+j]
                    records.append(dict(bank=bank,source=source,task_id=task,tick=row['begin'],
                        old_mc=old,new_mc=new,old_prediction=values['old'][j],new_prediction=values['new'][j]))
    grouped={}
    for bank in ['preheat_train','preheat_val']:
        selected=[r for r in records if r['bank']==bank];groups=defaultdict(list)
        for row in selected:
            for phase in phases(row['tick']):groups[phase].append(row)
        grouped[bank]={key:dict(metrics(part),state_fraction=len(part)/len(selected)) for key,part in groups.items()}
    expected=json.loads((args.session/'mc_calibration/preheat.json').read_text())['after']['mse']
    actual=grouped['preheat_val']['all']['new']['mse']
    if abs(expected-actual)>1e-3:raise ValueError('Reloaded Critic does not reproduce recorded validation MSE')
    rollout=defaultdict(int)
    if args.advantage_states:
        with gzip.open(args.advantage_states,'rt',encoding='utf8') as stream:
            for line in stream:
                row=json.loads(line)
                for phase in phases(row['control_tick_begin']):rollout[phase]+=1
    result=dict(mc=grouped,rollout_stage_counts=dict(rollout),validation_mse_reproduction=dict(expected=expected,actual=actual),
        worst_new_underestimates=sorted(records,key=lambda r:r['new_prediction']-r['new_mc'])[:20],
        scope='same_real_complete_episodes_correct_gamma_per_critic_no_model_updates_no_causal_action_ranking')
    if rollout:
        result['rollout_stage_fractions']={k:v/rollout['all'] for k,v in rollout.items()}
    with gzip.open(args.output_dir/'mc_states.jsonl.gz','wt',encoding='utf8') as stream:
        for row in records:stream.write(json.dumps(row,ensure_ascii=False)+'\n')
    atomic_json(args.output_dir/'summary.json',result)
    print(json.dumps(result,ensure_ascii=False))


if __name__=='__main__':main()
