"""只读拆解已保存CPU/GPU物理重放中的奖励差异，不重新仿真或修改验收容限。

逐rank核验相同请求顺序和实际控制tick，用同一份现行ExecutionReward计算两侧完整
奖励；额外对同一CPU证据重复计算，确认差异不来自不确定奖励实现。分解积分奖励、
活动门控、节拍和强度，记录哪些因果窗口的离散动作节拍数量发生变化。输入绑定SHA，
输出明确区分物理输入变化与奖励公式变化；不能凭此把先前未通过的容限自动改为通过。
本工具只在服务器1读取既有八卡结果，不创建GPU上下文、不启动训练、不改原报告。
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from tools.train_closedloop_stage10 import configuration
from tools.replay_stage10_runtime_v4 import reward_rows
from gem.closedloop.dppo.full_dataset import FullMusicCatalog,SOURCES
from gem.closedloop.dppo.rollback_audit import exact_state_digest


def stats(values):
    a=np.asarray(values,dtype=np.float64)
    return dict(rms=float(np.sqrt(np.mean(a*a))),max_abs=float(np.max(np.abs(a))),sum=float(a.sum()))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('cpu-replay','gpu-replay','data-audit','output'):parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():raise FileExistsError(args.output)
    torch.set_num_threads(1)
    reports=[]
    for rank in range(8):
        cpu_path=args.cpu_replay/f'rank{rank:02d}/baseline/requests_and_full_replies.pt'
        gpu_path=args.gpu_replay/f'rank{rank:02d}/gpu_requests_and_full_replies.pt'
        cfg=configuration(cpu_path.parent/'resolved_config.yaml')
        catalog=FullMusicCatalog(cfg['paths']['data_root']);catalog.apply_audit(json.loads(args.data_audit.read_text()))
        sample=catalog.samples['val'][SOURCES[rank%4]][0];music=catalog.load_music(sample)
        gpu=torch.load(gpu_path,map_location='cpu',weights_only=False)
        cpu=[r for r in torch.load(cpu_path,map_location='cpu',weights_only=False) if not r.get('remote_error')][:len(gpu)]
        if [r['method'] for r in cpu]!=[r['method'] for r in gpu]:raise ValueError('Different replay request order')
        ticks=lambda records:[s['tick'] for r in records if r['method']=='advance' for s in r['result']['trace']]
        if ticks(cpu)!=ticks(gpu):raise ValueError('Different physical control times')
        left,right=reward_rows(cpu,cfg,sample,music),reward_rows(gpu,cfg,sample,music)
        repeat=reward_rows(cpu,cfg,sample,music)
        if exact_state_digest(left)!=exact_state_digest(repeat):raise AssertionError('Same-input reward computation differs')
        components={name:stats([a['components'][name]['integrated_reward']-b['components'][name]['integrated_reward']
            for a,b in zip(left,right)]) for name in left[0]['components']}
        music_diff={}
        for key in ('beat_alignment','intensity_alignment','motion_beat_count'):
            values=[(a['components']['music']['raw'].get(key),b['components']['music']['raw'].get(key)) for a,b in zip(left,right)]
            valid=[(a,b) for a,b in values if a is not None and b is not None]
            music_diff[key]=dict(valid_windows=len(valid),changed_windows=sum(a!=b for a,b in valid),
                **stats([a-b for a,b in valid]))
        reports.append(dict(rank=rank,source=SOURCES[rank%4],controls=len(left),same_input_reward_exact=True,
            cpu_sha256=hashlib.sha256(cpu_path.read_bytes()).hexdigest(),gpu_sha256=hashlib.sha256(gpu_path.read_bytes()).hexdigest(),
            cpu_reward=sum(r['reward'] for r in left),gpu_reward=sum(r['reward'] for r in right),
            total=stats([a['reward']-b['reward'] for a,b in zip(left,right)]),components=components,music=music_diff))
    args.output.write_text(json.dumps(dict(schema='stage10.replay_reward_diagnosis.v1',ranks=reports,
        original_thresholds_unchanged=True,original_acceptance_report_unchanged=True,
        scope='same_reward_code_different_CPU_GPU_physical_inputs_not_policy_quality'),ensure_ascii=False,indent=2)+'\n')


if __name__=='__main__':main()
