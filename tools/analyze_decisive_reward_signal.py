"""审计真实完整物理轨迹中八项跟踪奖励是否在严重错误处饱和。

读取SHA绑定评估控制档案与明确的Stage2奖励配置，逐50Hz步用原平方误差和
已声明reward_std重建exp评分；按全轨迹、100秒以后、失败前五秒和数据来源统计。
同时保留实际误差RMS、低于1e-4评分的时间占比及原权重，不调整奖励或采样。
这里只诊断现有标量奖励对错误改善的敏感性，不把指数函数的导数当成可反传物理
梯度，也不因某项饱和就宣称所有RL优势为零；失败事件与其他奖励仍可能传递信息。
"""
from __future__ import annotations
import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import sys
import yaml

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from gem.closedloop.dppo.compact_evidence import report_episodes
from gem.closedloop.dppo.budget import atomic_json


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,required=True);parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    if args.output.exists():raise FileExistsError(args.output)
    config=yaml.safe_load(args.config.read_text());profile=config['stage10']['reward'];tracking=profile['tracking']
    report_path=args.run/'evaluation_baseline.json';report=json.loads(report_path.read_text())
    groups=defaultdict(lambda:defaultdict(lambda:dict(seconds=0.,error_integral=0.,score_integral=0.,saturated_seconds=0.)))
    failures=[]
    for episode in report_episodes(report):
        controls=[c for d in episode['decisions'] for c in d['controls']];first=controls[0]['tick'];last=controls[-1]['tick']
        for control in controls:
            elapsed=(control['tick']-first+12)/600
            labels=['all','source:'+episode['dataset']]
            if elapsed+episode['music_start_frame']/30>=100:labels.append('after100')
            if episode['physical_failure'] and last-control['tick']<3000:labels.append('before_failure_5s')
            for name,std in tracking['reward_std'].items():
                error=control['raw_errors'][name+'_squared_error'];score=math.exp(-error/std**2);dt=control['dt_s']
                for label in labels:
                    item=groups[label][name];item['seconds']+=dt;item['error_integral']+=dt*error;item['score_integral']+=dt*score
                    item['saturated_seconds']+=dt*(score<1e-4)
        if episode['physical_failure']:
            last_errors=controls[-1]['raw_errors']
            failures.append(dict(task_id=episode['task_id'],dataset=episode['dataset'],sample_id=episode['sample_id'],seed=episode['seed'],
                reason=episode['end_reason'],elapsed_seconds=(last-first+12)/600,
                last_errors=last_errors,last_root_position_score=math.exp(-last_errors['anchor_pos_squared_error']/tracking['reward_std']['anchor_pos']**2),
                last_root_orientation_score=math.exp(-last_errors['anchor_ori_squared_error']/tracking['reward_std']['anchor_ori']**2)))
    result={}
    for label,terms in groups.items():
        result[label]={name:dict(seconds=item['seconds'],rms=math.sqrt(item['error_integral']/item['seconds']),
            mean_score=item['score_integral']/item['seconds'],score_below_1e4_fraction=item['saturated_seconds']/item['seconds'],
            normalized_weight=tracking['weights'][name],maximum_reward_per_second=profile['track_weight']*tracking['weights'][name]) for name,item in terms.items()}
    output=dict(status='passed',groups=result,failures=failures,reward_profile=profile,
        report_path=str(report_path),report_sha256=hashlib.sha256(report_path.read_bytes()).hexdigest(),
        config_path=str(args.config),config_sha256=hashlib.sha256(args.config.read_bytes()).hexdigest(),
        scope='observed_reward_sensitivity_not_policy_gradient_or_physics_derivative')
    atomic_json(args.output,output);print(json.dumps({k:output[k] for k in ('status','failures')},ensure_ascii=False))


if __name__=='__main__':main()
