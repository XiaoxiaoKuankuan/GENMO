"""汇总完整P0门槛之后的固定Stage1参考实验C，并验收真实消费拼接连续性。

读取每个候选的独立PhysX报告及原始控制侧车，复算配对奖励、八项误差、活动度和
音乐聚类区间；30Hz提交前检查仅作第一层，最终还必须检查实际50Hz中心参考的
跨计划关节/根姿态/全身FK速度加速度增量。原有超限与新增超限分开报告。
20秒结果只作为完整困难音乐的准入依据，不把局部收益当作长期收益；完整音乐
超出基线失败点的部分不伪造固定参考配对。多个方向仍为探索实验，不作显著性宣称。
"""
from __future__ import annotations
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
import sys
import yaml

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from tools.summarize_decisive_physics import compare_one
from gem.closedloop.dppo.budget import atomic_json
from gem.robots.bumi.kinematics import BumiKinematics


def consumed_legality(result):
    c=result['executed_reference_checks'];m=c['maxima']
    return (c['new_limit_coordinate_count']==0 and m['joint_velocity_delta_max']<=.6
        and m['joint_acceleration_delta_max']<=20 and m['body_position_velocity_delta_max']<=.25
        and m['body_position_acceleration_delta_max']<=12 and m['root_orientation_velocity_delta_max']<=.25
        and m['root_orientation_acceleration_delta_max']<=3)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True)
    p.add_argument('--workers',type=int,default=4);a=p.parse_args()
    source=json.loads((a.run/'reference_c/summary.json').read_text())
    if source['status']!='passed' or not all(r['zero_gate']['passed'] for r in source['results']):
        raise ValueError('No complete zero-control gate')
    output=a.run/'reference_c_analysis';output.mkdir(exist_ok=False)
    config=yaml.safe_load(next(a.run.glob('sessions/*/resolved_config.yaml')).read_text())
    kin=BumiKinematics(config['paths']['kinematics'])
    legs=[7+i for i,n in enumerate(kin.joint_order) if any(k in n.lower() for k in ('leg_','hip','knee','ankle'))]
    jobs=[(r['rank'],'C',r['baseline'],r['candidate'],output,legs,config['paths']['kinematics']) for r in source['results']]
    with ProcessPoolExecutor(max_workers=a.workers) as pool:results=list(pool.map(compare_one,jobs))
    rows=[]
    for original,result in zip(source['results'],results):
        legal=consumed_legality(result) and not original['audit'].get('invalid_plans',0)
        metrics=result['metrics'];tracking=result['common_prefix_tracking_rms']
        local_gain=(metrics['reward_per_second']['mean']>0 and metrics['failure']['mean']<=0
            and tracking['anchor_pos']['final']<tracking['anchor_pos']['initial']
            and tracking['anchor_ori']['final']<tracking['anchor_ori']['initial'])
        rows.append(dict(variant=original['variant'],rank=original['rank'],legal_consumed_reference=legal,
            paired_initial_and_pre_influence=result['strict_pre_influence_match'],
            local_mean_gain=local_gain,eligible_for_full=legal and result['strict_pre_influence_match'] and local_gain,
            audit=original['audit'],**{k:result[k] for k in ('metrics','baseline_metrics','candidate_metrics',
                'common_prefix_tracking_rms','tracking_confidence','executed_reference_changes','executed_reference_checks')}))
    summary=dict(schema='genmo.reference_c_after_p0.audit.v1',run=str(a.run),full_music=source['full_music'],
        all_zero_passed=True,all_legal=all(r['legal_consumed_reference'] for r in rows),candidates=rows,
        eligible_full_variants=[r['variant'] for r in rows if r['eligible_for_full']],
        scope='fixed_music_exploratory_no_multiplicity_correction; full_extension_not_fixed_counterfactual')
    atomic_json(output/'summary.json',summary)
    print(json.dumps(dict(full_music=summary['full_music'],all_legal=summary['all_legal'],
        candidates=[{k:r[k] for k in ('variant','legal_consumed_reference','local_mean_gain','eligible_for_full')} for r in rows]),ensure_ascii=False),flush=True)


if __name__=='__main__':main()
