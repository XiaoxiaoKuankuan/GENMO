"""全量Stage1评价的独立CPU分析与自动收尾，不干预物理工作进程。

从逐曲持久结果读取真实跟踪分、21关节误差、上下肢内容与参考运动难度，补充可解释
的失败率、分数分布、最差案例、逐关节误差和时长吞吐估计。原SM物理终止包含根高度
或倾斜偏差，不能一律称为摔倒。只读取已完成结果；任务结束后发布最终报告，工程
失败和未测音乐显式保留。误差到零、得分到100只是指标缺口，不是可实现收益承诺。
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
import html
import json
import math
from pathlib import Path
import time

from .census_metrics import aggregate, distribution
from .stage1_census import connect, utc, write_json


def analyze(root):
    root=Path(root);manifest=json.loads((root/'manifest.json').read_text())
    with connect(root) as db: jobs=db.execute('SELECT status,task,result FROM jobs ORDER BY ordinal').fetchall()
    tasks=[json.loads(j['task']) for j in jobs]
    rows=[json.loads(j['result']) for j in jobs if j['status']=='done' and j['result']]
    physical=[r for r in rows if r.get('status')=='physical_result']
    complete=len(rows)==len(tasks)
    result=dict(updated_utc=utc(),all_tasks_finished=complete,expected=len(tasks),results=len(rows),
        raw_manifest_entries=manifest['original_dataset_entries'],physical=aggregate(rows),
        definition={'complete':'按SM原终止规则执行到整个原库音乐输入末端；不代表动作精度合格',
            'score':'原八项跟踪分乘100；高分更好',
            'coverage_score':'跟踪分乘实际执行比例；提前结束的未执行尾部明确记为零分',
            'rmse':'只对已经真实执行的控制步统计，不能填补未执行尾部',
            'music_unit':'原库不同EDGE35音乐输入；原库本身的音乐切片不拼接成未经提供的完整原曲',
            'headroom':'测到的缺口，不等于第二阶段必然可恢复的收益',
            'complexity':'实际被消费参考的运动量，早停歌曲存在右删失，非人工复杂舞蹈标签'},
        score_bins={},upper={},lower={},per_joint=[],timing={},worst_tracking=[],failures=[])
    result['stopped_by_user']=(root/'user_requested_stop.json').exists()
    result['execution_protocol']={k:manifest.get(k) for k in ('timing_mode','num_envs_per_gpu','fixed_latency_seconds','wm_forward_calls_required')}
    result['execution_protocol']['forecast_enabled']=manifest['model'].get('forecast_enabled')
    for lo,hi in [(0,50),(50,70),(70,80),(80,90),(90,95),(95,100.000001)]:
        selected=[r for r in physical if lo<=r['statistics']['tracking_score_100']<hi]
        result['score_bins'][f'{lo}-{min(hi,100):g}']=dict(count=len(selected),completed=sum(r['complete'] for r in selected))
    for group in ('upper','lower'):
        for key in ('joint_rmse_rad','velocity_rmse_rad_s','range_ratio','speed_ratio','motion_content_cosine','velocity_phase_cosine'):
            result[group][key]=distribution([r['statistics'][group].get(key) for r in physical])
    if physical:
        names=physical[0]['statistics']['joint_position_per_joint_rmse_rad']
        steps=sum(r['statistics']['steps'] for r in physical)
        for name in names:
            position=math.sqrt(sum(r['statistics']['joint_position_per_joint_rmse_rad'][name]**2*r['statistics']['steps'] for r in physical)/steps)
            velocity=math.sqrt(sum(r['statistics']['joint_velocity_per_joint_rmse_rad_s'][name]**2*r['statistics']['steps'] for r in physical)/steps)
            hard=sum(r['physical_metrics_mean'].get('joint_hard_frequency/'+name,0)*r['statistics']['steps'] for r in physical)/steps
            result['per_joint'].append(dict(name=name,position_rmse_rad=position,velocity_rmse_rad_s=velocity,hard_limit_frequency=hard))
        result['per_joint'].sort(key=lambda x:x['position_rmse_rad'],reverse=True)
        phys=sum(r['physical_seconds'] for r in physical);wall=sum(r['wall_seconds'] for r in physical)
        remaining=sum(json.loads(j['task']).get('music_seconds',0) for j in jobs if j['status']!='done')
        result['timing']=dict(completed_episode_physical_seconds=phys,completed_episode_wall_seconds=wall,
            wall_to_physical_ratio=wall/phys if phys else None,remaining_planned_music_seconds=remaining,
            estimated_remaining_seconds_assuming_full_execution=remaining*wall/phys/len(manifest['gpus']) if phys else None,
            note='按已结束歌曲实际墙钟/物理时长估计；剩余歌曲假设完整执行，含在途歌曲全长，不是截止保证')
    def brief(r):
        return dict(task=r['task'],source=r.get('source'),split=r.get('split'),complete=r.get('complete'),reason=r.get('reason'),
            music_seconds=r.get('music_seconds'),physical_seconds=r.get('physical_seconds'),statistics=r.get('statistics'),trajectory_artifact=r.get('trajectory_artifact'))
    result['worst_tracking']=[brief(r) for r in sorted(physical,key=lambda x:x['statistics']['tracking_score_100'])[:30]]
    result['failures']=[brief(r) for r in rows if not r.get('complete')]
    result['by_source']={source:aggregate([r for r in rows if r.get('source')==source]) for source in manifest['unique_by_source']}
    result['by_split']={split:aggregate([r for r in rows if r.get('split')==split]) for split in manifest['unique_by_split']}
    result['reference_motion_quartiles']={}
    import numpy as np
    for key in ('reference_joint_speed_rms_rad_s','reference_range_mean_rad','reference_yaw_speed_rms_rad_s'):
        values=[r['statistics'][key] for r in physical]
        if values:
            bounds=np.quantile(values,[.25,.5,.75]).tolist();groups=[]
            for i in range(4):
                selected=[r for r in physical if (i==0 or r['statistics'][key]>=bounds[i-1]) and (i==3 or r['statistics'][key]<bounds[i])]
                groups.append(aggregate(selected))
            result['reference_motion_quartiles'][key]=dict(boundaries=bounds,groups=groups,early_end_censored=sum(not r['complete'] for r in physical))
    write_json(root/'optimization_space_analysis.json',result)
    def fmt(v,digits=3):return '待核验' if v is None else f'{v:.{digits}f}'
    a=result['physical'];s=a['song_equal_weight']
    page='<!doctype html><html lang="zh"><meta charset="utf-8"><title>Stage1全音乐评价与第二阶段优化空间</title><style>body{font:17px system-ui;max-width:1250px;margin:32px auto;padding:0 20px;color:#163047}table{border-collapse:collapse;width:100%}td,th{border-bottom:1px solid #ddd;padding:10px;text-align:left}.note{background:#fff1d6;padding:16px}small{color:#526579}</style>'
    page+='<h1>Stage1全音乐评价与第二阶段优化空间</h1><p>'+html.escape(result['updated_utc'])+'</p>'
    if manifest['timing_mode']=='sim_clock_fixed_latency':
        page+=f'<p>每卡 {manifest["num_envs_per_gpu"]} 个独立真实物理环境；WM前瞻关闭；沿仿真时钟推进，固定 {manifest["fixed_latency_seconds"]:.2f} 秒参考交付，每 {manifest["clock"]["request_interval_ticks"]/600:.2f} 秒生成一次。本报告评估动作执行和跟踪，不评价墙钟部署超时。</p>'
    state_text='本轮已由用户停止，结果作为旧链路证据保留。' if result['stopped_by_user'] else ('全量任务已结束。' if complete else '全量评价仍在执行，当前数字只代表已完成子集。')
    page+=f'<p class="note">'+state_text+f' 原库{manifest["original_dataset_entries"]}条舞蹈记录，去重后{len(tasks)}份音乐输入；已有{len(rows)}份结果，真实物理{len(physical)}份。</p>'
    page+='<table><tr><th>指标</th><th>实测</th><th>说明</th></tr>'
    cells=[('完成率',fmt(a['completion_rate']*100 if a['completion_rate'] is not None else None)+'%',f'{a["completed"]}/{len(physical)}，工程异常另列'),
        ('执行期间跟踪分',fmt(s['tracking_score_100']['mean'])+'/100','逐音乐等权，原八项定义'),
        ('全曲覆盖折算分',fmt(s['full_song_coverage_score_100']['mean'])+'/100','未完成尾部按零分计'),
        ('21关节角RMSE',fmt(s['joint_rmse_rad']['mean'])+' rad','逐音乐等权，只统计真实执行时段'),
        ('根XY位置RMS',fmt(s['root_xy_rmse_m']['mean'])+' m','实际机器人与当时原始参考'),
        ('根yaw RMS',fmt(s['root_yaw_rmse_rad']['mean'])+' rad','世界朝向差，不每步对齐'),
        ('工程异常',str(a['engineering_errors']),'不伪称倒地、不从覆盖数量删除')]
    for row in cells:page+='<tr>'+''.join('<td>'+html.escape(v)+'</td>' for v in row)+'</tr>'
    page+='</table><h2>如何解释优化空间</h2><p>未完成的歌曲说明有连续执行或调度问题；完整但跟踪误差大的歌曲说明仍有控制适应空间。上下肢、转身和速度分组用于定位主要缺口。它们都不直接证明冻结SM条件下第二阶段一定能消除这些误差。Stage1参考来自当前音乐生成，并非要求它逐帧复制原数据集舞者。</p>'
    for field,label in [('by_source','按音乐来源'),('by_split','按原数据划分')]:
        page+='<h2>'+label+'</h2><table><tr><th>组</th><th>完成/已测</th><th>跟踪分</th><th>全曲分</th><th>关节误差(rad)</th><th>根XY(m)</th></tr>'
        for key,v in result[field].items():
            z=v['song_equal_weight'];vals=[key,f'{v["completed"]}/{v["physical_episodes"]}',fmt(z['tracking_score_100']['mean']),fmt(z['full_song_coverage_score_100']['mean']),fmt(z['joint_rmse_rad']['mean']),fmt(z['root_xy_rmse_m']['mean'])]
            page+='<tr>'+''.join('<td>'+html.escape(x)+'</td>' for x in vals)+'</tr>'
        page+='</table>'
    page+='<h2>上下肢动作保持</h2><table><tr><th>部分</th><th>关节误差(rad)</th><th>速度误差(rad/s)</th><th>幅度比</th><th>速度比</th><th>速度相位余弦</th></tr>'
    for group,label in [('upper','上肢'),('lower','下肢')]:
        vals=[label]+[fmt(result[group][k]['mean']) for k in ['joint_rmse_rad','velocity_rmse_rad_s','range_ratio','speed_ratio','velocity_phase_cosine']]
        page+='<tr>'+''.join('<td>'+x+'</td>' for x in vals)+'</tr>'
    page+='</table><h2>最低跟踪分案例</h2><table><tr><th>音乐输入</th><th>结果</th><th>跟踪分</th><th>执行/目标秒数</th><th>关节误差(rad)</th></tr>'
    for r in result['worst_tracking'][:15]:
        vals=[r['task'],r['reason'],fmt(r['statistics']['tracking_score_100']),fmt(r['physical_seconds'])+'/'+fmt(r['music_seconds']),fmt(r['statistics']['joint_rmse_rad'])]
        page+='<tr>'+''.join('<td>'+html.escape(v)+'</td>' for v in vals)+'</tr>'
    mode=html.escape(manifest['timing_mode'])
    detail='Stage1全量音乐评价.html' if complete else 'per_music_results.json'
    page+=f'</table><p><a href="{detail}">全部逐曲明细</a> · <a href="optimization_space_analysis.json">分数分布、逐关节及难度统计JSON</a> · <a href="manifest.json">原始评价合同</a></p><small>原s595000、SM ckpt_024000、完整DDIM20、CFG2.5、seed42；真实Newton，时间模式：{mode}。单种子不代表所有生成噪声。复杂度按已观察参考统计，早停歌曲可能低估后续动作难度。</small></html>'
    tmp=root/'Stage1优化空间分析.html.tmp';tmp.write_text(page);tmp.replace(root/'Stage1优化空间分析.html')
    return result


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--root',required=True);parser.add_argument('--watch',action='store_true')
    args=parser.parse_args();root=Path(args.root)
    while True:
        result=analyze(root)
        if not args.watch or (root/'supervisor_complete.json').exists():
            if args.watch:
                from .stage1_census import report
                report(root)
            if args.watch:write_json(root/'analysis_complete.json',dict(utc=utc(),all_tasks_finished=result['all_tasks_finished'],results=result['results'],expected=result['expected']))
            break
        time.sleep(30)


if __name__=='__main__':main()
