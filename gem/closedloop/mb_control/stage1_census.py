"""冻结Stage1所有音乐的真实Newton评价、持久任务队列和逐曲证据汇总。

prepare只读取原Stage1四来源train/val/test清单及现有EDGE35，不重建配对库。同一
特征内容、同初态、同噪声只执行一次，所有舞者别名保留。supervise运行独立多GPU
评价工作进程，故障重试与已完成结果持久化；真实倒地、参考耗尽均计入物理失败，
软件异常另列，不能从分母静默删除。worker冻结s595000与SM，按真实墙钟异步连续
生成和执行整首歌曲，保存消费时刻参考、原始参考速度、机器人轨迹和接管事件。
report只统计已落盘真实结果，明确未完成数量、音乐分割和来源；完成率、物理得分、
跟踪误差、运动内容分别报告。原训练进程和checkpoint从不由本模块修改或停止。
"""

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import html
import json
import os
from pathlib import Path
import random
import sqlite3
import subprocess
import sys
import time
import traceback


def utc(): return datetime.now(timezone.utc).isoformat()


def write_json(path, value):
    path = Path(path)
    tmp = path.with_name(path.name + f'.tmp.{os.getpid()}')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False))
    tmp.replace(path)


def connect(root):
    db = sqlite3.connect(str(Path(root)/'queue.sqlite'), timeout=60)
    db.execute('PRAGMA busy_timeout=60000')
    db.row_factory = sqlite3.Row
    return db


def claim(root):
    with connect(root) as db:
        db.execute('BEGIN IMMEDIATE')
        row = db.execute("SELECT * FROM jobs WHERE status='pending' ORDER BY ordinal LIMIT 1").fetchone()
        if row is None: return None
        db.execute("UPDATE jobs SET status='running',owner=?,attempts=attempts+1,started=? WHERE key=?",(os.getpid(),utc(),row['key']))
        return json.loads(row['task']), row['attempts']+1


def finish(root, task, result, *, retry=False):
    with connect(root) as db:
        db.execute("UPDATE jobs SET status=?,result=?,owner=NULL WHERE key=?",
                   ('pending' if retry else 'done',json.dumps(result,ensure_ascii=False,allow_nan=False),task['census_key']))


def prepare(config, output, gpus):
    from omegaconf import OmegaConf
    import torch
    root = Path(output).resolve(); root.mkdir(parents=True, exist_ok=False)
    cfg = OmegaConf.to_container(OmegaConf.load(config), resolve=True)
    data = OmegaConf.to_container(OmegaConf.load(cfg['stage1']['data_config']), resolve=True)
    grouped = {}; manifest_counts = {}; entry_count = 0
    for split in ('train','val','test'):
        for dataset in data['datasets'][split].values():
            source_root = Path(dataset['root']); manifest = source_root/'manifests'/f'{split}.jsonl'
            rows = [json.loads(line) for line in manifest.read_text().splitlines() if line.strip()]
            manifest_counts[f'{source_root.name}/{split}'] = len(rows)
            for row in rows:
                entry_count += 1
                path = source_root/row['music_feature_path']
                digest = row['source_music_feature_sha256']
                item = grouped.setdefault(digest, dict(census_key=digest, id=f'{source_root.name}__{row["sample_id"]}',
                    music_features=str(path), source=source_root.name, split=split, seeds=[42], aliases=[],
                    generator_unseen_music=split!='train', controller_unseen_motion=None))
                if item['split'] != split:
                    raise ValueError(f'同音乐特征跨split，必须先报告泄漏: {path}')
                item['aliases'].append(dict(source=source_root.name,split=split,sample_id=row['sample_id'],
                    source_manifest=str(manifest),motion_path=str(source_root/row['motion_path'])))
    tasks = list(grouped.values()); asset_errors = []
    for task in tasks:
        try:
            path = Path(task['music_features'])
            actual_digest = hashlib.sha256(path.read_bytes()).hexdigest()
            if actual_digest != task['census_key']: raise ValueError('音乐内容与原清单不符')
            music = torch.load(path,map_location='cpu',weights_only=True)
            if not isinstance(music,torch.Tensor) or music.ndim!=2 or music.shape[1]!=35 or not torch.isfinite(music).all():
                raise ValueError('原音乐必须是有限[N,35]Tensor')
            task['music_frames'] = len(music); task['music_seconds'] = len(music)/30
        except Exception as error:
            task['asset_error'] = repr(error); asset_errors.append(task['census_key'])
    # 确定性打乱后分层交错，运行中的初步结果不被全是短曲或单一来源主导。
    rng = random.Random(20261027); buckets = defaultdict(list)
    for task in tasks: buckets[(task['source'],task['split'])].append(task)
    for values in buckets.values(): rng.shuffle(values)
    ordered=[]
    while any(buckets.values()):
        for key in sorted(buckets):
            if buckets[key]: ordered.append(buckets[key].pop())
    tasks = ordered
    write_json(root/'all_tasks.json',tasks)
    with connect(root) as db:
        db.execute('PRAGMA journal_mode=WAL')
        db.execute('CREATE TABLE jobs (key TEXT PRIMARY KEY,ordinal INTEGER,task TEXT,status TEXT,attempts INTEGER DEFAULT 0,owner INTEGER,started TEXT,result TEXT)')
        for i,task in enumerate(tasks):
            result = dict(status='engineering_error',reason='invalid_source_asset',error=task['asset_error'],task=task['id'],source=task['source'],split=task['split'],census_key=task['census_key']) if 'asset_error' in task else None
            db.execute('INSERT INTO jobs(key,ordinal,task,status,result) VALUES(?,?,?,?,?)',
                (task['census_key'],i,json.dumps(task,ensure_ascii=False),'done' if result else 'pending',json.dumps(result,ensure_ascii=False) if result else None))
    cfg['tasks'] = str(root/'all_tasks.json')
    cfg['teacher'] = dict(enabled=False)
    cfg['rewards']['mode']='upper_eight'; cfg['rewards']['episode_dance_weight']=0.
    cfg['model']['forecast_enabled']=True
    cfg['execution']['physical_worker_gpus']=[]; cfg['execution']['start_large_scale_training']=False
    cfg['evaluation']=dict(timing_mode='wall_clock_async')
    cfg['census']=dict(root=str(root),seed=42)
    OmegaConf.save(OmegaConf.create(cfg),root/'config.yaml')
    manifest=dict(schema='genmo.stage1.full_music_census.v1',created_utc=utc(),source_config=str(config),
        original_dataset_entries=entry_count,unique_music_inputs=len(tasks),manifest_counts=manifest_counts,
        unique_by_source=dict(Counter(t['source'] for t in tasks)),unique_by_split=dict(Counter(t['split'] for t in tasks)),
        music_hours=sum(t.get('music_seconds',0) for t in tasks)/3600,asset_errors=asset_errors,gpus=gpus,
        stage1=cfg['stage1'],supermimic=cfg['supermimic'],model=cfg['model'],clock=cfg['clock'],
        seed=42,initial_state='nominal_robot_spec_consistent_with_prior_paired_baseline',
        timing_mode='wall_clock_async',future_encoder='fresh_zero_output_original_stage1_function',
        duplicate_policy='identical_feature_bytes_same_nominal_initial_state_same_seed_one_physical_trial',
        score_definition='100*upper_eight_tracking; coverage_score additionally multiplies executed/music duration',
        no_stage2_optimizer=True,all_physical_failures_kept=True,
        source_revision=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip())
    write_json(root/'manifest.json',manifest)
    print(json.dumps(manifest,ensure_ascii=False),flush=True)


def run_worker(collector,cfg,output,device):
    import numpy as np
    import torch
    from .application import seed_all
    from .evaluation import evaluate_tasks
    from .census_metrics import episode_statistics
    root=Path(cfg['census']['root']); collector.noise_seed=cfg['census']['seed']
    collector.timing_mode='wall_clock_async'; collector.record_reference_derivatives=True
    collector.actor.eval(); collector.actor.requires_grad_(False)
    if any(bool(v.detach().count_nonzero()) for v in collector.actor.future_encoder.output.parameters()):
        raise ValueError('全量Stage1评价禁止混入非零未来编码输出或Stage2参数')
    collector.adapter.bundle.assert_frozen() if hasattr(collector.adapter.bundle,'assert_frozen') else None
    write_json(output/'worker_ready.json',dict(pid=os.getpid(),utc=utc(),stage1_checkpoint=cfg['stage1']['checkpoint'],
        sm=collector.adapter.bundle.metadata['sha256'],all_actor_parameters_frozen=True,optimizer_created=False))
    while True:
        item=claim(root)
        if item is None: break
        task,attempt=item; episode=root/'episodes'/task['census_key']/f'attempt_{attempt}'
        episode.mkdir(parents=True,exist_ok=False)
        write_json(output/'current_task.json',dict(task=task['id'],key=task['census_key'],attempt=attempt,started_utc=utc()))
        try:
            seed_all(cfg['census']['seed'])
            summary=evaluate_tasks(collector,[task],episode,mode='frozen_stage1_full_path',device=device)
            row=summary['physical_results'][0]
            path=Path(row['trajectory_artifact'])
            with np.load(path,allow_pickle=False) as f: arrays={k:f[k] for k in f.files}
            arrays.update(ref_root_v=torch.cat([v[0] for v in collector.reference_derivatives]).numpy(),
                ref_root_w=torch.cat([v[1] for v in collector.reference_derivatives]).numpy(),
                ref_qd=torch.cat([v[2] for v in collector.reference_derivatives]).numpy())
            np.savez_compressed(episode/'physical_with_reference_velocity.npz',**arrays)
            row['statistics']=episode_statistics(row,arrays,list(collector.adapter.bundle.spec.joint_names))
            row.update(status='physical_result',census_key=task['census_key'],alias_count=len(task['aliases']),
                result_utc=utc(),attempt=attempt,trajectory_artifact=str(episode/'physical_with_reference_velocity.npz'))
            write_json(episode/'result.json',row)
            finish(root,task,row)
            print(json.dumps(dict(event='census_episode_finished',task=task['id'],complete=row['complete'],
                reason=row['reason'],score=row['statistics']['tracking_score_100'],joint_rmse=row['statistics']['joint_rmse_rad']),ensure_ascii=False),flush=True)
        except Exception as error:
            row=dict(status='engineering_error',reason='evaluation_exception',error=repr(error),traceback=traceback.format_exc(),
                task=task['id'],source=task['source'],split=task['split'],census_key=task['census_key'],attempt=attempt,result_utc=utc())
            write_json(episode/'error.json',row);finish(root,task,row,retry=attempt<2)
            raise  # 新进程重建真实物理，不能在潜在损坏的solver上继续下一首。
    write_json(output/'worker_complete.json',dict(pid=os.getpid(),utc=utc()))


def report(root):
    from .census_metrics import aggregate, distribution
    root=Path(root); manifest=json.loads((root/'manifest.json').read_text())
    with connect(root) as db:
        jobs=db.execute('SELECT key,status,task,result FROM jobs ORDER BY ordinal').fetchall()
    rows=[json.loads(j['result']) for j in jobs if j['status']=='done' and j['result']]
    states=dict(Counter(j['status'] for j in jobs)); physical=[r for r in rows if r.get('status')=='physical_result']
    totals=aggregate(rows)
    result=dict(schema='genmo.stage1.full_music_census.report.v1',updated_utc=utc(),
        complete=states.get('done',0)==len(jobs),evaluation_coverage=states.get('done',0)/len(jobs),
        expected_unique_music_inputs=len(jobs),original_dataset_entries=manifest['original_dataset_entries'],
        states=states,aggregate=totals,by_source={},by_split={},difficulty={},
        timing_mode=manifest['timing_mode'],seed=manifest['seed'],
        limitation='完成率不等于复杂舞蹈动作成功率；误差缺口不是第二阶段已证明可恢复的收益；单种子不代表所有噪声。')
    for field in ('source','split'):
        result['by_'+field]={key:aggregate([r for r in rows if r.get(field)==key]) for key in sorted({json.loads(j['task'])[field] for j in jobs})}
    for key in ('reference_joint_speed_rms_rad_s','reference_range_mean_rad','reference_yaw_speed_rms_rad_s'):
        vals=sorted(r['statistics'][key] for r in physical)
        if vals:
            import numpy as np
            q1,q3=np.quantile(vals,[.25,.75]).tolist()
            result['difficulty'][key]=dict(threshold_p25=q1,threshold_p75=q3,
                observed_reference_only=True,thresholds_final_only_when_all_done=result['complete'],
                lower_quartile=aggregate([r for r in physical if r['statistics'][key]<=q1]),
                upper_quartile=aggregate([r for r in physical if r['statistics'][key]>=q3]),
                censored_early_episodes=sum(not r['complete'] for r in physical))
    if physical:
        elapsed=(datetime.now(timezone.utc)-datetime.fromisoformat(manifest['created_utc'])).total_seconds()
        result['elapsed_seconds']=elapsed
        result['estimated_remaining_seconds_by_observed_throughput']=elapsed*(len(jobs)-states.get('done',0))/len(rows)
    write_json(root/'summary.json',result)
    write_json(root/'per_music_results.json',rows)
    def fmt(value): return '待核验' if value is None else f'{value:.4f}'
    def tab_row(r):
        s=r.get('statistics',{})
        fields=[r['task'],r.get('source'),r.get('split'),'完成' if r.get('complete') else r.get('reason'),
            fmt(s.get('tracking_score_100')),fmt(s.get('full_song_coverage_score_100')),
            fmt(s.get('joint_rmse_rad')),fmt(s.get('root_xy_rmse_m')),fmt(s.get('root_yaw_rmse_rad'))]
        return '<tr>'+''.join('<td>'+html.escape(str(x))+'</td>' for x in fields)+'</tr>'
    page='<!doctype html><html lang="zh"><meta charset="utf-8"><title>Stage1全量音乐真实物理评价</title><style>body{font:16px system-ui;margin:32px;color:#152536}table{border-collapse:collapse;width:100%}td,th{padding:8px;border-bottom:1px solid #ddd;text-align:left}th{position:sticky;top:0;background:#eef3f8}code{white-space:pre-wrap}.note{background:#fff3da;padding:16px}</style>'
    page+='<h1>冻结 Stage1＋SuperMimic：全量音乐评价</h1>'
    page+=f'<p>更新时间 {html.escape(result["updated_utc"])}；已形成结果 {states.get("done",0)}/{len(jobs)}；原舞蹈记录 {manifest["original_dataset_entries"]}。</p>'
    page+='<p class="note">'+('全量任务已结束。' if result['complete'] else '评价进行中，以下只代表已执行子集，不能外推全库。')+' 完成表示执行到音乐结束；跟踪误差只统计已执行时段，失败尾部不补造轨迹。全曲覆盖折算得分将缺失尾部按零分计入。原八项分数的100分是指标上限，不是已证明可实现的Stage2收益。</p>'
    page+=f'<p>真实物理 {totals["physical_episodes"]} 首，完成 {totals["completed"]} 首，工程错误 {totals["engineering_errors"]} 首。完成率 {fmt(totals["completion_rate"])}。</p>'
    for field,label in [('by_source','来源'),('by_split','数据划分')]:
        page+='<h2>'+label+'</h2><table><tr><th>组</th><th>已测</th><th>完成</th><th>完成率</th><th>跟踪得分</th><th>全曲折算得分</th><th>关节RMSE(rad)</th></tr>'
        for key,v in result[field].items():
            ss=v['song_equal_weight']; cells=[key,v['physical_episodes'],v['completed'],fmt(v['completion_rate']),fmt(ss['tracking_score_100']['mean']),fmt(ss['full_song_coverage_score_100']['mean']),fmt(ss['joint_rmse_rad']['mean'])]
            page+='<tr>'+''.join('<td>'+html.escape(str(x))+'</td>' for x in cells)+'</tr>'
        page+='</table>'
    page+='<h2>逐曲原始结果</h2><p>种子42；原s595000；DDIM20/eta0/CFG2.5；真实墙钟异步；SM ckpt_024000；统一原基线nominal初态。</p><table><tr>'+''.join('<th>'+x+'</th>' for x in ['音乐输入','来源','划分','结果','跟踪分','全曲折算分','关节RMSE(rad)','根XY RMS(m)','yaw RMS(rad)'])+'</tr>'
    page+=''.join(tab_row(r) for r in rows)+'</table><p><a href="summary.json">完整分组统计</a> · <a href="per_music_results.json">逐曲JSON</a> · <a href="manifest.json">固定评价合同</a></p></html>'
    tmp=root/'Stage1全量音乐评价.html.tmp';tmp.write_text(page);tmp.replace(root/'Stage1全量音乐评价.html')
    return result


def supervise(root):
    root=Path(root).resolve();manifest=json.loads((root/'manifest.json').read_text());children={};failures=Counter();generation=Counter()
    write_json(root/'supervisor.json',dict(pid=os.getpid(),utc=utc(),gpus=manifest['gpus']))
    def launch(gpu):
        generation[gpu]+=1;directory=root/'workers'/f'gpu{gpu}_attempt{generation[gpu]}';directory.parent.mkdir(exist_ok=True)
        env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',PYTHONDONTWRITEBYTECODE='1')
        log=(root/'workers'/f'gpu{gpu}_attempt{generation[gpu]}.log').open('w')
        cmd=[sys.executable,'-u','tools/evaluate_stage1_census.py','--config',str(root/'config.yaml'),'--output',str(directory),'--mode','full']
        process=subprocess.Popen(cmd,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        children[gpu]=(process,log,directory)
        write_json(root/'workers'/f'gpu{gpu}_launch.json',dict(pid=process.pid,command=cmd,utc=utc()))
    for gpu in manifest['gpus']: launch(gpu)
    last_report=0
    while children:
        for gpu,(process,log,directory) in list(children.items()):
            if process.poll() is None: continue
            log.close();del children[gpu]
            with connect(root) as db:
                db.execute("UPDATE jobs SET status='pending',owner=NULL WHERE status='running' AND owner=?",(process.pid,))
                pending=db.execute("SELECT count(*) FROM jobs WHERE status='pending'").fetchone()[0]
            if process.returncode:
                failures[gpu]+=1
                write_json(root/'workers'/f'gpu{gpu}_exit{generation[gpu]}.json',dict(pid=process.pid,returncode=process.returncode,utc=utc()))
            if pending and failures[gpu]<5: launch(gpu)
        if time.time()-last_report>30:
            status=report(root);last_report=time.time()
            write_json(root/'progress.json',dict(updated_utc=utc(),states=status['states'],active_workers={str(g):p.pid for g,(p,_,_) in children.items()}))
        time.sleep(2)
    status=report(root)
    write_json(root/'supervisor_complete.json',dict(utc=utc(),complete=status['complete'],states=status['states'],worker_failures=dict(failures)))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=['prepare','supervise','report'])
    parser.add_argument('--output',required=True);parser.add_argument('--config');parser.add_argument('--gpus',type=int,nargs='+',default=list(range(8)))
    args=parser.parse_args()
    if args.action=='prepare':prepare(args.config,args.output,args.gpus)
    elif args.action=='supervise':supervise(args.output)
    else:print(json.dumps(report(args.output),ensure_ascii=False))


if __name__=='__main__':main()
