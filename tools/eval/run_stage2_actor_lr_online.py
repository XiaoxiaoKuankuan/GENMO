"""从完整配对物理证据中选择少量代表配置，串行执行最多八轮在线学习。

本入口专用于服务器2统一Actor学习率实验，不是正式训练启动器。必须先完成全部
不同扫描权重的32任务物理评估及逐控制证据分析，再按本文件中预先写明的经验筛选
规则选择配置；规则只排除明显退化，不能解释成理论安全保证或统计显著提升。

优先保留5e-8坐标DPPO对照、两种目标各自的1e-6原KL候选；若后者物理退化，则
从同一目标的原门槛候选中选择完成率、总回报较好的替代。先选一个确有配对收益
的放宽KL代表；若此时缺少另一种目标，再保留该目标合格的探索代表，保证A/B的
有限在线可比较性。最多四组，所有放宽KL项仍是探索。筛选门槛不因覆盖要求降低，
没有合格项的目标必须缺席。所有候选评估结果及不选择理由全部写入计划。
每个在线实验均从同一个Stage1、已验收MC Critic及空Actor Adam独立开始，禁止把
扫描后的Actor或其他候选优化器当作续训。实际提交最多八步，每四轮评估并保存真实
权重及当前行为策略的完整MC价值诊断，失败不补跑凑数。

默认只生成计划；显式--execute才运行。--wait允许只读等待既有物理/分析队列完成。
执行前检查GPU空闲、磁盘、配置SHA和八轮上限；进程异常保留证据并停止。恢复队列
只跳过已完整核验的成功实验，失败另建attempt，不覆盖任何已运行目录或checkpoint。
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import threading
import time

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
import yaml
from gem.closedloop.dppo.budget import atomic_json
from tools.eval.run_stage2_actor_lr_physics import sha, gpu_monitor, verify_summary


SELECTION_RULES=dict(
    schema='genmo.stage2.uniform_lr_finite_selection.v2',
    scope='empirical_finite_simulation_screen_not_formal_safety_or_significance',
    maximum_experiments=4, accepted_iterations_each=8,
    minimum_completion_delta=0, minimum_total_reward_ratio=.98,
    minimum_execution_seconds_ratio=.98, minimum_activity_ratio=.9,
    maximum_hard_excess_increase_rad=dict(actual=.03,consumed_reference=.05,generated=.05),
    maximum_hard_fraction_increase=.002,
    exploratory_minimum_reward_ratio=1.01,
    objective_coverage='retain_each_objective_if_an_original_or_improving_exploratory_candidate_is_eligible',
    maximum_exploratory_experiments=2,
    exploration_requires_larger_accepted_lr_than_5e8_baseline=True,
    ranking=['complete_tasks','total_reward','fewer_backtracking_attempts'],
    preferred_primary=['coordinate_dppo_lr5e-08_kl1','coordinate_dppo_lr1e-06_kl1',
                       'joint_score_pg_lr1e-06_kl1'])


def assess(baseline,candidate):
    """同配对任务的总量门禁，同时防止早失败和静止动作刷每秒分数。"""
    b=baseline['aggregate'];a=candidate['aggregate'];fail=[]
    for record in (b,a):
        if (record['episode_count']!=32 or record['valid_episode_count']!=32 or
                record['infrastructure_failure_count'] or record['administrative_truncation_count']):
            raise ValueError('Selection requires 32 valid complete-or-physically-failed episodes')
    if baseline['plan_sha256']!=candidate['plan_sha256']:
        raise ValueError('Cannot select from unpaired physical tasks')
    reward_ratio=a['reward_sum']/b['reward_sum']
    seconds_ratio=a['executed_seconds']/b['executed_seconds']
    activity_ratio=a['activity']['actual_activity_rad_s']['mean']/b['activity']['actual_activity_rad_s']['mean']
    if a['music_completion_count']<b['music_completion_count']:fail.append('fewer_complete_tasks')
    for key,value,bound in (
            ('total_reward',reward_ratio,SELECTION_RULES['minimum_total_reward_ratio']),
            ('execution_seconds',seconds_ratio,SELECTION_RULES['minimum_execution_seconds_ratio']),
            ('actual_activity',activity_ratio,SELECTION_RULES['minimum_activity_ratio'])):
        if value<bound:fail.append(key+'_regression')
    for domain,tolerance in SELECTION_RULES['maximum_hard_excess_increase_rad'].items():
        current=a['joint_limits'][domain]['hard'];old=b['joint_limits'][domain]['hard']
        if current['max_excess_rad']-old['max_excess_rad']>tolerance:
            fail.append(domain+'_hard_max_increase')
        if current['violation_fraction']-old['violation_fraction']>SELECTION_RULES['maximum_hard_fraction_increase']:
            fail.append(domain+'_hard_fraction_increase')
    return dict(eligible=not fail,failures=fail,reward_ratio=reward_ratio,seconds_ratio=seconds_ratio,
                activity_ratio=activity_ratio,completion=a['music_completion_count'],
                paired=candidate.get('paired'))


def select_candidates(scan,analyses):
    baseline=analyses['stage1'];assessments=[];by_name={}
    for c in scan['candidates']:
        name=c['name']
        if not c['screening']['passed']:
            item=dict(name=name,eligible=False,failures=['fixed_data_prescreen'])
        else:
            if name not in analyses:raise ValueError('Missing distinct-weight physical analysis: '+name)
            item=dict(name=name,**assess(baseline,analyses[name]))
        item.update(objective=c['objective'],proposed_lr=c['actor_lr'],kl_multiplier=c['kl_multiplier'],
                    config=c['path'],config_sha256=c['sha256'],
                    accepted_lr=c['backtracking']['accepted_lr'],
                    attempts=c['backtracking']['attempt_count'])
        assessments.append(item);by_name[name]=item
    selected=[]
    def add(item,reason):
        if item['eligible'] and item['name'] not in [v['name'] for v in selected]:
            selected.append(dict(item,selection_reason=reason))
    control=by_name['coordinate_dppo_lr5e-08_kl1']
    add(control,'predeclared_low_LR_control_if_not_physically_degraded')
    def ranking(row):return (row['completion'],row['reward_ratio'],-row['attempts'])
    for objective in ('coordinate_dppo','joint_score_pg'):
        preferred=by_name[objective+'_lr1e-06_kl1']
        if preferred['eligible']:add(preferred,'predeclared_1e-6_original_KL_primary')
        else:
            eligible=[r for r in assessments if r['eligible'] and r['objective']==objective and
                      r['kl_multiplier']==1 and r['name']!=control['name']]
            if eligible:add(max(eligible,key=ranking),'original_KL_alternative_after_primary_regression')
    exploratory=[r for r in assessments if r['eligible'] and r['kl_multiplier']>1 and
        r['reward_ratio']>=SELECTION_RULES['exploratory_minimum_reward_ratio'] and
        r['accepted_lr']>by_name[r['objective']+'_lr5e-08_kl1']['accepted_lr']]
    if exploratory:add(max(exploratory,key=ranking),'best_finite_relaxed_KL_exploration_not_formal_promotion')
    for objective in ('coordinate_dppo','joint_score_pg'):
        if any(r['objective']==objective for r in selected):continue
        eligible=[r for r in exploratory if r['objective']==objective]
        if eligible:add(max(eligible,key=ranking),'eligible_exploration_to_preserve_A_B_online_comparison')
    assert len(selected)<=SELECTION_RULES['maximum_experiments']
    assert sum(r['kl_multiplier']>1 for r in selected)<=SELECTION_RULES['maximum_exploratory_experiments']
    return dict(rules=SELECTION_RULES,assessments=assessments,selected=selected,
        warning='Selection on these 16 music groups is exploratory; confidence is not independent confirmation')


def build_plan(scan_path,analysis_path):
    scan=json.loads(scan_path.read_text());index=json.loads(analysis_path.read_text())
    if scan.get('status')!='completed_fixed_data_scan_not_physical_acceptance' or index['status']!='completed':
        raise ValueError('Online selection waits for complete fixed-data and physical analyses')
    queue=json.loads(Path(index['queue']).read_text())
    if queue['status']!='completed_all_distinct_survivors' or queue['scan_sha256']!=sha(scan_path):
        raise ValueError('Analysis must identify the same completed physical scan queue')
    analyses={};sources=[]
    for item in index['attempts']:
        if item['status']!='passed':continue
        path=Path(item['analysis'])
        if sha(path)!=item['sha256']:raise ValueError('Analysis bytes changed')
        value=json.loads(path.read_text())
        for name in [item['name'],*item['aliases']]:analyses[name]=value
        sources.append(dict(path=str(path),sha256=item['sha256']))
    plan=select_candidates(scan,analyses)
    plan.update(schema='genmo.stage2.uniform_lr_finite_online_plan.v1',scan=str(scan_path),
        scan_sha256=sha(scan_path),analysis_index=str(analysis_path),analysis_sha256=sha(analysis_path),
        sources=sources,initialization='fresh_Stage1_same_strict_MC_Critic_empty_Actor_Adam')
    return plan


def verify_config(job):
    if sha(job['config'])!=job['config_sha256']:raise ValueError('Selected configuration changed')
    c=yaml.safe_load(Path(job['config']).read_text());s=c['stage10']
    if (s['run_control']['max_training_iterations']!=8 or s['limits']['accepted_iterations']!=8 or
            s['training']['ppo_epochs']!=1 or s['scale']['actor_minibatch_chains']!=16384 or
            s['evaluation']['every_actor_iterations']!=4 or s['critic']['complete_mc']['every_actor_iterations']!=4):
        raise ValueError('Only eight on-policy steps with every-four evaluation/MC are authorized')
    artifact=s['critic']['preheat_artifact']
    if sha(artifact['path'])!=artifact['sha256']:raise ValueError('Accepted Critic artifact changed')


def verify_online(directory):
    evidence=verify_summary(directory)
    summary=json.loads(Path(evidence['summary']).read_text());state=summary['final_state']
    if state['iteration']!=8 or state['actor_updates']!=8 or state['policy_version']!=8:
        raise ValueError('Finite experiment must complete exactly eight accepted Actor updates')
    if not summary['initial_state'].get('critic_initialization'):
        raise ValueError('Missing strict MC Critic initialization')
    session=Path(evidence['summary']).parent;calibrations=[]
    for version in (4,8):
        path=session/'mc_calibration'/f'policy_{version}.json';result=json.loads(path.read_text())
        if (result['behavior_version']!=version or result['fit'] or result['after']['episodes']!=32 or
                result['after']['scope']!='independent_complete_MC_current_fixed_behavior_no_bootstrap'):
            raise ValueError('Missing corresponding frozen-current-policy independent MC diagnosis')
        calibrations.append(dict(path=str(path),sha256=sha(path),metrics=result['after']))
    for artifact in summary['evaluation_artifacts']:
        path=Path(directory)/artifact['path']
        if sha(path)!=artifact['sha256']:raise ValueError('Saved evaluation report changed')
    evaluated={r['iteration'] for r in summary['evaluation_artifacts']}
    if not {0,4,8}<=evaluated:raise ValueError('Missing independently saved evaluation iteration')
    pointer=json.loads((Path(directory)/'latest.json').read_text())
    if sha(Path(directory)/pointer['path'])!=pointer['sha256']:
        raise ValueError('Complete eight-step checkpoint changed or missing')
    evidence.update(mc=calibrations,checkpoint=pointer,accepted_actor_steps=8,
                    optimizer_attempts=state['optimizer_attempts'])
    return evidence


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--scan',type=Path,required=True);p.add_argument('--analysis-index',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--execute',action='store_true')
    p.add_argument('--wait',action='store_true');p.add_argument('--resume',action='store_true')
    args=p.parse_args()
    while args.wait:
        index=json.loads(args.analysis_index.read_text())
        if index['status']=='completed':break
        if index['status']!='running':raise RuntimeError('Evidence queue stopped: '+index['status'])
        time.sleep(30)
    plan=build_plan(args.scan,args.analysis_index)
    if not args.execute:
        print(json.dumps(plan,ensure_ascii=False,indent=2));return
    if args.resume:
        record=json.loads((args.output/'queue.json').read_text())
        if record['plan']!=plan:raise ValueError('Cannot resume against changed selection evidence')
    else:
        args.output.mkdir(parents=True,exist_ok=False)
        record=dict(schema='genmo.stage2.uniform_lr_finite_online_queue.v1',plan=plan,attempts=[],status='running')
        atomic_json(args.output/'queue.json',record)
    active=None;interrupted=False
    def stop(signum,frame):
        nonlocal interrupted
        interrupted=True
        if active is not None and active.poll() is None:os.killpg(active.pid,signal.SIGINT)
    signal.signal(signal.SIGINT,stop);signal.signal(signal.SIGTERM,stop)
    try:
        for job in plan['selected']:
            verify_config(job)
            previous=[v for v in record['attempts'] if v['name']==job['name']]
            if previous and previous[-1]['status']=='passed':
                if verify_online(previous[-1]['run'])!=previous[-1]['evidence']:
                    raise ValueError('Completed online evidence changed')
                continue
            if interrupted:break
            busy=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,process_name','--format=csv,noheader'],text=True).strip()
            if busy:raise RuntimeError('GPU occupied; refusing online launch: '+busy)
            if shutil.disk_usage(args.output).free<200*1024**3:raise RuntimeError('Insufficient evidence disk reserve')
            path=args.output/job['name']/f'attempt{len(previous)+1:02d}';path.mkdir(parents=True,exist_ok=False)
            command=['bash',str(ROOT/'scripts/train_stage10_8gpu_server1.sh'),'--config',job['config'],
                     '--output-dir',str(path/'run'),'--stop-after-iteration','8']
            item=dict(name=job['name'],run=str(path/'run'),command=command,status='running',started=time.time())
            record['attempts'].append(item);atomic_json(args.output/'queue.json',record)
            stopped=threading.Event();monitor=threading.Thread(target=gpu_monitor,args=(path/'gpu.csv',stopped),daemon=True)
            monitor.start()
            try:
                with (path/'console.log').open('x') as stream:
                    active=subprocess.Popen(command,cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
                    item['pid']=active.pid;atomic_json(args.output/'queue.json',record);code=active.wait()
                item['exit_code']=code
                if code:raise RuntimeError(f'Finite online experiment exited {code}')
                item.update(status='passed',evidence=verify_online(path/'run'))
            except BaseException as exc:
                item.update(status='failed',error=repr(exc));raise
            finally:
                stopped.set();monitor.join(timeout=10);active=None
                item['elapsed_seconds']=time.time()-item['started'];atomic_json(args.output/'queue.json',record)
            print(json.dumps(dict(name=job['name'],status=item['status'],seconds=item['elapsed_seconds'])),flush=True)
        record['status']='interrupted' if interrupted else 'completed_selected_finite_experiments'
    except BaseException as exc:
        record.update(status='stopped_with_failure',error=repr(exc));raise
    finally:atomic_json(args.output/'queue.json',record)


if __name__=='__main__':main()
