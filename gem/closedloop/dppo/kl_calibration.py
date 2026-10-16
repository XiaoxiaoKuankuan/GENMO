"""四步KL标定证据的运行门禁，不用旧全20步阈值冒充新算法验收。

有限固定rollout试验及同新奖励的真实物理对照必须先落盘。正式学习只接受绑定
当前行为核、奖励、选定soft/hard门槛的SHA证据，pending配置仅可用于采集、评估、
部署延迟标定或Critic预热。这里不猜测阈值、不修改采样，也不读取旧P1优化状态。
"""
from __future__ import annotations
import hashlib
import json
from pathlib import Path


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def _read(reference):
    if not isinstance(reference,dict) or set(reference)!={'path','sha256'}:
        raise ValueError('Calibration evidence requires an explicit file and SHA')
    content=Path(reference['path']).read_bytes()
    if hashlib.sha256(content).hexdigest()!=reference['sha256']:raise ValueError('Calibration source evidence SHA differs')
    return json.loads(content)


def validate_calibration(limits,kernel,reward,soft_limit,*,optimization=None):
    binding=limits['calibration']
    if binding.get('status')!='validated' or not binding.get('evidence'):
        raise ValueError('Four-step KL requires real fixed-rollout and physical calibration before Actor learning')
    evidence=binding['evidence'];path=Path(evidence['path'])
    content=path.read_bytes()
    if hashlib.sha256(content).hexdigest()!=evidence['sha256']:raise ValueError('KL calibration evidence SHA differs')
    report=json.loads(content)
    schema='genmo.stage2.four_step_kl_calibration.v2' if optimization is not None else 'genmo.stage2.four_step_kl_calibration.v1'
    if optimization is not None and report.get('optimization')!=optimization:
        raise ValueError('KL calibration does not bind actual Actor LR/epochs/minibatch/objective/backtracking/limits')
    if (report.get('schema')!=schema or report.get('status')!='passed' or
            report.get('kernel_sha256')!=digest(kernel) or report.get('reward_sha256')!=digest(reward)):
        raise ValueError('KL calibration belongs to another policy or reward')
    keys=('mean_coordinate','maximum_step_mean_coordinate','p95_coordinate')
    if report.get('hard_limits')!={key:limits[key] for key in keys} or report.get('soft_limit')!=soft_limit:
        raise ValueError('KL thresholds differ from measured calibration')
    if not isinstance(report.get('fixed_rollout_trials'),list) or not report['fixed_rollout_trials'] or not report.get('physical_comparison'):
        raise ValueError('Probability replay alone cannot calibrate a policy update')
    actors=[]
    for trial in report['fixed_rollout_trials']:
        measured=_read(trial['evidence'])
        if (measured.get('schema')!='genmo.stage2.real_data_four_step_actor_calibration.v1' or
                measured.get('kernel_sha256')!=digest(kernel) or measured.get('reward_sha256')!=digest(reward)):
            raise ValueError('Fixed rollout trial belongs to another four-step policy or reward')
        candidates=[c for c in measured['candidates'] if c['actor_lr']==trial['actor_lr']]
        if len(candidates)!=1:raise ValueError('Selected learning-rate trial is missing or ambiguous')
        candidate=candidates[0];kl=candidate['trainable4_kl'];update=candidate['update']
        if optimization is not None and candidate.get('optimization') != optimization:
            raise ValueError('Measured candidate optimization differs from published Actor settings')
        from .updater import check_kl
        if (kl['gaussian_kl_steps']!=[15,16,17,18] or kl['covered_internal_transitions']!=4*kl['covered_chains']
                or not kl['frozen_identity']['checked'] or not candidate['hard_kl']['passed'] or not check_kl(kl,limits)['passed']
                or update['optimizer_steps']!=update['planned_optimizer_steps'] or not update['optimizer_steps']
                or any(s['pre_step_mean_coordinate_kl']>=soft_limit for s in update['steps'])):
            raise ValueError('Fixed rollout does not prove a complete bounded four-step update')
        actors.append(candidate['evaluation_actor']['sha256'])
    comparison=_read(report['physical_comparison'])
    if (comparison.get('schema')!='genmo.closedloop.stage10.paired_comparison.v1' or comparison.get('status')!='passed'
            or not comparison.get('all_planned_tasks_included') or comparison.get('paired_task_count',0)<32
            or comparison.get('execution_kernel_sha256')!=digest(kernel)
            or comparison.get('execution_reward_sha256')!=digest(reward)):
        raise ValueError('Independent paired physical evidence is incomplete or has a different identity')
    if (comparison.get('evaluated_actor_initialization',{}).get('sha256') not in actors
            or comparison['overall']['failure_pair_counts']['new_failure']):
        raise ValueError('Physical comparison must evaluate the selected Actor without introducing new paired failures')
    if not isinstance(report.get('selection_reason'),str) or len(report['selection_reason'].strip())<20:
        raise ValueError('Calibration must explain thresholds using measured policy change and physical results')
    return report


def publish_calibration(config_path,fixed_report,physical_comparison,actor_rate,selection_reason,output):
    """从已完成的真实试验发布只读标定证据；不会生成虚构物理结果或改写候选配置。"""
    import copy
    import os
    import torch
    from .training_config import configuration
    config=configuration(config_path)
    def reference(path):
        path=Path(path).resolve(strict=True)
        return dict(path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    fixed=reference(fixed_report);measured=_read(fixed)
    case=Path(measured['source_case']);record=json.loads((case/'case.json').read_text())
    if hashlib.sha256((case/'case.json').read_bytes()).hexdigest()!=measured['source_case_sha256']:
        raise ValueError('Fixed rollout case identity differs')
    from .training_checkpoint import file_sha256
    if file_sha256(case/'learning_state.pt')!=record['learning_state_sha256']:
        raise ValueError('Fixed rollout learning state SHA differs')
    source=torch.load(case/'learning_state.pt',map_location='cpu',weights_only=False,mmap=True)
    limits=copy.deepcopy(config['stage10']['kl']);soft=config['stage9']['kl_soft_stop_coordinate']
    from .actor_experiment import optimization_binding
    optimization=optimization_binding(config['stage10'])
    if actor_rate != config['stage10']['training']['actor_lr']:
        raise ValueError('Published calibration must bind the configured Actor learning rate')
    report=dict(schema='genmo.stage2.four_step_kl_calibration.v2',status='passed',optimization=optimization,
        kernel_sha256=digest(source['kernel']),reward_sha256=digest(config['stage9']['reward']),
        hard_limits={key:limits[key] for key in ('mean_coordinate','maximum_step_mean_coordinate','p95_coordinate')},
        soft_limit=soft,fixed_rollout_trials=[dict(evidence=fixed,actor_lr=actor_rate)],
        physical_comparison=reference(physical_comparison),selection_reason=selection_reason,
        learning_benefit_not_implied_by_kl_pass=True)
    output=Path(output);output.parent.mkdir(parents=True,exist_ok=True)
    temporary=output.with_suffix(output.suffix+'.pending')
    if output.exists() or temporary.exists():raise FileExistsError(output)
    try:
        with temporary.open('x') as stream:
            json.dump(report,stream,ensure_ascii=False,indent=2,allow_nan=False);stream.flush();os.fsync(stream.fileno())
        limits['calibration']=dict(status='validated',scope='four_trainable_steps_only',evidence=reference(temporary))
        validate_calibration(limits,source['kernel'],config['stage9']['reward'],soft,optimization=optimization)
        os.link(temporary,output)
    finally:temporary.unlink(missing_ok=True)
    return reference(output)


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',required=True);parser.add_argument('--fixed-report',required=True)
    parser.add_argument('--physical-comparison',required=True);parser.add_argument('--actor-rate',type=float,required=True)
    parser.add_argument('--selection-reason',required=True);parser.add_argument('--output',required=True)
    args=parser.parse_args()
    print(json.dumps(publish_calibration(args.config,args.fixed_report,args.physical_comparison,
        args.actor_rate,args.selection_reason,args.output),ensure_ascii=False))
