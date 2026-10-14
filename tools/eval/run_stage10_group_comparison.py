"""安全停训后的Stage1/中期/当前模型完整音乐组对照入口。

prepare命令仅在CPU核验三个完整checkpoint、原训练源码/资产/随机核身份，冻结
独立val/test音乐组清单、种子及负载均衡分片。run命令由单机八卡torchrun启动，
每卡一个N=1 Isaac世界，复用已验收的真实GMT控制、四物理子步、参考提交和部署
时钟；三模型使用同一任务分片和初态逐首reset，不创建优化器、不恢复训练run。

Actor仍真实生成60帧、20步随机扩散，完整物理episode到音乐结束或物理失败。
每首保存逐控制奖励和紧凑实际轨迹，扩散链只在当前决策内存在，落盘proposal
明确不含完整链。退出核验Actor/Critic及GMT未变化，报告不冒充训练恢复或原有
32任务成绩。--smoke仅写入调用者指定的临时目录，行政截断绝不产生MC真值。
"""
from __future__ import annotations

import argparse
import copy
from datetime import timedelta
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))

import numpy as np
import torch
import yaml

from gem.closedloop.dppo.evaluation import _digest, _model_fingerprint, _publish_json
from gem.closedloop.dppo.training_checkpoint import SCHEMA, file_sha256
from tools.eval.stage10_group_comparison import music_group_plan, partition_by_music


def prepare(args):
    from gem.closedloop.dppo.full_dataset import FullMusicCatalog
    from gem.closedloop.dppo.training import _sources, _physical_configuration_identity
    run = args.training_run.resolve(strict=True)
    pointer = json.loads((run/'latest.json').read_text())
    if not (pointer['metadata']['reason']=='controlled_end' or pointer['metadata']['reason'].startswith('evaluated_final_')):
        raise ValueError('Current model requires completed controlled stop checkpoint')
    current = run/pointer['path']
    if file_sha256(current)!=pointer['sha256']:
        raise ValueError('Final checkpoint SHA differs')
    preheat = sorted((run/'checkpoints').glob('*r000000*complete_mc_preheat.pt'))
    if len(preheat)!=1:
        raise ValueError('Exactly one Stage1 Actor / post-MC-preheat Critic checkpoint required')
    middle = sorted((run/'checkpoints').glob('*r000147*evaluated_periodic*.pt'))
    if len(middle)!=1:
        raise ValueError('Exactly one evaluated iteration147 checkpoint required')
    paths = [('stage1',preheat[0]),('iteration147',middle[0]),('current',current)]
    records=[]; identity=None; config=None
    for label,path in paths:
        payload=torch.load(path,map_location='cpu',weights_only=False,mmap=True)
        if payload.get('schema')!=SCHEMA:
            raise ValueError('Unsupported checkpoint schema')
        if identity is None: identity=payload['identity'];config=payload['config']
        if payload['identity']!=identity:
            raise ValueError('Three checkpoints must have identical training/physics/kernel identity')
        records.append(dict(label=label,path=str(path),sha256=file_sha256(path),
            iteration=payload['state']['iteration'],actor_fingerprint=payload['actor_fingerprint'],
            critic_updates=payload['state']['critic_updates']))
        del payload
    if _sources(config)!=identity['source_files']:
        raise ValueError('Bound training or physics implementation differs; do not bypass the source identity')
    if _physical_configuration_identity(config)!=identity['physical_configuration']:
        raise ValueError('Bound physical configuration differs')
    for name,expected in identity['assets'].items():
        if file_sha256(config['paths'][name])!=expected:
            raise ValueError('Asset SHA differs: '+name)
    catalog=FullMusicCatalog(config['paths']['data_root'])
    audit=catalog.audit_files(require_audio=True)
    catalog.apply_audit(audit)
    plans={split:music_group_plan(catalog,split,args.seeds,maximum=1 if args.smoke else None,
        seconds=1. if args.smoke else None) for split in (['val'] if args.smoke else ['val','test'])}
    out=args.output.resolve();out.mkdir(parents=True,exist_ok=False)
    _publish_json(out/'catalog_audit.json',audit)
    for split,plan in plans.items():_publish_json(out/(split+'_plan.json'),plan)
    extension={str(p.relative_to(ROOT)):file_sha256(p) for p in [Path(__file__),Path(__file__).with_name('stage10_group_comparison.py')]}
    spec=dict(schema='genmo.stage10.three_model_group_comparison.v1',training_run=str(run),
        current_stop_iteration=pointer['iteration'],models=records,config=config,identity=identity,
        plans={split:dict(file=split+'_plan.json',sha256=file_sha256(out/(split+'_plan.json')),
            groups=plan['selected_unique_group_count'],tasks=plan['task_count']) for split,plan in plans.items()},
        evaluation_extension=extension,world_size=8,seeds=args.seeds,smoke=bool(args.smoke),
        transport='inprocess_N1_same_real_backend',updates_performed=False,
        critic_initial_scope='Stage1 Actor with its independent complete-MC-preheated Critic',
        storage='actual_50Hz_trajectory_rewards_and_proposals_without_archived_diffusion_chain')
    spec['comparison_id']=_digest(spec)
    _publish_json(out/'comparison_spec.json',spec)
    print(json.dumps(dict(output=str(out),models=records,plans=spec['plans']),ensure_ascii=False),flush=True)


def run_worker(args):
    import torch.distributed as dist
    rank,world,local=(int(os.environ.get(k,'-1')) for k in ('RANK','WORLD_SIZE','LOCAL_RANK'))
    if world!=8 or rank!=local or os.environ.get('CUDA_VISIBLE_DEVICES')!='0,1,2,3,4,5,6,7':
        raise ValueError('Explicit single-node eight-GPU torchrun required')
    output=args.output.resolve(strict=True)
    spec=json.loads((output/'comparison_spec.json').read_text())
    config=copy.deepcopy(spec['config'])
    dist.init_process_group('gloo',timeout=timedelta(minutes=90))
    check=[None]
    if rank==0:
        try:
            busy=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
            if busy:raise RuntimeError('Existing GPU jobs: '+busy)
            for path,expected in spec['evaluation_extension'].items():
                if file_sha256(ROOT/path)!=expected:raise ValueError('Prepared evaluation implementation changed')
            for record in spec['models']:
                if file_sha256(record['path'])!=record['sha256']:raise ValueError('Checkpoint changed')
            check[0]=dict(ok=True)
        except BaseException as error:check[0]=dict(ok=False,error=repr(error))
    dist.broadcast_object_list(check,src=0)
    if not check[0]['ok']:raise RuntimeError(check[0]['error'])
    rank_dir=output/f'rank{rank:02d}';rank_dir.mkdir(exist_ok=False)
    config['runtime'].update(rank=rank,genmo_device=f'cuda:{local}',physics_device=f'cuda:{local}',
        num_envs=1,generation_batch=1,asset_conversion_dir=str(rank_dir/'usd_assets'))
    config['stage9']['run_id']=spec['comparison_id']
    config['stage10']['physical_episode']['safety_limit_seconds']=1. if spec['smoke'] else None
    gmt=Path(config['paths']['gmt_repo'])
    for path in [gmt/'rsl_rl',gmt/'source/NoetixRobot',gmt/'scripts',gmt/'scripts/motion',gmt/'scripts/rsl_rl']:
        sys.path.insert(0,str(path))
    os.environ.setdefault('ISAACLAB_PATH',config['runtime']['isaaclab_root'])
    from isaaclab.app import AppLauncher
    parser=argparse.ArgumentParser(add_help=False);AppLauncher.add_app_launcher_args(parser)
    app_args=parser.parse_args([]);app_args.headless=True;app_args.device=f'cuda:{local}'
    from vector_app_resources import configure_vector_app
    configure_vector_app(app_args)
    removed={k:os.environ.pop(k) for k in list(os.environ) if k in ['RANK','LOCAL_RANK','WORLD_SIZE',
        'LOCAL_WORLD_SIZE','GROUP_RANK','ROLE_RANK','ROLE_WORLD_SIZE','MASTER_ADDR','MASTER_PORT'] or k.startswith('TORCHELASTIC_')}
    try:app=AppLauncher(app_args).app
    finally:os.environ.update(removed);os.environ['CUDA_VISIBLE_DEVICES']='0,1,2,3,4,5,6,7'
    physics=None;world_client=None;code=1
    try:
        import NoetixRobot.tasks
        from NoetixRobot.tasks.mimic.mimic_noetix_bumi4340_mha_sonic.closedloop.vector_backend import VectorFrozenGmtBackend
        from NoetixRobot.tasks.mimic.mimic_noetix_bumi4340_mha_sonic.closedloop.inprocess_service import InProcessVectorService
        from gem.closedloop.dppo.trainer import load_actor
        from gem.closedloop.dppo.pretrained_critic import PretrainedConditionCritic
        from gem.closedloop.dppo.run_control import build_policy
        from gem.closedloop.dppo.training import _validate_calibrated_execution
        from gem.closedloop.dppo.full_dataset import FullMusicCatalog
        from gem.closedloop.dppo.runtime import _DirectTransport
        from gem.closedloop.dppo.vector_collector import VectorWorldClient,VectorLaneBackend,validate_vector_close
        from gem.closedloop.dppo.vector_evaluation import SingleVectorLaneTransport
        from gem.closedloop.dppo.vector_environment import VectorUpperEnvironment
        from gem.closedloop.dppo.memory_execution import MemoryReceiveJournal,PrepaidMemoryBudget
        from gem.closedloop.dppo.run_management import DiskGuard
        from gem.closedloop.dppo.evaluation import evaluate_policy
        from gem.closedloop.online_conditions import OnlineConditionBuilder
        from gem.robots.bumi.feature_codec import BumiMotionFeatureCodec
        from gem.robots.bumi.kinematics import BumiKinematics
        from tools.eval.stage10_group_comparison import EpisodeTrajectorySink
        torch.cuda.set_device(local);torch.set_num_threads(4)
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        random.seed(42);np.random.seed(42);torch.manual_seed(42)
        actor,_,loading=load_actor(config)
        source_fingerprint=_model_fingerprint(actor)
        cc=config['stage10']['critic']
        critic=PretrainedConditionCritic(actor,dimension=cc['dimension'],layers=cc['layers'],
            heads=cc['heads'],freeze_encoders=True).to(f'cuda:{local}').eval()
        policy=build_policy(actor,config['stage9'],config['stage10']['precision'])
        if policy.kernel_config!=spec['identity']['kernel']:raise ValueError('Actual behavior kernel differs')
        _validate_calibrated_execution(config['timing']['deployment_profile'],policy.numerical_execution,
            config['stage10']['assets'],policy.kernel_config)
        catalog=FullMusicCatalog(config['paths']['data_root'])
        catalog.apply_audit(json.loads((output/'catalog_audit.json').read_text()))
        physics=InProcessVectorService(VectorFrozenGmtBackend(config))
        world_client=VectorWorldClient(_DirectTransport(physics),MemoryReceiveJournal(rank_dir/'world_memory'))
        backend=VectorLaneBackend(SingleVectorLaneTransport(world_client),MemoryReceiveJournal(rank_dir/'lane_memory'))
        kinematics=BumiKinematics(config['paths']['kinematics'])
        builder=OnlineConditionBuilder(BumiMotionFeatureCodec(kinematics),motion_frames=60)
        guard=DiskGuard(rank_dir,min_free_bytes=100*1024**3,max_run_bytes=75*1024**3)

        class CompactEvaluationEnvironment(VectorUpperEnvironment):
            def _save_evidence(self,value,path):
                if 'trace' in value:
                    value=dict(schema='genmo.stage10.evaluation_proposal.v1',generated=value['generated'],
                        policy_version=value['policy_version'],kernel=value['trace']['kernel_config'],
                        diffusion_chain_storage='not_archived_evaluation_only_no_optimizer')
                return super()._save_evidence(value,path)

        model_reports=[]
        for model in spec['models']:
            started=time.perf_counter()
            payload=torch.load(model['path'],map_location='cpu',weights_only=False,mmap=True)
            actor.load_state_dict(payload['actor'],strict=True);critic.load_state_dict(payload['critic'],strict=True)
            del payload
            fingerprint=_model_fingerprint(actor)
            if fingerprint!=model['actor_fingerprint']:raise ValueError('Loaded Actor fingerprint differs')
            if model['label']=='stage1' and fingerprint!=source_fingerprint:
                raise ValueError('Stage1 baseline checkpoint Actor is not the original s595000')
            actor.eval();critic.eval()
            reports=[]
            for split,item in spec['plans'].items():
                plan_path=output/item['file']
                if file_sha256(plan_path)!=item['sha256']:raise ValueError('Prepared task plan changed')
                plan=json.loads(plan_path.read_text());child=partition_by_music(plan,world)[rank]
                directory=rank_dir/model['label']/split;directory.mkdir(parents=True,exist_ok=False)
                if not child['tasks']:
                    _publish_json(directory/'empty.json',dict(rank=rank,parent_plan=plan['plan_sha256'],tasks=0))
                    continue
                controls=sum(int(np.ceil(t['remaining_music_seconds']*50))+100 for t in child['tasks'])
                budget=PrepaidMemoryBudget(dict(control_steps=controls,physics_steps=4*controls,
                    generations=sum(int(np.ceil(t['remaining_music_seconds']/.5))+10 for t in child['tasks'])))
                env=CompactEvaluationEnvironment(config,backend,builder,policy,budget,directory)
                env.disk_guard=guard;env.policy_version=model['iteration'];env.iteration=model['iteration']
                sink=EpisodeTrajectorySink(directory/'trajectories',critic,joint_names=kinematics.joint_order,
                    gmt_joint_names=physics.world.policy.joint_names,gamma_upper=config['stage9']['gamma_upper'])
                def progress(event):
                    print('[MODEL_EVAL] '+json.dumps(dict(rank=rank,model=model['label'],split=split,**event),ensure_ascii=False),flush=True)
                result=evaluate_policy(env,policy,child,directory/'report',catalog=catalog,actor_identity=model,
                    frozen_modules={'critic':critic},local_cuda_only=True,transition_sink=sink,progress=progress)
                if sink.active is not None:raise RuntimeError('Last episode evidence unfinished')
                if not spec['smoke'] and any(not e['complete_episode'] for e in sink.records):
                    raise RuntimeError('Full-song comparison encountered administrative truncation')
                _publish_json(directory/'trajectory_manifest.json',dict(model=model,parent_plan_sha256=plan['plan_sha256'],
                    records=sink.records,budget=budget.state_dict(),status=result['status'],
                    report_sha256=file_sha256(directory/'report/report.json'),
                    archived_evidence='actual_physical_npz_control_rewards_and_generated_proposals_no_full_chain'))
                reports.append(str(directory/'report/report.json'));budget.close()
            entry=dict(model=model,reports=reports,wall_seconds=time.perf_counter()-started,
                actor_unchanged=_model_fingerprint(actor)==fingerprint,rank=rank)
            if not entry['actor_unchanged']:raise RuntimeError('Evaluation changed Actor')
            _publish_json(rank_dir/model['label']/'completion.json',entry)
            model_reports.append(entry)
            dist.barrier()
        closed=world_client.call('close');validate_vector_close(closed);world_client=None
        _publish_json(rank_dir/'completion.json',dict(status='passed',models=model_reports,gmt_close=closed))
        dist.barrier()
        if rank==0:_publish_json(output/'completion.json',dict(status='passed',comparison_id=spec['comparison_id'],
            ranks=[f'rank{i:02d}/completion.json' for i in range(world)],updates_performed=False))
        code=0
    finally:
        if world_client is not None:
            try:world_client.call('close')
            except Exception as error:print('close failed',repr(error),flush=True)
        dist.destroy_process_group()
        import omni.kit.app
        omni.kit.app.get_app().post_quit(code);app.close()
    return code


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode',choices=['prepare','run'])
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--training-run',type=Path)
    parser.add_argument('--seeds',type=int,nargs='+',default=[20261014,20261015])
    parser.add_argument('--smoke',action='store_true')
    args=parser.parse_args()
    if args.mode=='prepare':
        if args.training_run is None:parser.error('--training-run is required for prepare')
        prepare(args);return 0
    return run_worker(args)


if __name__=='__main__':raise SystemExit(main())
