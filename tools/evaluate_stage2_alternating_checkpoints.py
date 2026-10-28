"""2026-10-28失败checkpoint的独立真实配对评价入口；不训练、不更新归一化。

读取指定完整checkpoint的Adapter和原奖励/投影配置，使用相同真实歌曲初态及
无探索噪声依次评价零Adapter、重复零Adapter、已有Adapter。采用原冻结Actor及
实际Newton物理，保存分歌曲误差、接触脚滑、完整率、初态身份和重复性噪声。
所有输出必须使用新目录，旧模型/Replay/日志只读；参考失败按episode单独记录。
"""

from pathlib import Path
import argparse
import json
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint',required=True)
    parser.add_argument('--config',required=True)
    parser.add_argument('--output',required=True)
    parser.add_argument('--splits',nargs='+',default=['validation','evaluation'])
    parser.add_argument('--max-steps',type=int)
    parser.add_argument('--no-repeat',action='store_true')
    pre,_=parser.parse_known_args()
    import yaml
    config=yaml.safe_load(Path(pre.config).read_text())
    sys.path.insert(0,config['supermimic']['repo'])
    from isaaclab.app import add_launcher_args,launch_simulation
    add_launcher_args(parser);args=parser.parse_args()
    import torch
    from supermimic.env import make_sim
    from supermimic.bridge.streaming_env import StreamingPhysicalEnv
    from gem.closedloop.reference_adapter.application import build_engine,read_tasks,seed_all
    from gem.closedloop.reference_adapter.vector_bank import VectorBank
    from gem.closedloop.reference_adapter.vector_control import VectorControl
    from gem.closedloop.reference_adapter.vector_collector import VectorCollector
    from gem.closedloop.reference_adapter.reference_constraints import ReferenceProjection
    from gem.closedloop.reference_adapter.precision import PrecisionObjective
    from gem.closedloop.reference_adapter.vector_entry import compile_kinematics
    from gem.closedloop.reference_adapter.paired_evaluation import evaluate_fixed,paired_comparison,tensor_digest
    output=Path(args.output);output.mkdir(parents=True,exist_ok=False)
    seed_all(42);torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False
    saved=torch.load(args.checkpoint,map_location='cpu',weights_only=False,mmap=True)
    cfg=saved['config'];cfg['supermimic']['repo']=config['supermimic']['repo'];cfg['observation_noise']=False
    engine=build_engine(cfg,'cuda:0');engine.step.adapter.load_state_dict(saved['adapter'])
    tasks=[t for t in read_tasks(cfg['tasks']) if t['split'] in args.splits]
    bank=VectorBank(tasks,engine,'cuda:0');control=VectorControl(engine,bank)
    settings=cfg['alternating'];control.adapter.disable_root_xy=True
    control.projection=ReferenceProjection(control.bundle.spec,control.adapter.bounds,**settings['projection']).cuda()
    calibration=json.loads(Path(settings['calibration']).read_text())
    control.precision=PrecisionObjective(calibration['scales'],**settings['precision'])
    control.soft_joint_limit_cost=False;control.sm_soft_termination_only=True
    actor_snapshot=control.bundle.frozen_snapshot()
    manifest=dict(checkpoint=args.checkpoint,checkpoint_updates=saved['state']['accepted_updates'],config=cfg,
        command=sys.argv,started_unix=time.time(),optimizer_steps=0,
        adapter_sha256=tensor_digest(torch.cat([x.flatten() for x in saved['adapter'].values()])))
    (output/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2))
    del saved
    (output/'compile.json').write_text(json.dumps(compile_kinematics(engine,bank),indent=2))
    sim_cfg=make_sim('cuda:0',args)
    with launch_simulation(sim_cfg,args):
        env=StreamingPhysicalEnv(sim_cfg,max(len(bank.by_split[s]) for s in args.splits),control.bundle)
        collector=VectorCollector(env,control,seed=42)
        for split in args.splits:
            reports={}
            modes=[('zero',True),('repeat',True),('adapter',False)] if not args.no_repeat else [('zero',True),('adapter',False)]
            for name,zero in modes:
                seed_all(42);collector.generator.manual_seed(42)
                report=evaluate_fixed(collector,zero=zero,split=split,max_steps=args.max_steps)
                reports[name]=report
                (output/f'{split}_{name}.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
                print(json.dumps(dict(split=split,mode=name,tracking=report['mean_full_horizon_tracking'],
                    completion=report['completion_rate'],constraint_failures=report['constraint_failure_count']),ensure_ascii=False),flush=True)
            comparison=paired_comparison(reports['zero'],reports['adapter'])
            if 'repeat' in reports:comparison['repeat_noise']=paired_comparison(reports['zero'],reports['repeat'])
            (output/f'{split}_paired.json').write_text(json.dumps(comparison,ensure_ascii=False,indent=2))
        control.bundle.assert_unchanged(actor_snapshot)
        (output/'completed.json').write_text(json.dumps(dict(ended_unix=time.time(),real_transitions=collector.real_transitions,
            optimizer_steps=0,actor_unchanged=True),indent=2))
    return 0


if __name__=='__main__':
    raise SystemExit(main())
