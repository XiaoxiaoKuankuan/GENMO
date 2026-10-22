"""在原DPPO固定物理评价中保留真实GMT消费身份与中心参考的只读观察入口。

本入口仅允许evaluate-only和显式Actor，调用原训练脚本及原完整音乐任务。
通过现有transition_sink观察真实执行结果，按任务保存50Hz控制开始tick、GMT
实际21点窗口的参考tick与计划身份，以及控制结束时用于奖励的中心关节参考和
实际关节状态。窗口首次观察与奖励中心时刻分别记录，不将二者混称为消费时刻。

不改生成、前缀、计划提交、部署时钟、GMT、物理奖励或终止条件；不恢复或更新
正式训练。新增磁盘观察耗时不进入固定部署时钟。保存的中心轨迹可用于相同任务
物理执行前后对照；分叉后状态不同，因此轨迹差异不能冒充同状态因果动作比较。
"""
from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def validate_arguments(argv):
    parser=argparse.ArgumentParser(add_help=False)
    parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--evaluate-only',action='store_true')
    parser.add_argument('--evaluation-actor',type=Path,required=True)
    parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--resume')
    args,_=parser.parse_known_args(argv)
    if not args.evaluate_only or args.resume:
        raise ValueError('真实消费观察只允许显式Actor的独立evaluate-only运行')
    import yaml
    if yaml.safe_load(args.config.read_text())['stage10'].get('actor_initialization'):
        raise ValueError('独立评价必须关闭训练父模型初始化，防止覆盖显式评价Actor')
    root=args.output_dir.resolve()
    if root.name in ('train','capture') or root.exists():
        raise ValueError('评价必须使用尚不存在的独立目录')
    return root


def main():
    output=validate_arguments(sys.argv[1:])
    import numpy as np
    from gem.closedloop.dppo import evaluation
    from gem.closedloop.dppo.budget import atomic_json
    original=evaluation.evaluation_episode_flow
    rank=int(os.environ.get('RANK','0'))

    def observed(env,task,index,plan,*,catalog,limit,deterministic,transition_sink,progress):
        controls=[];decisions=[]

        def sink(current_task,result):
            if transition_sink is not None:transition_sink(current_task,result)
            metadata=result.get('metadata',{}) if isinstance(result,dict) else result.metadata
            generated=metadata.get('generated',{})
            decisions.append(dict(plan_id=generated.get('plan_id'),decision_tick=generated.get('decision_tick'),
                seed=generated.get('seed'),published=metadata.get('published'),rejection=metadata.get('rejection')))
            for row in metadata.get('execution_trace',[]):
                controls.append(dict(begin=int(row['control_tick_begin']),reward_tick=int(row['reference_tick']),
                    window_ticks=np.asarray(row['consumed_reference_ticks'],dtype=np.int64).copy(),
                    window_plans=np.asarray(row['consumed_plan_ids'],dtype=str).copy(),
                    center_plan=str(row['reference_plan_id']),
                    center_joint=np.asarray(row['reference']['joint_pos'],dtype=np.float64).copy(),
                    actual_joint=np.asarray(row['actual_joint_pos_gmt'],dtype=np.float64).copy()))

        try:
            return (yield from original(env,task,index,plan,catalog=catalog,limit=limit,
                deterministic=deterministic,transition_sink=sink,progress=progress))
        finally:
            folder=output/'consumed'/f'rank{rank:02d}'
            folder.mkdir(parents=True,exist_ok=True)
            path=folder/(task['task_id']+'.npz')
            if controls:
                begin=np.asarray([c['begin'] for c in controls],dtype=np.int64)
                if len(set(begin.tolist()))!=len(begin) or np.any(np.diff(begin)!=12):
                    raise ValueError('物理评价消费证据出现重复或缺失控制步')
                arrays=dict(control_begin_tick=begin,
                    reward_reference_tick=np.asarray([c['reward_tick'] for c in controls]),
                    gmt_window_reference_ticks=np.stack([c['window_ticks'] for c in controls]),
                    gmt_window_plan_ids=np.stack([c['window_plans'] for c in controls]),
                    reward_center_plan_id=np.asarray([c['center_plan'] for c in controls]),
                    reward_center_joint_pos=np.stack([c['center_joint'] for c in controls]),
                    actual_joint_pos=np.stack([c['actual_joint'] for c in controls]))
                with path.with_suffix('.tmp').open('wb') as handle:np.savez(handle,**arrays)
                path.with_suffix('.tmp').replace(path)
            atomic_json(folder/(task['task_id']+'.json'),dict(task_id=task['task_id'],dataset=task['dataset'],
                sample_id=task['sample_id'],seed=task['seed'],plan_sha256=plan['plan_sha256'],
                controls=len(controls),arrays=path.name if controls else None,
                arrays_sha256=hashlib.sha256(path.read_bytes()).hexdigest() if controls else None,
                decisions=evaluation._plain(decisions),scope='real_GMT_window_identity_at_control_begin_and_reward_center_at_control_end'))

    evaluation.evaluation_episode_flow=observed
    from tools.train_closedloop_stage10 import main as original_main
    original_main()


if __name__=='__main__':main()
