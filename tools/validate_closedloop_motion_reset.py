"""在单GPU双环境中有限验证随机音乐位置对应的真实运动reset。

本工具只读正式训练清单与冻结GMT模型，不加载/更新GENMO，不启动训练循环。每个
指定的source/sample/秒数案例均从配对动作构造初态，独立回读PhysX中的根姿态、
根速度、关节角和关节速度，检查另一环境的物理状态及GMT/GENMO历史没有被重置。
随后实际执行50个GMT控制步，报告预热是否完成、终止原因和真实历史数量。

机械接口通过与动作可跟踪性分开报告：某段配对动作在一秒内摔倒会如实记录，不能
将它计作Actor覆盖，也不能用示范历史冒充物理历史。输出为指定的新JSON文件；USD
转换资产仅写入临时目录并自动清理。入口拒绝覆盖报告或抢占已有GPU计算任务。
默认案例特意包含FineDance长曲中后段；此验证不能证明更新后的Actor质量已提升。
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=ROOT / 'configs/closedloop/stage2_credit_server2.yaml')
    parser.add_argument('--gmt-repo', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--case', action='append', help='训练来源/样本名/音乐起点秒数，可重复指定')
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Output must be a new file')
    busy = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid,process_name',
                                    '--format=csv,noheader'], text=True).strip()
    if busy:
        raise RuntimeError('已有GPU计算任务，未启动验证: ' + busy)
    from gem.closedloop.dppo.training_config import configuration
    from gem.closedloop.dppo.full_dataset import FullMusicCatalog
    from gem.closedloop.dppo.target_activity import load_paired_activity
    from gem.closedloop.dppo.motion_initialization import make_motion_initialization
    config = configuration(args.config, genmo_repo=ROOT, gmt_repo=args.gmt_repo)
    catalog = FullMusicCatalog(config['paths']['data_root'])
    cases = args.case or ['FineDance/047/50', 'FineDance/047/120', 'FineDance/047/200',
                         'FineDance/047/240', 'FineDance/051/90', 'Mine/dance_2__badboy/40']
    packets = []
    for case in cases:
        source, sample_id, seconds = case.split('/')
        start = round(float(seconds) * 30)
        if not abs(start / 30 - float(seconds)) < 1e-9:
            raise ValueError('案例起点须落在30Hz源帧上')
        sample = next(s for s in catalog.samples['train'][source] if s['row']['sample_id'] == sample_id)
        target, qpos = load_paired_activity(catalog.data_root, sample, music_start_frame=start, return_motion=True)
        packets.append((case, make_motion_initialization(qpos, target.source, start_frame=start,
                        plan_frames=config['stage10']['generation_horizon']['frames'])))
    for directory in ['rsl_rl', 'source/NoetixRobot', 'scripts', 'scripts/motion', 'scripts/rsl_rl']:
        sys.path.insert(0, str(args.gmt_repo / directory))
    os.environ.setdefault('ISAACLAB_PATH', config['runtime']['isaaclab_root'])
    os.environ.setdefault('CUDA_VISIBLE_DEVICES', '0,1,2,3,4,5,6,7')
    visible_devices = os.environ['CUDA_VISIBLE_DEVICES']
    from isaaclab.app import AppLauncher
    app_parser = argparse.ArgumentParser(add_help=False)
    AppLauncher.add_app_launcher_args(app_parser)
    app_args = app_parser.parse_args([])
    app_args.headless, app_args.device = True, 'cuda:0'
    from vector_app_resources import configure_vector_app
    configure_vector_app(app_args)
    app = AppLauncher(app_args).app
    os.environ.update(LOCAL_RANK='0', CUDA_VISIBLE_DEVICES=visible_devices)
    report = dict(schema='paired_motion_reset.physical_validation.v1', cases=[],
                  scope='real_GMT_PhysX_reset_and_warmup_no_Actor_update', source_cases=cases)
    world, code = None, 1
    try:
        import numpy as np
        import torch
        import NoetixRobot.tasks  # 注册Isaac任务后才能导入环境。
        from NoetixRobot.tasks.mimic.mimic_noetix_bumi4340_mha_sonic.closedloop.vector_backend import VectorFrozenGmtBackend
        torch.cuda.set_device(0)
        torch.set_num_threads(4)
        with tempfile.TemporaryDirectory(prefix='paired-motion-usd-') as assets:
            config['runtime'].update(rank=0, num_envs=2, generation_batch=2,
                physics_device='cuda:0', genmo_device='cuda:0', asset_conversion_dir=assets)
            world = VectorFrozenGmtBackend(config)
            report['gmt_policy_sha256'] = world.policy.sha256
            report['runtime_fingerprint'] = world.fingerprint
            for case, packet in packets:
                world.reset([0, 1])
                world.advance_world(5)
                robot = world.env.robot
                neighbor = {key: getattr(robot.data, key)[1].clone() for key in
                            ('root_state_w', 'joint_pos', 'joint_vel')}
                history = world.snapshots([1])[0]
                manager = world.env.observation_manager
                buffers = [(buffer, buffer._buffer[:, 1].clone(), buffer._num_pushes[1].clone())
                    for group in manager._group_obs_term_history_buffer.values() for buffer in group.values()]
                actual = world.reset([0], initializations=[packet])[0]
                evidence = actual['reset_initialization']
                assert evidence['actual_state_verified'] and int(actual['history_append_count']) == 1
                assert int(np.asarray(actual['history_valid']).sum()) == 1
                assert actual['tracking_history_schema'] == 'gmt.causal_tracking_history90.v1'
                assert np.asarray(actual['tracking_history_values']).shape == (50, 90)
                assert int(np.asarray(actual['tracking_history_valid']).sum()) == 1
                np.testing.assert_array_equal(actual['tracking_history_ticks'], actual['tracking_reference_ticks'])
                assert evidence['music_start_frame'] == packet['music_start_frame']
                for key, before in neighbor.items():
                    torch.testing.assert_close(getattr(robot.data, key)[1], before, atol=1e-6, rtol=0)
                after = world.snapshots([1])[0]
                for key in ('tick', 'history_values', 'history_valid', 'history_ticks', 'history_append_count', 'tracking_history_values', 'tracking_history_valid', 'tracking_history_ticks'):
                    np.testing.assert_array_equal(after[key], history[key])
                for buffer, before, pushes in buffers:
                    torch.testing.assert_close(buffer._buffer[:, 1], before, atol=0, rtol=0)
                    torch.testing.assert_close(buffer._num_pushes[1], pushes, atol=0, rtol=0)
                for _ in range(2):
                    if not world.done[0]:
                        world.advance_world(25)
                final = world.snapshots([0])[0]
                assert int(final['history_append_count']) == final['tick'] // 12 + 1
                assert np.isfinite(final['history_values']).all()
                assert np.isfinite(final['tracking_history_values']).all()
                np.testing.assert_array_equal(final['tracking_history_ticks'], final['tracking_reference_ticks'])
                item = dict(tracking_history90_verified=True, case=case, reset_state_verified=True, neighbor_state_and_history_unchanged=True,
                    music_start_seconds=packet['music_start_frame'] / 30, reset_initialization=evidence,
                    warmup_completed=final['tick'] == 600 and not final['done'],
                    actual_warmup_control_steps=final['tick'] // 12, terminated=bool(final['done']),
                    reason=final['reason'], real_history_samples=int(final['history_append_count']),
                    valid_history_samples=int(np.asarray(final['history_valid']).sum()))
                report['cases'].append(item)
                print(json.dumps({k: v for k, v in item.items() if k != 'reset_initialization'}, ensure_ascii=False), flush=True)
            report.update(status='mechanical_checks_passed', warmup_completed_cases=sum(
                item['warmup_completed'] for item in report['cases']))
            world.close()
            world = None
        code = 0
    except BaseException as error:
        report.update(status='failed', error=repr(error))
        raise
    finally:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
        if world is not None:
            world.close()
        import omni.kit.app
        omni.kit.app.get_app().post_quit(code)
        app.close()
    return code


if __name__ == '__main__':
    raise SystemExit(main())
