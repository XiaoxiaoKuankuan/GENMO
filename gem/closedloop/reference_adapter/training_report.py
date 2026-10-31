"""八个独立训练任务的只读自动效果汇总，不参与任何模型更新或选择。

各进程只读取同一正式根目录下已经完成的状态、配置和物理评价JSON，分别保留
短片段与完整歌曲结果。只用各自专用零Adapter基线计算变化，附带评价点、真实
环境数和进程身份；未完成整歌时明确为空。共享的只是报告，不共享策略、WM、
Replay、优化器或随机流，也不以报告好坏阻止训练。并发写入用独立临时名原子替换。
"""

import json
import os
from pathlib import Path
import re
import time


def update_report(root):
    root = Path(root)
    result = dict(captured_unix=time.time(), source='completed_real_physics_reports_not_imagined_reward', runs=[])
    keys = ['completion_rate', 'mean_tracking', 'mean_joint_q_rmse_rad', 'mean_joint_qd_rmse_rad_s',
            'mean_body_orientation_rmse_rad', 'mean_root_position_rmse_m', 'mean_root_xy_rmse_m', 'mean_jerk_rms',
            'mean_contact_foot_slip_mps', 'mean_beat_f1']
    for gpu in range(8):
        directory = root/f'gpu{gpu}'
        try:
            status = json.loads((directory/'status.json').read_text())
        except (FileNotFoundError, json.JSONDecodeError):
            result['runs'].append(dict(gpu=gpu, status='not_initialized'))
            continue
        command_file = Path(f'/proc/{status["pid"]}/cmdline')
        alive = command_file.exists() and str(directory).encode() in command_file.read_bytes()
        row = dict(gpu=gpu, status=status, verified_pid_argument_alive=alive,
                   latest_full_song=None, latest_prefix=None, best=None)
        try:
            row['best'] = json.loads((directory/'best_checkpoint.json').read_text())
        except (FileNotFoundError, json.JSONDecodeError):
            pass
        reports = []
        for path in directory.glob('evaluation_*.json'):
            if re.fullmatch(r'evaluation_\d{6}\.json', path.name) is None:
                continue
            try:
                reports.append((path, json.loads(path.read_text())))
            except (FileNotFoundError, json.JSONDecodeError):
                continue
        for path, report in sorted(reports, key=lambda item: item[1]['adapter_updates']):
            scope = report['evaluation_scope']
            count = report.get('actual_physics_envs', report['num_envs'])
            baseline = directory/f'evaluation_{scope}_zero_dedicated{count}.json'
            if not baseline.exists():
                continue
            zero = json.loads(baseline.read_text())
            changes = {key: 100*(report[key]/zero[key]-1)
                       if zero.get(key) and report.get(key) is not None else None for key in keys}
            info = dict(path=str(path), adapter_updates=report['adapter_updates'], scope=scope,
                metrics={key: report.get(key) for key in keys}, change_percent_vs_own_zero=changes,
                actual_physics_envs=count, constraint_failures=report['constraint_failure_count'],
                requested_song_count=report.get('requested_song_count', report['measured_evaluation_envs']),
                paired_evaluable_songs=report.get('paired_evaluable_songs', report['measured_evaluation_envs']),
                reference_generation_failures=report.get('reference_generation_failures', []),
                report_population=report.get('report_population', 'dataset_evaluation'),
                selection_population=report.get('selection_population', 'validation_only'))
            row['latest_full_song' if report['full_song_evaluation'] else 'latest_prefix'] = info
        result['runs'].append(row)
    temp = root/f'comparison_latest.{os.getpid()}.tmp'
    temp.write_text(json.dumps(result, ensure_ascii=False, indent=2))
    os.replace(temp, root/'comparison_latest.json')
    return result


def safe_update_report(root):
    """报告是可重建产物；某组退出或临时IO竞态不能中断其它组的真实训练。"""
    try:
        return update_report(root)
    except Exception as error:
        print(json.dumps(dict(comparison_report_warning=repr(error), timestamp=time.time()), ensure_ascii=False), flush=True)
        return None
