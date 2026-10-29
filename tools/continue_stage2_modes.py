"""四模式200更新对照结束后的validation决策与独立复现实验调度。

本进程不加载CUDA，不改变运行中的训练或评价，不读取evaluation来选择参数。
等待八组自然完成200更新或明确保护退出；只根据最后两次完整validation的真实
配对结果选择持续有益的2至4组。每组seed42从自己的完整latest恢复到1000，
seed43从同一原始零Adapter/预训练WM重新开始，分别占一张空闲GPU；网络、
优化器和Replay均独立。少于两组符合预声明条件时只保存证据，不凑数扩大训练。
终端输出、配置、完整命令和决策JSON保留在新的目录，旧目录始终只读。
"""

import argparse
import copy
import json
from pathlib import Path
import subprocess
import sys
import time


def eligible_validation(zero, report):
    """两次独立更新时点均要求实际收益；这不是置信区间或硬件验证。"""
    if report.get('max_steps') is not None or report.get('constraint_failure_count', 0):
        return False
    limits = {'mean_joint_q_rmse_rad': .98, 'mean_body_orientation_rmse_rad': .99,
              'mean_joint_qd_rmse_rad_s': 1.02, 'mean_root_position_rmse_m': 1.02,
              'mean_jerk_rms': 1.05, 'mean_contact_foot_slip_mps': 1.05}
    if report['completion_rate'] < zero['completion_rate']:
        return False
    if report['mean_full_horizon_tracking'] <= zero['mean_full_horizon_tracking']:
        return False
    return all(report.get(k) is not None and zero.get(k) is not None
               and report[k] <= max(zero[k], 1e-12)*limit for k, limit in limits.items())


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--phase-one', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--session-prefix', required=True)
    args = p.parse_args()
    import yaml
    source = Path(__file__).resolve().parents[1]
    root, output = Path(args.phase_one), Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    finished = {'completed_configured_Adapter_updates', 'failed_runtime_protection',
                'stopped_sustained_validation_degradation'}
    while True:
        states = []
        for gpu in range(8):
            f = root/f'gpu{gpu}/status.json'
            states.append(json.loads(f.read_text()) if f.exists() else {'phase': 'not_started'})
        state = dict(phase='waiting_phase_one', groups=states, timestamp=time.time())
        (output/'status.json').write_text(json.dumps(state, indent=2))
        if all(s['phase'] in finished for s in states):
            break
        time.sleep(30)
    candidates, rejected = [], []
    for gpu, state in enumerate(states):
        path = root/f'gpu{gpu}'
        zero_path = path/'validation_full_song_zero_dedicated4.json'
        if state['phase'] != 'completed_configured_Adapter_updates' or not zero_path.exists():
            rejected.append(dict(gpu=gpu, reason=state['phase']))
            continue
        zero = json.loads(zero_path.read_text())
        reports = [json.loads((path/f'validation_{u:06d}.json').read_text()) for u in (100, 200)]
        if not all(eligible_validation(zero, r) for r in reports):
            rejected.append(dict(gpu=gpu, reason='100_and_200_validation_not_both_qualified'))
            continue
        last = reports[-1]
        candidates.append(dict(gpu=gpu, validation=last, zero=zero,
            ranking=[last['completion_rate'], -last['mean_root_position_rmse_m'],
                     last['mean_full_horizon_tracking'], -last['mean_joint_q_rmse_rad']]))
    candidates.sort(key=lambda x: x['ranking'], reverse=True)
    selected = candidates[:4] if len(candidates) >= 2 else []
    decision = dict(selected_source_gpus=[x['gpu'] for x in selected], candidates=candidates,
                    rejected=rejected, selection_split='validation_only', evaluation_used=False,
                    status='expansion_authorized_by_validation' if selected else 'insufficient_consistent_gain_no_expansion')
    (output/'decision.json').write_text(json.dumps(decision, ensure_ascii=False, indent=2))
    if not selected:
        (output/'status.json').write_text(json.dumps(dict(phase=decision['status'], timestamp=time.time())))
        return 0
    config_dir = output/'configs'; config_dir.mkdir()
    resume = {}
    for pair, selected_row in enumerate(selected):
        old_gpu = selected_row['gpu']
        original = yaml.safe_load((root/f'config_gpu{old_gpu}.yaml').read_text())
        for slot, seed in enumerate((42, 43)):
            gpu = 2*pair+slot; config = copy.deepcopy(original)
            config['vector']['target_accepted_updates'] = 1000
            config['training']['seed'] = seed; config['vector']['sampling_seed'] = seed
            config['run_stage'] = 'validation_selected_1000_update_extension'
            if seed == 43:
                config['alternating']['experiment_label'] = config['alternating']['experiment_label'].replace('seed42', 'seed43')
            # seed42保留全部算法/优化配置，使恢复检查验证真正连续训练。
            (config_dir/f'gpu{gpu}.yaml').write_text('# validation选定配置；seed42恢复本组，seed43独立从零复现。\n'+
                yaml.safe_dump(config, sort_keys=False, allow_unicode=True))
            if seed == 42:
                resume[str(gpu)] = json.loads((root/f'gpu{old_gpu}/latest_checkpoint.json').read_text())['path']
    resume_path = output/'resume_map.json'; resume_path.write_text(json.dumps(resume, indent=2))
    command = [sys.executable, '-B', str(source/'tools/launch_stage2_alternating.py'),
        '--root', str(output/'formal'), '--session-prefix', args.session_prefix,
        '--config-dir', str(config_dir), '--gpus', ','.join(str(i) for i in range(2*len(selected))),
        '--resume-map', str(resume_path), '--wait-for-free-gpus']
    (output/'command.json').write_text(json.dumps(command, indent=2))
    subprocess.run(command, check=True)
    (output/'status.json').write_text(json.dumps(dict(phase='expansion_submitted', timestamp=time.time(),
        actual_training_status='see formal/gpuN/status.json; queue is not an Adapter update')))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
