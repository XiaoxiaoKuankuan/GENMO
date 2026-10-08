"""对真实Stage1权重和已保存条件进行有界数值执行探针，不推进GMT或正式训练。

本工具只读取指定配置、模型和一份原始采样证据，使用相同条件/固定噪声比较原标量
执行与固定形状、固定去噪行位置执行。记录全部20步的自身概率、跨路径各层误差、
反向有限性及临时Adam更新前后行为；临时更新结束恢复权重，不发布checkpoint。
输出必须为新的JSON文件。工具不选择空闲GPU、不更改资源预算，调用者须先核验设备
无人占用，并通过timeout限定本次独立诊断。它不替代八卡PPO/GMT完整验收。
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import yaml

from gem.closedloop.dppo.execution_profile import probe_profiles, select_profile, compare_learning_execution
from gem.closedloop.dppo.policy import DPPODiffusionPolicy
from gem.closedloop.dppo.run_management import file_sha256
from gem.closedloop.dppo.trainer import load_actor


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--context-raw', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--shape', type=int, default=4)
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(output)
    torch.set_num_threads(4)
    torch.set_float32_matmul_precision('highest')
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    config = yaml.safe_load(Path(args.config).read_text())
    config['runtime']['genmo_device'] = args.device
    settings = config['stage10']['training']
    started = time.perf_counter()
    actor, _, source = load_actor(config)
    raw = torch.load(args.context_raw, map_location='cpu', weights_only=False)
    context = {key: value.to(args.device) for key, value in raw['trace']['conditions'].items()}
    policy = DPPODiffusionPolicy(actor, steps=settings['denoising_steps'], eta=settings['eta'],
        std_floor=settings['std_floor'], guidance_scale=settings['guidance_scale'], cfg_batch=False)
    baseline = probe_profiles(policy, context, maximum_microbatch=4, seed=12345)
    candidate = probe_profiles(policy, context, maximum_microbatch=4, seed=12345,
        fixed_execution_shape=args.shape, optimizer_probe=True, actor_lr=settings['actor_lr'])
    try:
        selected = select_profile([candidate])
    except RuntimeError as error:
        selected = dict(blocked=str(error))
    learning = None
    if 'blocked' not in selected:
        policy.execution_batch_size = selected.get('execution_batch_size')
        policy.cfg_batch = selected['cfg_batch']
        learning = compare_learning_execution(policy, context, microbatch=selected['microbatch'], actor_lr=settings['actor_lr'])
    report = dict(learning_comparison=learning, schema='genmo.stage10.real_actor_execution_probe.v1', device=args.device,
        source_checkpoint_sha256=file_sha256(config['paths']['checkpoint']),
        context_raw_sha256=file_sha256(args.context_raw), source_model=source,
        baseline=baseline, candidate=candidate, selected=selected,
        seconds=time.perf_counter()-started, scope='single_gpu_no_gmt_no_training_publication',
        torch_version=torch.__version__, cuda_version=torch.version.cuda,
        tf32_enabled=False, dtype='float32')
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x') as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps(dict(output=str(output), selected=selected, seconds=report['seconds']), ensure_ascii=False))


if __name__ == '__main__':
    main()
