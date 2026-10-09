"""固定真实条件和权重的单 GPU 批量数值复现工具。

读取已保存的 Stage2 初始权重、原 Stage1 架构与真实条件；不连接 GMT、不发布策略。
比较新数值契约在同一条实际新采样链上的 2/4/8/16/32 有效计算切片，保持旧概率
原样，分别报告条件、CFG、均值、方差、逐步联合概率、ratio、独立高斯公式和耗时。
微批可以混合不同链的同一去噪步，最后不足一批直接计算真实尾批，不填充。输入原
证据另作跨旧契约比较，不能作为新路径自身一致性的替代。可选 Adam 诊断会恢复权重。
本工具不冒充 160 条八卡四次更新的性能验收，输出明确记录工作量和设备型号。
"""
from pathlib import Path
import argparse
import json
import sys
import time
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from omegaconf import OmegaConf
from gem.closedloop.training import build_stage1_actor
from gem.closedloop.dppo.policy import DPPODiffusionPolicy, masked_joint_log_prob
from gem.closedloop.dppo.batch_execution import batch_accounting
from gem.closedloop.dppo.execution_profile import compare_learning_execution
from gem.closedloop.dppo.tensor_cache import RolloutTensorCache, ConditionGraphCache
from gem.closedloop.dppo.updater_v2 import _parameters
from gem.closedloop.dppo.run_management import file_sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('stage1-config', 'weights', 'stats', 'kinematics', 'output'):
        parser.add_argument('--'+name, required=True)
    parser.add_argument('--raw', nargs='+', required=True)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--adam', action='store_true')
    args = parser.parse_args()
    if Path(args.output).exists():
        raise FileExistsError(args.output)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    config = OmegaConf.create(json.loads(Path(args.stage1_config).read_text()))
    config.endecoder.stats_path = args.stats
    config.endecoder.kinematics_path = args.kinematics
    data = OmegaConf.create(dict(qpos30_stats=dict(path=args.stats),
        dataset_defaults=dict(kinematics_path=args.kinematics), sample_contract=dict(history_steps=50), datasets={}))
    actor = build_stage1_actor(config, data)
    payload = torch.load(args.weights, map_location='cpu', mmap=True, weights_only=False)
    actor.load_state_dict(payload['actor'], strict=True)
    actor = actor.float().eval().to(args.device)
    policy = DPPODiffusionPolicy(actor, cfg_batch=True, numerical_layout='sample_matrix_bmm_fp32.v1', defer_checks=True)
    rows = []
    with torch.no_grad():
        for i, path in enumerate(args.raw):
            old = torch.load(path, map_location='cpu', weights_only=False)['trace']
            context = {key: value.to(args.device) for key, value in old['conditions'].items()}
            trace = policy.sample_rollout(context, generator=torch.Generator(device=args.device).manual_seed(912+i))
            rows.append(SimpleNamespace(context=context, chain=trace['chain'][0], old_log_prob=trace['old_log_probs'][0],
                free_mask=trace['free_mask'][0], identity=dict(policy_version=0), transition_valid=True,
                metadata=dict(sampler_trace=trace, remaining_music_seconds=10.)))
    cache = RolloutTensorCache(rows, dict(advantages=torch.ones(len(rows))), args.device)
    pairs = [(row, step) for step in range(20) for row in rows]
    reports = []
    for micro in (2, 4, 8, 16, 32):
        torch.cuda.reset_peak_memory_stats()
        report = dict(microbatch=micro, effective_internal_transitions=len(pairs), padded_rows=0, passed=False)
        try:
            timings, differences = [], []
            for repeat in range(3):
                torch.cuda.synchronize(); start = time.perf_counter()
                with torch.no_grad():
                    conditions = ConditionGraphCache(policy, cache)
                    for offset in range(0, len(pairs), micro):
                        chunk = pairs[offset:offset+micro]
                        selected, steps = zip(*chunk)
                        result, mask = _parameters(policy, selected, steps, args.device, cache, conditions)
                        observed = cache.get('chain', selected, [s+1 for s in steps])
                        new = masked_joint_log_prob(observed, result['mean'], result['std'], mask)
                        old_prob = cache.get('old_log_prob', selected, steps)
                        oracle = torch.distributions.Normal(result['mean'].double(), result['std'].double()).log_prob(observed.double())
                        oracle = oracle.masked_fill(~mask, 0.).sum((-2, -1))
                        if repeat == 0:
                            differences.append(torch.stack(((new-old_prob).abs().max(), torch.expm1(new-old_prob).abs().max(),
                                (new-oracle).abs().max(), (result['mean']-cache.get('old_means', selected, steps)).abs().max())).cpu())
                torch.cuda.synchronize(); timings.append(time.perf_counter()-start)
            delta = torch.stack(differences).amax(0).tolist()
            report.update(max_logprob_error=delta[0], max_ratio_error=delta[1], independent_gaussian_error=delta[2],
                max_mean_error=delta[3], passed=delta[0]<=1e-4 and delta[1]<=1e-3 and delta[2]<=1e-8,
                repeat_seconds=timings, peak_bytes=torch.cuda.max_memory_allocated(),
                batches=[batch_accounting(min(micro,len(pairs)-i), cfg=True) for i in range(0,len(pairs),micro)])
        except torch.cuda.OutOfMemoryError as error:
            report.update(error=str(error), passed=False)
            torch.cuda.empty_cache()
        reports.append(report)
        print(json.dumps(report), flush=True)
    learning = None
    if args.adam and any(r['passed'] for r in reports):
        best = max(r['microbatch'] for r in reports if r['passed'])
        learning = compare_learning_execution(policy, rows[0].context, microbatch=best)
    result = dict(schema='stage10.batch_v4.probe.v1', scope='single_gpu_two_real_conditions_no_gmt_not_full_round',
        device=torch.cuda.get_device_name(), kernel=policy.kernel_config, reports=reports, learning=learning,
        source_sha256=file_sha256(args.weights), raw_sha256=[file_sha256(p) for p in args.raw])
    Path(args.output).write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n')
    cache.close()


if __name__ == '__main__':
    main()
