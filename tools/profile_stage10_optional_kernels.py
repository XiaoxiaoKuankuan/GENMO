"""服务器1八卡的可选算子实验，不修改默认Actor、数值契约或正式训练配置。

逐次显式选择编译、CUDA Graph、FP32高效注意力或保持有效历史顺序的融合GRU。
每个rank使用封存真实rollout的不同条件，冻结原Actor，实际重新采样20步链并按
原logp/ratio/独立高斯门槛复算；另外检查标量/缓存反向、一次Adam及各模块梯度，
最后分别测三次采样和带梯度概率重算。对不支持、掩码错误、概率漂移、显存不足
如实记录失败，不启用混合精度、不降低概率门槛、不将纯核耗时写成闭环时间。

所有修改只存在本进程的临时forward绑定，退出即销毁，不发布策略或可恢复训练。
Graph缓存按实际形状及条件是否需要梯度分开建立，无复制填充；GRU共享原cell的
四个参数，压紧有效槽时保留顺序，全空历史返回严格零。注意力保留RoPE和掩码，
强制高效FP32后端，不把回退普通注意力当作融合成功。所有测试只能在八卡执行。
"""
import argparse
import json
import os
from datetime import timedelta
from pathlib import Path
import sys
import time
import types
import traceback

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
import torch.distributed as dist
from tools.train_closedloop_stage10 import configuration
from tools.train_closedloop_stage10_8gpu import _available_gpus
from gem.closedloop.dppo.trainer import load_actor
from gem.closedloop.dppo.policy import DPPODiffusionPolicy
from gem.closedloop.dppo.execution_profile import _log_probs, compare_learning_execution
from gem.closedloop.dppo.execution_checks import policy_phase
from gem.network.base_arch.transformer.encoder_rope import RoPEAttention


def attention_forward(self, x, context=None, attn_mask=None, key_padding_mask=None):
    from torch.nn.attention import sdpa_kernel, SDPBackend
    context = x if context is None else context
    batch, length, _ = x.shape
    keys = context.shape[1]
    q = self.query(x).reshape(batch, length, self.num_heads, -1).transpose(1, 2)
    k = self.key(context).reshape(batch, keys, self.num_heads, -1).transpose(1, 2)
    v = self.value(context).reshape(batch, keys, self.num_heads, -1).transpose(1, 2)
    q, k = self.rope.rotate_queries_or_keys(q), self.rope.rotate_queries_or_keys(k)
    allowed = torch.ones((batch, 1, length, keys), device=x.device, dtype=torch.bool)
    if attn_mask is not None:
        allowed = allowed & ~attn_mask.reshape(-1, 1, length, keys)
    if key_padding_mask is not None:
        allowed = allowed & ~key_padding_mask[:, None, None, :]
    # 高效后端使用浮点bias，禁止它因bool mask不支持而静默退回普通实现。
    bias = torch.zeros_like(allowed, dtype=x.dtype).masked_fill(~allowed, float('-inf'))
    bias = bias.expand(batch, self.num_heads, length, keys).contiguous()
    with sdpa_kernel(SDPBackend.EFFICIENT_ATTENTION):
        result = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=bias, dropout_p=0.)
    return self.proj(result.transpose(1, 2).reshape(batch, length, -1))


class _GraphDenoiser(torch.nn.Module):
    def __init__(self, denoiser, original):
        super().__init__()
        self.denoiser = denoiser
        self.original = original

    def forward(self, x, timestep, condition, length):
        value = self.original(x, timestep, y=dict(f_cond=condition, length=length), inputs={})
        return value['pred_x_start'], value['static_conf_logits']


def install(actor, variant):
    """不注册额外可训练参数，返回用于报告的真实执行信息。"""
    info = dict(variant=variant, default_training_changed=False)
    if variant == 'compile':
        actor.denoiser.forward = torch.compile(actor.denoiser.forward, fullgraph=True, dynamic=False)
    elif variant == 'graph':
        original, graphs = actor.denoiser.forward, {}

        def forward(x, timestep, y=None, inputs=None, **kwargs):
            if inputs or kwargs or set(y) != {'f_cond', 'length'}:
                raise ValueError('Graph experiment accepts only the actual DPPO tensor interface')
            args = (x, timestep, y['f_cond'], y['length'])
            key = tuple((tuple(a.shape), a.dtype, a.requires_grad) for a in args)
            if key not in graphs:
                module = _GraphDenoiser(actor.denoiser, original).eval()
                with torch.enable_grad(), torch.autocast('cuda', enabled=False, cache_enabled=False):
                    graphs[key] = torch.cuda.make_graphed_callables(module, args,
                        num_warmup_iters=3, allow_unused_input=True)
                info['captured_shapes'] = [str(k) for k in graphs]
            outputs = graphs[key](*args)
            return dict(pred_x_start=outputs[0], static_conf_logits=outputs[1])

        actor.denoiser.forward = forward
    elif variant == 'attention':
        count = 0
        for module in actor.denoiser.modules():
            if isinstance(module, RoPEAttention):
                module.forward = types.MethodType(attention_forward, module)
                count += 1
        info['replaced_attention_modules'] = count
    elif variant == 'gru':
        encoder = actor.history_encoder
        gru = torch.nn.GRU(49, encoder.cell.hidden_size, batch_first=True).to(next(encoder.parameters()).device)
        for target, source in (('weight_ih_l0', 'weight_ih'), ('weight_hh_l0', 'weight_hh'),
                               ('bias_ih_l0', 'bias_ih'), ('bias_hh_l0', 'bias_hh')):
            setattr(gru, target, getattr(encoder.cell, source))
        gru.train()  # 无dropout；cuDNN反向需要训练reserve，不改变外层Actor的eval状态。
        gru.flatten_parameters()

        def forward(history, valid, relative_times):
            features = torch.cat((history, relative_times[..., None]), -1)
            features = torch.where(valid[..., None], features, 0.)
            positions = (~valid).to(torch.int32).argsort(dim=1, stable=True)
            packed_values = features.gather(1, positions[..., None].expand_as(features))
            lengths = valid.sum(1).clamp_min(1).cpu()
            packed = torch.nn.utils.rnn.pack_padded_sequence(packed_values, lengths,
                batch_first=True, enforce_sorted=False)
            _, hidden = gru(packed)
            output = encoder.out_proj(hidden[0])
            return torch.where(valid.any(1, keepdim=True), output, 0.)

        encoder.forward = forward
        info['parameter_aliases_preserved'] = all(getattr(gru, a) is getattr(encoder.cell, b)
            for a, b in (('weight_ih_l0','weight_ih'),('weight_hh_l0','weight_hh'),
                         ('bias_ih_l0','bias_ih'),('bias_hh_l0','bias_hh')))
    else:
        raise ValueError(variant)
    return info


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True, type=Path)
    parser.add_argument('--rollout', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--variant', required=True, choices=('compile','graph','attention','gru'))
    args = parser.parse_args()
    rank, world = int(os.environ['RANK']), int(os.environ['WORLD_SIZE'])
    if world != 8 or rank != int(os.environ['LOCAL_RANK']):
        raise ValueError('Exactly eight local GPUs required')
    dist.init_process_group('gloo', timeout=timedelta(minutes=20))
    status = [None]
    if rank == 0:
        try:
            if args.output.exists():
                raise FileExistsError(args.output)
            status[0] = dict(devices=_available_gpus())
        except Exception as error:
            status[0] = dict(error=str(error))
    dist.broadcast_object_list(status, 0)
    if 'error' in status[0]:
        raise RuntimeError(status[0]['error'])
    if args.variant == 'compile':
        # 八个编译器不争用默认共享缓存，也不占用用户原有cache；任务结束按精确路径清理。
        cache = args.output.parent / (args.output.stem+f'.compile_cache_rank{rank:02d}')
        if cache.exists():raise FileExistsError(cache)
        os.environ['TORCHINDUCTOR_CACHE_DIR'] = str(cache/'inductor')
        os.environ['TRITON_CACHE_DIR'] = str(cache/'triton')
        os.environ['TORCHINDUCTOR_COMPILE_THREADS'] = '1'
    torch.cuda.set_device(rank); torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    config = configuration(args.config); config['runtime']['genmo_device'] = f'cuda:{rank}'
    report = dict(rank=rank, variant=args.variant, status='failed', scope='optional_kernel_experiment_no_policy_publication',
                  temporary_compiler_cache=str(cache) if args.variant=='compile' else None)
    try:
        actor, _, _ = load_actor(config)
        policy = DPPODiffusionPolicy(actor, cfg_batch=True, numerical_layout='sample_matrix_bmm_fp32.v1', defer_checks=True)
        fixture = torch.load(args.rollout, map_location='cpu', weights_only=False)
        context = {k:v.to(f'cuda:{rank}') for k,v in fixture['traces'][rank*20]['conditions'].items()}
        seed = 9871+rank
        sample = lambda: policy.sample_rollout(context, generator=torch.Generator(device=f'cuda:{rank}').manual_seed(seed))
        reference = sample()
        with torch.no_grad(), policy_phase(policy):
            baseline = _log_probs(policy, context, reference, 16)
        before_names = [(name, id(p), p.requires_grad) for name,p in actor.named_parameters()]
        # 同样20步、同样一条真实条件；这里只比较算子，不冒充160链整轮。
        def benchmark():
            timings = []
            for _ in range(3):
                torch.cuda.synchronize(); start = time.perf_counter()
                trace = sample()
                torch.cuda.synchronize(); sampled = time.perf_counter()
                actor.zero_grad(set_to_none=True)
                with policy_phase(policy):
                    probability = _log_probs(policy, context, trace, 16)
                    (-probability.mean()).backward()
                torch.cuda.synchronize()
                timings.append(dict(sampling_seconds=sampled-start, forward_backward_seconds=time.perf_counter()-sampled))
            actor.zero_grad(set_to_none=True)
            return timings
        report['eager_timings'] = benchmark()
        original_history = actor.history_encoder.forward
        info = install(actor, args.variant); report['execution'] = info
        if args.variant == 'gru':
            adapted = actor.adapt_conditions(context)
            report['history_mask_cases'] = []
            for label in ('actual', 'holes', 'empty'):
                valid = adapted['history_valid'].clone()
                if label == 'holes':valid[:,::3] = False
                if label == 'empty':valid[:] = False
                history = adapted['history'].detach().clone().requires_grad_(True)
                expected = original_history(history,valid,adapted['history_relative_times'])
                actual = actor.history_encoder(history,valid,adapted['history_relative_times'])
                torch.testing.assert_close(actual,expected,atol=1e-6,rtol=1e-4)
                gradient, = torch.autograd.grad(actual.sum(),history)
                if not bool((gradient[~valid]==0).all()):raise ValueError('Fused GRU leaked invalid history gradient')
                if label == 'empty' and not bool((actual==0).all()):raise ValueError('Empty history must be exactly zero')
                report['history_mask_cases'].append(dict(case=label,max_abs=float((actual-expected).abs().max()),
                    invalid_input_gradient_exact_zero=True))
        torch.cuda.reset_peak_memory_stats()
        prepared = time.perf_counter()
        own = sample()
        with torch.no_grad(), policy_phase(policy):
            probability = _log_probs(policy, context, own, 16)
            cross = _log_probs(policy, context, reference, 16)
        error = probability-own['old_log_probs'][0]
        oracle = torch.distributions.Normal(own['old_means'].double(), own['old_stds'].double()).log_prob(own['chain'][:,1:].double())
        oracle = oracle.masked_fill(~own['free_mask'][:,None], 0.).sum((-2,-1))
        report.update(max_logprob_error=float(error.abs().max()),max_ratio_error=float(torch.expm1(error).abs().max()),
            independent_gaussian_error=float((oracle-own['old_log_probs']).abs().max()),
            cross_eager_logprob_error=float((cross-baseline).abs().max()),
            masks_equal=bool(torch.equal(own['free_mask'],reference['free_mask'])),
            parameter_aliases_and_freeze_preserved=before_names==[(n,id(p),p.requires_grad) for n,p in actor.named_parameters()],
            preparation_seconds=time.perf_counter()-prepared)
        report['gradient_and_adam'] = compare_learning_execution(policy,context,microbatch=16)
        report['candidate_timings'] = benchmark()
        report['peak_memory_bytes'] = torch.cuda.max_memory_allocated()
        report['status'] = 'passed_self_probability' if all((report['max_logprob_error']<=1e-4,
            report['max_ratio_error']<=1e-3,report['independent_gaussian_error']<=1e-8,
            report['masks_equal'],report['parameter_aliases_and_freeze_preserved'])) else 'rejected_probability_or_identity'
    except Exception as error:
        report['error'] = f'{type(error).__name__}: {error}'
        report['error_traceback'] = traceback.format_exc()
    finally:
        args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.with_name(args.output.stem+f'.rank{rank:02d}.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    reports=[None]*8; dist.all_gather_object(reports,report)
    if rank==0:
        args.output.write_text(json.dumps(dict(variant=args.variant,ranks=reports),ensure_ascii=False,indent=2)+'\n')
        print(json.dumps([dict(rank=r['rank'],status=r['status'],error=r.get('error')) for r in reports]),flush=True)
    dist.destroy_process_group()


if __name__=='__main__':main()
