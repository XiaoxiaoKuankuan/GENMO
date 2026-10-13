"""服务器1八rank执行的新精度与批量条件图回归。

测试明确选择LOCAL_RANK对应GPU，比较融合GRU与原GRUCell的值和全部参数梯度，
验证无效/全空历史语义、SDPA mask和eval dropout，以及条件叶子的梯度能回传原图。
这里使用缩小架构隔离逻辑错误；真实模型全部概率、梯度和吞吐另由八卡工具验收，
不把单元测试通过解释为大规模训练可发布。无CUDA时显式跳过并由验收汇总标记。
"""
import copy
import os
from types import SimpleNamespace
import pytest
import torch
from gem.closedloop.actor import TemporalHistoryEncoder
from gem.closedloop.dppo.numerical_execution import precision_scope
from gem.closedloop.dppo.policy import DPPODiffusionPolicy
from gem.closedloop.dppo.tensor_cache import ConditionGraphCache
from gem.network.base_arch.transformer.encoder_rope import RoPEAttention
from tests.closedloop.test_stage1_actor import actor_factory, _conditions, _activate_branches


@pytest.fixture
def device():
    if not torch.cuda.is_available(): pytest.skip('Requires Server1 CUDA acceptance')
    target = f"cuda:{int(os.environ.get('LOCAL_RANK', 0))}"
    torch.cuda.set_device(target)
    return target


def test_batched_fused_gru_mask_and_all_gradients(device):
    torch.manual_seed(147)
    reference = TemporalHistoryEncoder(32, 16).to(device)
    torch.nn.init.normal_(reference.out_proj.weight, std=.02)
    candidate = copy.deepcopy(reference); candidate.execution_backend = 'sample_bmm_fused_gru'
    history = torch.randn(17, 50, 48, device=device)
    valid = torch.rand(17, 50, device=device)>.3; valid[0] = False
    times = torch.arange(50, device=device).float()[None].expand(17,-1)/50-1
    a, b = reference(history, valid, times), candidate(history, valid, times)
    torch.testing.assert_close(a, b, atol=3e-6, rtol=2e-5)
    assert torch.equal(a[0], torch.zeros_like(a[0])) and torch.equal(b[0], torch.zeros_like(b[0]))
    a.square().sum().backward(); b.square().sum().backward()
    for x, y in zip(reference.parameters(), candidate.parameters()):
        torch.testing.assert_close(x.grad, y.grad, atol=3e-5, rtol=2e-4)
    altered = torch.where(valid[...,None], history, torch.full_like(history, 991.))
    torch.testing.assert_close(candidate(altered, valid, times), b, rtol=0, atol=0)


@pytest.mark.parametrize('dimension,heads', [(64,4),(1024,8)])
def test_forced_sdpa_mask_and_eval_dropout(device, dimension, heads):
    reference = RoPEAttention(dimension, heads, dropout=.7).to(device).eval()
    candidate = copy.deepcopy(reference); candidate.execution_backend = 'sdpa_math'
    x = torch.randn(5, 11, dimension, device=device)
    blocked = torch.ones(5,11,device=device,dtype=torch.bool); blocked[:, :6] = False
    a, b = reference(x, key_padding_mask=blocked), candidate(x, key_padding_mask=blocked)
    torch.testing.assert_close(a, b, atol=2e-6, rtol=2e-5)
    torch.testing.assert_close(candidate(x, key_padding_mask=blocked), b, atol=0, rtol=0)
    assert candidate.last_execution_backend == 'sdpa_math'


def test_batched_condition_graph_restores_encoder_gradients(actor_factory, device):
    actor, batch = actor_factory(starts=(45,47))
    actor.to(device); _activate_branches(actor)
    policy = DPPODiffusionPolicy(actor, cfg_batch=True, numerical_layout='sample_matrix_bmm_fp32.v1',
                                precision_mode='fp32_fast', defer_checks=True)
    context = {key:value.to(device) for key,value in _conditions(batch).items()}
    rows = [SimpleNamespace(context={key:v[i:i+1] for key,v in context.items()}) for i in range(2)]
    direct = policy.prepare_conditions(context)
    (direct['conditional'].square().sum()+direct['unconditional'].square().sum()).backward()
    expected = {name:p.grad.clone() for name,p in actor.named_parameters() if p.grad is not None}
    actor.zero_grad(set_to_none=True)
    graph = ConditionGraphCache(policy); graph.prime(rows)
    assert len(graph.bank['originals']) == 1 and graph.bank['batched']
    for _ in range(20):
        prepared = graph.prepare(rows, context)
        ((prepared['conditional'].square().sum()+prepared['unconditional'].square().sum())/20).backward()
    graph.backward()
    for name,p in actor.named_parameters():
        if name in expected: torch.testing.assert_close(p.grad, expected[name], atol=3e-5, rtol=2e-4)
    assert all(any(name.startswith(prefix) for name in expected) for prefix in
               ('history_encoder','prefix_encoder','music_embedder'))
    assert policy.kernel_config['version'].endswith('.v5')


def test_precision_scope_restores_global_switch():
    original = torch.backends.cuda.matmul.allow_tf32
    with precision_scope(dict(matmul_tf32=not original)):
        assert torch.backends.cuda.matmul.allow_tf32 is not original
    assert torch.backends.cuda.matmul.allow_tf32 is original


def test_explicit_variant_bound_before_sampling_and_wrong_mode_rejected(actor_factory, device):
    from gem.closedloop.dppo.batch_execution import SampleMatrixLinear
    actor, _ = actor_factory(starts=(45,47)); actor.to(device)
    policy = DPPODiffusionPolicy(actor, cfg_batch=True, numerical_layout='sample_matrix_bmm_fp32.v1',
        precision_mode='fp32_fast', numerical_variant='blocked64_fp32_gemm', defer_checks=True)
    contract = policy.kernel_config['numerical_execution']
    assert contract['variant'] == 'blocked64_fp32_gemm' and contract['network_row_capacity'] == 64
    assert all(module.forward_backend == 'blocked64_fp32_gemm'
        for name, module in actor.denoiser.named_modules()
        if isinstance(module, SampleMatrixLinear) and not name.startswith('embed_timestep.'))
    with pytest.raises(ValueError, match='fp32_fast'):
        DPPODiffusionPolicy(actor, numerical_layout='sample_matrix_bmm_fp32.v1',
            precision_mode='fp32_reference', numerical_variant='blocked64_fp32_gemm')
    with pytest.raises(ValueError, match='Unknown explicit numerical variant'):
        DPPODiffusionPolicy(actor, numerical_layout='sample_matrix_bmm_fp32.v1',
            precision_mode='fp32_fast', numerical_variant='silent_fallback')


def test_compiled_candidate_keeps_bc_dropout_forward_gradients_and_rng(device, monkeypatch):
    from gem.closedloop.dppo.numerical_execution import compile_fixed_denoiser
    class Denoiser(torch.nn.Module):
        def __init__(self):
            super().__init__(); self.linear=torch.nn.Linear(4,4); self.dropout=torch.nn.Dropout(.1)
        def forward(self,x,timesteps,y=None,inputs=None):
            value=self.dropout(self.linear(x)+y['f_cond'])
            return dict(pred_x_start=value,static_conf_logits=value[..., :2])
    model=Denoiser().to(device).train(); reference=copy.deepcopy(model); calls=[]
    def compile_spy(function,**kwargs):
        def wrapped(*args):calls.append(1);return function(*args)
        return wrapped
    monkeypatch.setattr(torch,'compile',compile_spy)
    policy=SimpleNamespace(actor=SimpleNamespace(denoiser=model),numerical_execution={})
    compile_fixed_denoiser(policy)
    x=torch.randn(5,3,4,device=device);t=torch.zeros(5,dtype=torch.long,device=device)
    y=dict(f_cond=torch.randn_like(x),length=torch.full((5,),3,device=device))
    rng=torch.cuda.get_rng_state(device)
    expected=reference(x,t,y=y,inputs={});after=torch.cuda.get_rng_state(device)
    torch.cuda.set_rng_state(rng,device)
    actual=model(x,t,y=y,inputs={})
    assert torch.equal(torch.cuda.get_rng_state(device),after) and calls==[]
    for key in actual:torch.testing.assert_close(actual[key],expected[key],rtol=0,atol=0)
    actual['pred_x_start'].square().sum().backward();expected['pred_x_start'].square().sum().backward()
    for a,b in zip(model.parameters(),reference.parameters()):torch.testing.assert_close(a.grad,b.grad,rtol=0,atol=0)
