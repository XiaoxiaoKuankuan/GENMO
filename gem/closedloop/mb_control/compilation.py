"""可选的原网络FP32编译与显式预热，减少真实生成等待而不更改采样算法。

编译当前Stage1去噪器、原历史编码器及同一个冻结SM Actor/WM的计算，不创建替代
模型或优化器，不使用TF32/bf16加速、不减少DDIM步数。Module.compile保留原参数
和state_dict；WM仅编译step函数，仍可对输入求导。部署预热在正式物理episode前
显式完成并单独计时，不能把首次编译延迟藏进已有测量，也不能用预热证明物理质量。
启用后必须重新验证输入梯度和完整部署延迟；默认配置保持未编译，源码仅作记录。
"""

import time

import torch


def compile_networks(actor, adapter):
    # AOTAutograd默认捐赠反向缓冲区，不能用于执行探针、Critic输入导数与随后loss
    # backward共享同一图的retain_graph路径。关闭的只是缓存复用，不是输入求导。
    from torch._functorch import config as aot_config
    aot_config.donated_buffer = False
    # 同一冻结网络在BPTT图中被反复调用，保存的激活不能被后续调用复用写入。
    from torch._inductor import config as inductor_config
    inductor_config.inplace_buffers = False
    inductor_config.allow_buffer_reuse = False
    actor.stage1.denoiser.compile(fullgraph=True, dynamic=False)
    actor.stage1.history_encoder.compile(fullgraph=True, dynamic=False)
    adapter.bundle.actor.compile(fullgraph=True, dynamic=False)
    adapter.bundle.world_model.step = torch.compile(adapter.bundle.world_model.step, fullgraph=True, dynamic=False)


def warmup_inference(actor, adapter, scheduler, state):
    from .forecast import forecast_rollout
    ticket = scheduler.reserve(state)
    generator = torch.Generator(device=state.state.device).manual_seed(710)
    noise = torch.randn(1, 60, 30, device=state.state.device, generator=generator)
    started = time.perf_counter()
    with torch.no_grad():
        for _ in range(3):
            context = forecast_rollout(state, adapter, ticket.first_influence_tick,
                                       reuse_current_observation=getattr(actor,"fast_forecast",False)).context
            actor.generate(state, ticket, context, noise)
    if state.state.is_cuda:
        torch.cuda.synchronize(state.state.device)
    return {"seconds": time.perf_counter() - started, "calls": 3, "before_physics_episode": True,
            "scope": "nominal_tensor_compute_warmup_not_physics_evidence", "precision": "fp32"}
