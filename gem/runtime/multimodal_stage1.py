"""音乐或英文文本到 BUMI 完整名义参考的统一 PyTorch 推理入口。

本模块加载共享 Stage1 checkpoint 及其唯一动作统计量，将 EDGE35 或冻结 T5 条件
转换为同一纯条件 batch，再用已生成动作构造因果历史和十二帧承诺前缀进行滚动。
输入每次只能选择一种语义；不会读取训练标签、实际未来状态或以旧音乐模型替代联合
模型。机器人初态可由具名 state55 或标准贴地站姿提供，输出始终是世界系 qpos28。

Stage2 所需尾部支持也由模型继续生成；音频结束后的音乐 mask 为假，不复制末帧
动作充当 halo。前缀启动属于显式固定初态，contact_valid 标记其非预测接触。这里只
生成固定参考，不执行物理、不调用 Adapter/WM，不代表实时闭环或摄像头能力。
"""

from __future__ import annotations

import hashlib
import math
from pathlib import Path

import torch
from omegaconf import OmegaConf

from gem.closedloop.checkpoint import load_stage1_checkpoint
from gem.closedloop.stage1_dataset import CausalDemoProprio48Builder
from gem.closedloop.training import batch_to_device, build_stage1_actor
from gem.robots.bumi.feature_codec import make_quaternion_continuous
from gem.robots.bumi.kinematics import sha256_file


def load_actor(config, device):
    """仅按 checkpoint 内模型配置加载，不要求推理机器持有训练数据集。"""
    payload = torch.load(config["checkpoint"], map_location="cpu", weights_only=False)
    cfg = OmegaConf.create(payload["config"])
    cfg.endecoder.stats_path = str(Path(config["stats"]).expanduser().resolve())
    cfg.endecoder.kinematics_path = str(Path(config["kinematics"]).expanduser().resolve())
    data = OmegaConf.create(dict(
        datasets={}, sample_contract=dict(history_steps=int(cfg.sample_contract.history_steps)),
        qpos30_stats=dict(path=cfg.endecoder.stats_path),
        dataset_defaults=dict(kinematics_path=cfg.endecoder.kinematics_path),
    ))
    actor = build_stage1_actor(cfg, data)
    load_stage1_checkpoint(actor, payload)
    actor.__dict__["_text_encoder_assets"] = payload.get("runtime_state", {}).get(
        "training_contract", {}).get("text_encoder_assets")
    return actor.to(device).eval().requires_grad_(False), cfg


def validate_request(task):
    """在输入边界确定唯一模态、任务时钟与时长，内部不重复做输入判定。"""
    has_music = task.get("music_features") is not None
    has_text = isinstance(task.get("text"), str) and bool(task["text"].strip())
    if has_music == has_text:
        raise ValueError("每次必须且只能提供 music_features 或非空英文 text")
    kind = "music" if has_music else "text"
    if task.get("condition_kind", kind) != kind:
        raise ValueError("condition_kind 与实际输入不一致")
    start = int(task.get("start_tick", task.get("music_start_tick", 0)))
    if start < 0 or start % 60:
        raise ValueError("start_tick 须为非负且同时对齐 30/50 Hz 的 60 tick 倍数")
    duration = task.get("duration_ticks")
    if duration is None and task.get("duration_s") is not None:
        seconds = float(task["duration_s"])
        if not math.isfinite(seconds) or seconds <= 0:
            raise ValueError("duration_s 必须为有限正数")
        duration = round(seconds * 600)
    if duration is not None and (int(duration) != duration or duration < 12):
        raise ValueError("duration_ticks 至少为一个控制周期 12 tick")
    if kind == "text" and duration is None:
        raise ValueError("文本生成必须显式提供 duration_s 或 duration_ticks")
    return kind, start, None if duration is None else int(duration)


def initial_qpos(actor, task, joint_names=None):
    """将 state55 的具名关节转换到 Stage1 顺序；没有初态时使用真实几何站姿。"""
    kin = actor.endecoder.kinematics
    if not task.get("initial_state"):
        return kin.make_standing_qpos().cpu().reshape(28)
    path = Path(task["initial_state"])
    if path.suffix == ".npy":
        import numpy as np
        state = torch.from_numpy(np.load(path, allow_pickle=False))
    else:
        state = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(state, dict):
        joint_names = state.get("joint_names", joint_names)
        state = state["state"]
    state = torch.as_tensor(state, dtype=torch.float32).reshape(-1, 55)
    if len(state) != 1 or not torch.isfinite(state).all():
        raise ValueError("每个参考任务必须提供一个有限 state55 初态")
    if joint_names is None:
        raise ValueError("state55 初态需要 robot_spec_joint_names，不能假定关节顺序")
    if len(joint_names) != 21 or set(joint_names) != set(kin.joint_order):
        raise ValueError("state55 与 Stage1 的 BUMI 具名关节不一致")
    order = [list(joint_names).index(name) for name in kin.joint_order]
    return torch.cat((state[0, :7], state[0, 13:34][order]))


@torch.no_grad()
def generate_sequence(actor, *, total_frames, seed=42, music=None, text_features=None,
                      text_attention_mask=None, initial=None, start_tick=0, steps=20,
                      guidance_scale=2.5):
    """滚动生成共享世界轨迹；返回概率、有效mask及可追溯窗口，支持测试注入小模型。"""
    if (music is None) == (text_features is None):
        raise ValueError("generate_sequence 只接受一种条件")
    if total_frames < 2:
        raise ValueError("生成序列至少需要两帧")
    device = next(actor.parameters()).device
    codec, kin = actor.endecoder.codec, actor.endecoder.kinematics
    history_builder = CausalDemoProprio48Builder(kin)
    initial = kin.make_standing_qpos().cpu().reshape(28) if initial is None else initial.cpu()
    generated = initial.repeat(min(12, total_frames - 1), 1)
    contacts = torch.zeros(len(generated), 2)
    contact_valid = torch.zeros(len(generated), dtype=torch.bool)
    windows, decision = [], 0
    if music is not None:
        music = torch.as_tensor(music, dtype=torch.float32).cpu()
        if music.ndim != 2 or music.shape[1] != 35 or not torch.isfinite(music).all():
            raise ValueError("音乐特征必须为有限 [T,35]")
    # 联合模型必须显式携带缺失文本，旧音乐模型仍可用于历史基线生成。
    multimodal = bool(getattr(actor.denoiser, "encode_text", False))
    if text_features is not None and not multimodal:
        raise ValueError("该 checkpoint 没有文本条件分支")
    while len(generated) < total_frames:
        count = min(120, total_frames - decision)
        prefix = generated[decision:decision + 12]
        encoded = codec.encode(prefix)
        known = torch.zeros(120, 30)
        mask = torch.zeros(120, 30, dtype=torch.bool)
        mask[:len(prefix), 2:] = True
        mask[:len(prefix) - 1, :2] = True
        known[:len(prefix)] = encoded.physical_features
        known[~mask] = 0
        history, history_valid, history_times = history_builder.build_history(
            generated[:decision + 1], decision_frame=decision, history_steps=actor.history_steps,
        )
        music_window, music_valid = torch.zeros(120, 35), torch.zeros(120, dtype=torch.bool)
        if music is not None:
            music_count = max(0, min(count, len(music) - decision))
            music_window[:music_count] = music[decision:decision + music_count]
            music_valid[:music_count] = True
        time_offset = start_tick / 600
        conditions = dict(
            music_features=music_window[None], music_valid=music_valid[None],
            proprio_history=history[None], proprio_history_valid=history_valid[None],
            proprio_history_times=(history_times + time_offset)[None],
            known_qpos30=known[None], known_qpos30_mask=mask[None],
            future_valid=(torch.arange(120) < count)[None],
            future_times=((decision + torch.arange(120, dtype=torch.float64)) / 30 + time_offset)[None],
            decision_time=torch.tensor([decision / 30 + time_offset], dtype=torch.float64),
        )
        if multimodal:
            conditions.update(
                text_features=torch.zeros(1, 150, 1024) if text_features is None else text_features,
                text_attention_mask=torch.zeros(1, 150, dtype=torch.bool)
                if text_features is None else text_attention_mask,
                has_text=torch.tensor([text_features is not None]),
            )
        conditions = batch_to_device(conditions, device)
        noise = torch.randn((1, 120, 30), generator=torch.Generator().manual_seed(seed + len(windows)))
        result = actor.sample(conditions, steps=steps, guidance_scale=guidance_scale, noise=noise.to(device))
        error = float((result["qpos30"][conditions["known_qpos30_mask"]]
                       - known[None].to(device)[conditions["known_qpos30_mask"]]).abs().max())
        if error != 0:
            raise RuntimeError(f"Stage1 改写了承诺前缀: {error}")
        anchor = encoded.anchor
        segment = codec.apply_world_anchor(result["qpos"][0, :count].cpu(), dict(
            root_xy=anchor.position_w[..., :2], yaw=anchor.yaw, anchor_z=anchor.default_root_height,
        ))
        generated = torch.cat((generated, segment[len(prefix):]), 0)
        contacts = torch.cat((contacts, result["contact"][0, len(prefix):count].cpu()), 0)
        contact_valid = torch.cat((contact_valid, torch.ones(count - len(prefix), dtype=torch.bool)))
        windows.append(dict(decision_frame=decision, valid_frames=count,
                            seed=seed + len(windows), prefix_max_abs_error=error))
        decision = len(generated) - 12
    generated[:, 3:7] = make_quaternion_continuous(generated[:, 3:7])
    if not torch.isfinite(generated).all() or not torch.isfinite(contacts).all():
        raise FloatingPointError("生成轨迹包含非有限数值")
    return dict(world_qpos=generated[None], contact=contacts[None], contact_valid=contact_valid,
                joint_names=tuple(kin.joint_order), start_tick=start_tick, windows=windows)


@torch.no_grad()
def generate_stage1_reference(genmo_cfg, task, *, device, required_end_tick):
    """供 Stage2 缓存与CLI共用：根据明确任务时长生成足够的真实模型尾部支持。"""
    kind, start, duration = validate_request(task)
    actor, checkpoint_cfg = load_actor(genmo_cfg, device)
    music, features, token_mask = None, None, None
    condition = dict(condition_kind=kind)
    if kind == "music":
        path = Path(task["music_features"])
        if path.suffix == ".npy":
            import numpy as np
            music = torch.from_numpy(np.load(path, allow_pickle=False)).float()
        else:
            music = torch.load(path, map_location="cpu", weights_only=False)
        music = torch.as_tensor(music, dtype=torch.float32)
        if music.ndim != 2 or music.shape[0] < 2 or music.shape[1] != 35 or not torch.isfinite(music).all():
            raise ValueError("音乐特征文件必须为有限 [T>=2,35]，不能自动改排时间或通道")
        duration = (len(music) - 1) * 20 if duration is None else duration
        if duration > len(music) * 20:
            raise ValueError("任务时长超过实际音乐特征覆盖")
        condition["music_sha256"] = sha256_file(path)
    else:
        from gem.runtime.frozen_text_encoder import FrozenTextEncoder
        model_path = genmo_cfg.get("text_encoder_path", checkpoint_cfg.model.get("text_encoder_path"))
        if not model_path:
            raise ValueError("文本生成需要本地 text_encoder_path")
        encoder = FrozenTextEncoder(model_path, micro_batch_size=1, cache_entries=1)
        expected = actor.__dict__.get("_text_encoder_assets")
        actual = {name: {"sha256": asset["sha256"]} for name, asset in encoder.assets.items()}
        if expected != actual:
            raise ValueError("文本编码器内容与完整 Stage1 checkpoint 的训练身份不一致")
        features, token_mask = encoder.encode([task["text"]], device)
        condition.update(text_sha256=hashlib.sha256(task["text"].strip().encode()).hexdigest(),
                         text_encoder_assets=encoder.assets)
        del encoder
    end = start + duration
    total = math.ceil((max(required_end_tick, end) - start) / 20) + 1
    result = generate_sequence(
        actor, total_frames=total, seed=int(task.get("seed", 42)), music=music,
        text_features=features, text_attention_mask=token_mask,
        initial=initial_qpos(actor, task, genmo_cfg.get("robot_spec_joint_names")), start_tick=start,
        steps=int(genmo_cfg.get("sample_steps", 20)), guidance_scale=float(genmo_cfg.get("guidance_scale", 2.5)),
    )
    result["generation_contract"] = dict(
        **condition, task_end_tick=end, duration_ticks=duration,
        assets={key: sha256_file(genmo_cfg[key]) for key in ("checkpoint", "stats", "kinematics")},
        history="past_generated_nominal_only", generation_mode="fixed_autoregressive_reference",
        sample_steps=int(genmo_cfg.get("sample_steps", 20)), guidance_scale=float(genmo_cfg.get("guidance_scale", 2.5)),
        prefix_frames=12, seed=int(task.get("seed", 42)), required_end_tick=required_end_tick,
        windows=result.pop("windows"),
    )
    return result
