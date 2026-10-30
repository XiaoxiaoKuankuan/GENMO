"""Stage2-A 音乐输入到真实 MuJoCo 控制的装配与交互运行。

任意音频先复用 EDGE35 提取和冻结 Stage1 自主生成器，缓存原始世界动作及真实
生成的导数尾帧；生成历史只来自已生成的名义动作。这是固定参考 Stage2-A 演示，
不是训练时 Newton 采集的配对评价，也不宣称接入 Stage2-B 在线物理历史生成。
执行复用 VectorBank/VectorControl，包括 2654 输入、10×27 残差、SO(3)、Goal570、
历史和锚点更新。只加载 Adapter/冻结 Actor 进行控制，不调用 WM/Critic 或优化器。
蓝色透明影子始终是未修正 GENMO 参考，实体由 MuJoCo 推进；B 切换零 Adapter 后
从相同初态重播。每次运行独立保存配置身份、逐帧真实状态、动作和耗时，不覆盖训练。
"""

import hashlib
import json
from pathlib import Path
import queue
import time
from types import SimpleNamespace

import numpy as np
import torch

from supermimic.bridge.bundle import file_digest, load_bundle
from supermimic.bridge.runtime import SuperMimicRuntime
from .music_reward import AudioTrack, MusicReward
from .policy import ReferenceAdapter
from .rewards import Rewards, configure_sm_mdp, sm_mdp_contract
from .reference import make_nominal
from .vector_bank import VectorBank
from .vector_control import VectorControl
from .mujoco_env import MujocoPhysicalEnv, NominalGhost


def load_controller(config, checkpoint, device):
    """仅装配推理所需网络；checkpoint 内训练配置决定根通道和观测合同。"""
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    training = saved["config"]
    settings = training.get("alternating", {})
    if settings.get("projection") or settings.get("constraint_beta", 1.) != 1.:
        raise ValueError("此入口针对当前无硬投影模型，旧投影模型须使用对应实现")
    bundle = load_bundle(config["supermimic"]["checkpoint"], device,
                         robot_asset_dir=config["supermimic"]["robot_asset_dir"])
    runtime = SuperMimicRuntime(bundle, device)
    configure_sm_mdp(runtime, training.get("sm_mdp", {}))
    adapter = ReferenceAdapter(bundle.spec).to(device).eval()
    adapter.load_state_dict(saved["adapter"], strict=True)
    adapter.requires_grad_(False)
    adapter.disable_root_xy = not settings.get("root_xy", False)
    baseline = torch.load(config["music_baseline"], map_location="cpu", weights_only=False)
    if (baseline.get("source") != "zero_adapter_real_training_episodes" or
            tuple(baseline["joint_names"]) != bundle.spec.joint_names):
        raise ValueError("音乐历史标尺必须使用相同机器人训练基线")
    music = MusicReward(baseline["s0"], **{k: v for k, v in training["music"].items()
                                        if k not in ("sigma_s", "lag_steps")})
    rewards = Rewards(runtime, music, training["rewards"])
    step = SimpleNamespace(adapter=adapter, bundle=bundle, runtime=runtime,
                           rewards=rewards, observation_noise=False)
    engine = SimpleNamespace(step=step)
    # 保留真正的已训练参数供 B 切换；零模式通过 propose(zero=True)执行。
    return engine, settings, saved.get("state", {})


@torch.no_grad()
def prepare_reference(config, prepared, bundle, seed, device):
    """内容寻址的固定名义缓存；音乐末尾额外帧由 GENMO 生成而非线性外推。"""
    from .nominal import load_frozen_genmo
    from tools.eval.generate_stage1_music_dances import generate
    identity = dict(schema="genmo.sm_adapter.mujoco.nominal.v1", seed=seed,
                    audio_sha256=prepared.identity["audio_sha256"],
                    feature_sha256=prepared.identity["feature_values_sha256"],
                    assets={k: file_digest(config["genmo"][k]) for k in ("checkpoint", "stats", "kinematics")},
                    method="frozen_stage1_autoregressive_120_prefix12_ddim20_cfg2.5",
                    contact="sigmoid_logits_with_contact_valid",
                    history="past_generated_nominal_only_not_future_executed_state")
    key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    directory = Path(config["cache_root"])/"nominal"/key
    filename = directory/"reference.pt"
    if filename.exists():
        print(f"[REFERENCE] 复用固定 GENMO 缓存 {filename}", flush=True)
        payload = torch.load(filename, map_location="cpu", weights_only=False)
        if payload["identity"] != identity:
            raise ValueError("GENMO 缓存身份不匹配")
    else:
        directory.mkdir(parents=True, exist_ok=True)
        print("[REFERENCE] 首次音乐：冻结 GENMO 生成完整参考；完成后打开物理窗口。", flush=True)
        torch.manual_seed(seed)
        actor = load_frozen_genmo(config["genmo"], device).stage1
        music = torch.from_numpy(prepared.features).float()
        # 50Hz 最后时刻的 t+11 需要至少 0.22s 真正生成的名义尾部。
        qpos, contact, contact_valid, windows = generate(actor, actor, music, len(music)+8, seed, directory)
        payload = dict(identity=identity, world_qpos=qpos[None], joint_names=actor.endecoder.kinematics.joint_order,
                       contact=(contact.sigmoid()*contact_valid[:, None])[None],
                       contact_valid=contact_valid, windows=windows)
        temporary = directory/"reference.pending.pt"
        torch.save(payload, temporary)
        temporary.replace(filename)
        del actor
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    nominal = make_nominal(payload["world_qpos"].to(device), payload["joint_names"], bundle.spec,
                           contact=payload["contact"].to(device))
    return nominal, dict(path=str(filename), **identity)


def run(args, config, output):
    from gem.closedloop.sim2sim.music import Track, prepare_music, AudioPlayer
    import mujoco

    device = torch.device(args.device)
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    prepared = prepare_music(Track(args.audio.resolve(), args.audio.stem),
                             Path(config["cache_root"])/"music", decode_audio=not args.mute)
    checkpoint = args.adapter_checkpoint or Path(config["adapter_checkpoint"])
    engine, settings, checkpoint_state = load_controller(config, checkpoint, device)
    nominal, identity = prepare_reference(config, prepared, engine.step.bundle, args.seed, device)
    audio = AudioTrack.from_features(torch.from_numpy(prepared.features).float().to(device)[None])
    kin = engine.step.runtime.kin
    from supermimic.bridge.reference import build_reference
    from supermimic.mdp import flatten_state
    first = build_reference(nominal.pose_at, torch.tensor([0], device=device), nominal.lower, nominal.upper, kin)
    initial = flatten_state((first.root_p[:, 0], first.root_R[:, 0], first.root_v[:, 0], first.root_w[:, 0],
                             first.q[:, 0], first.qd[:, 0]))
    task = dict(id=args.audio.stem, split="interactive")
    bank = VectorBank.from_entries([(task, initial, nominal, audio)], kin, engine.step.rewards.music)
    control = VectorControl(engine, bank)
    control.disable_root_orientation = not settings.get("root_orientation", True)
    control.soft_joint_limit_cost = settings.get("soft_joint_limit", False)
    env = MujocoPhysicalEnv(engine.step.bundle.spec, device)
    ghost = NominalGhost(env)
    keys = queue.SimpleQueue()
    viewer = None
    if not args.headless:
        import mujoco.viewer
        viewer = mujoco.viewer.launch_passive(env.model, env.data, key_callback=keys.put)
        viewer.cam.distance, viewer.cam.azimuth, viewer.cam.elevation = 2.7, 130, -15
    player = AudioPlayer(prepared, enabled=not args.mute)
    launch = dict(argv=vars(args), checkpoint=str(checkpoint), checkpoint_sha256=file_digest(checkpoint),
                  checkpoint_updates=checkpoint_state.get("accepted_updates"), reference=identity,
                  physics_backend="mujoco_cpu", control_hz=50, physics_hz=200,
                  action_noise=0., reference_target="original_GENMO_nominal", wm_forward_calls=0,
                  sm_checkpoint_sha256=file_digest(config["supermimic"]["checkpoint"]),
                  robot=engine.step.bundle.metadata, sm_reward=sm_mdp_contract(engine.step.runtime),
                  caveat="MuJoCo demo, not paired Newton training evaluation; no automatic recovery after falls")
    (output/"launch.json").write_text(json.dumps(launch, default=str, ensure_ascii=False, indent=2))
    print(f"[OUTPUT] {output}\n[KEYS] 空格暂停 | R 同初态重播 | B 零Adapter/已训练模型切换并重播 | G 蓝色参考 | Esc 退出", flush=True)
    print("[VIEW] 蓝色=原始GENMO；实体=Actor/Adapter控制的MuJoCo。摔倒记录但不自动扶正。", flush=True)
    zero, paused, show_ghost, restart = args.zero_adapter, False, True, True
    episodes, records, timings, episode, first_failure = [], [], [], -1, None
    i, limit, reason = 0, int(bank.ends[0]), "window_closed"
    if args.max_seconds:
        limit = min(limit, round(args.max_seconds*50))
    deadline = time.perf_counter()

    def save_episode(why):
        if not records:
            return
        data = {k: np.stack([row[k] for row in records]) for k in records[0]}
        np.savez_compressed(output/f"episode_{episode:03d}.npz", **data)
        summary = dict(episode=episode, zero_adapter=zero, steps=len(records), simulated_seconds=len(records)*.02,
                       reason=why, first_failure_seconds=first_failure,
                       control_physics_metrics_ms_p50=float(np.percentile(timings, 50)),
                       control_physics_metrics_ms_p95=float(np.percentile(timings, 95)),
                       control_physics_metrics_ms_max=float(max(timings)),
                       compute_over_20ms_fraction=float(np.mean(np.asarray(timings)>20)),
                       root_rmse_m=float(np.sqrt(np.mean((data["state"][:, :3]-data["nominal_p"])**2)*3)),
                       joint_rmse_rad=float(np.sqrt(np.mean((data["state"][:, 13:34]-data["nominal_q"])**2))),
                       total_sm_reward=float(data["sm_reward"].sum()), audio=player.snapshot())
        episodes.append(summary)
        (output/"summary.json").write_text(json.dumps(episodes, ensure_ascii=False, indent=2))
        print("[EPISODE] "+json.dumps(summary, ensure_ascii=False), flush=True)

    try:
        with torch.inference_mode():
            while viewer is None or viewer.is_running():
                stop = False
                while not keys.empty():
                    key = keys.get()
                    if key == 256: stop = True
                    elif key == 32: paused = not paused
                    elif key in (ord("R"), ord("r")): restart = True
                    elif key in (ord("G"), ord("g")): show_ghost = not show_ghost
                    elif key in (ord("B"), ord("b")):
                        save_episode("switch_model")
                        records, timings = [], []
                        zero, restart = not zero, True
                if stop:
                    break
                if restart:
                    save_episode("restart")
                    records, timings, first_failure = [], [], None
                    episode += 1
                    z = control.initial(env.reset(initial), torch.zeros(1, device=device, dtype=torch.long),
                                        torch.zeros(1, device=device, dtype=torch.long),
                                        torch.full((1,), episode, device=device, dtype=torch.long))
                    i, paused, restart = 0, False, False
                    reason = "window_closed"
                    if player.enabled:
                        player.resync(0.)
                    deadline = time.perf_counter()
                if not paused and i < limit:
                    started = time.perf_counter()
                    proposal = control.propose(z, zero=zero)
                    actual = env.step_physics(proposal["action"])
                    z, _, raw, _, _, _, failed = control.advance(z, proposal, actual)
                    ref = bank.ref(z["song"], z["frame"])
                    if bool(failed[0]) and first_failure is None:
                        first_failure = (i+1)*.02
                        print(f"[FAILURE] {first_failure:.2f}s 触发原名义参考终止条件；继续显示真实物理，不扶正。", flush=True)
                    # 转CPU既用于逐帧记录，也完成CUDA同步；计时不是异步提交时间。
                    row = dict(state=actual[0].cpu().numpy().copy(), action=proposal["action"][0].cpu().numpy().copy(),
                               residual=proposal["residual"][0].cpu().numpy().copy(),
                               goal=proposal["goal"][0].cpu().numpy().copy(),
                               nominal_p=ref.root_p[0].cpu().numpy().copy(), nominal_q=ref.q[0].cpu().numpy().copy(),
                               sm_reward=raw["sm"][0].cpu().numpy().copy(), time_s=np.array((i+1)*.02))
                    records.append(row)
                    timings.append((time.perf_counter()-started)*1000)
                    i += 1
                    if i % 250 == 0:
                        print(f"[RUN] {i*.02:.1f}/{limit*.02:.1f}s | {'零Adapter' if zero else '已训练Adapter'} | 控制/物理/统计P50={np.median(timings[-250:]):.2f}ms", flush=True)
                player.update(i*.02, not paused and i < limit)
                if viewer is not None:
                    with viewer.lock():
                        viewer.cam.lookat[:] = env.data.qpos[:3]
                        viewer.cam.lookat[2] = max(.35, viewer.cam.lookat[2])
                        viewer.user_scn.ngeom = 0
                        if show_ghost:
                            ghost.update(bank.ref(z["song"], z["frame"]))
                            ghost.append(viewer.user_scn, viewer.cam)
                    viewer.sync()
                if i >= limit:
                    reason = "administrative_max_seconds" if args.max_seconds and i < int(bank.ends[0]) else "music_end"
                    if args.headless or args.max_seconds:
                        break
                    if not paused:
                        save_episode(reason)
                        records, timings = [], []
                        paused = True
                        print("[END] 歌曲结束，R 重播，B 切换对照，Esc 退出。", flush=True)
                if paused:
                    time.sleep(.02)
                    deadline = time.perf_counter()
                elif not args.fast:
                    deadline += .02
                    time.sleep(max(0., deadline-time.perf_counter()))
    except BaseException as exc:
        reason = f"exception:{type(exc).__name__}:{exc}"
        raise
    finally:
        save_episode(reason)
        player.close()
        if viewer is not None:
            viewer.close()
    return episodes
