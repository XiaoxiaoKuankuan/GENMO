"""在当前双工作树上启动有界、全冻结的 GENMO—Isaac GMT 闭环验收。

入口先检查真实模型/资产/音乐SHA、设备资源和当前代码来源，再用两个独立解释器启动
Actor与Isaac worker。默认完整评估包含10+100延迟校准、四库各两组/三种子/两模式；
--seconds、--max-episodes及校准数量仅用于明确标记的小规模联调，不冒充完整基线。
所有worker属于本次进程组，finally只关闭本轮服务；不停止已有GPU服务、不创建训练
或自动启动任务。报告及复现配置写入新建实验目录，已有结果拒绝覆盖。--preflight不启动
模型或仿真；可选--video为每个episode启用真实Isaac摄像机，关闭writer后按真实控制帧
同步原验证音乐，校验H264/AAC/50fps后清理无声中间片。关闭后独立审计时序与历史，
同时复核启动前捕获的源码指纹，不把进程正常退出当作动力学验收通过。
训练集专项通过独立配置选择完整 train 清单中的独立音乐，允许无音频、无录像运行，
并显式使用 yaw 分离终止配置。旧四库 val/48集验收仍保持原阈值、选择和判定边界。
--only-sample只从完整确定性清单精确选取指定数据集/样本，供显式补测；记录父清单
SHA和原始数量，不重排、不跨split补样，不把单曲补测冒充整套评估。
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
from datetime import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import signal
import socket
import subprocess
import sys
import tempfile
import time
import traceback

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from gem.closedloop.evaluation_music import (  # noqa: E402
    check_music_files, load_music_features, music_path, select_val_music, select_train_music, sha256_file,
)
from gem.runtime.closedloop_protocol import RpcClient  # noqa: E402
from gem.closedloop.baseline_provenance import (  # noqa: E402
    collect_source_provenance, collect_compute_context, verify_source_provenance,
)


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def select_evaluation_subset(selected, sample_keys=None):
    """在已确定的完整音乐清单上做精确白名单，留下可复核的父清单身份。"""
    keys = [f"{sample['dataset']}/{sample['row']['sample_id']}" for sample in selected]
    if len(set(keys)) != len(keys):
        raise ValueError("Parent music selection contains duplicate dataset/sample keys")
    requested = None if sample_keys is None else list(sample_keys)
    if requested is not None:
        if not requested or len(set(requested)) != len(requested):
            raise ValueError("Explicit sample subset must be nonempty and unique")
        missing = sorted(set(requested) - set(keys))
        if missing:
            raise ValueError(f"Requested samples not in the deterministic parent selection: {missing}")
    parent_bytes = json.dumps(selected, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    result = selected if requested is None else [sample for sample, key in zip(selected, keys) if key in set(requested)]
    return result, {"parent_selection_sha256": hashlib.sha256(parent_bytes).hexdigest(),
                    "parent_music_count": len(selected), "selected_music_count": len(result),
                    "requested_sample_keys": requested,
                    "parent_order_preserved": True}


def _command(args, cwd=None):
    result = subprocess.run(args, cwd=cwd, capture_output=True, text=True, timeout=20, check=True)
    return result.stdout.strip()


ASSET_KEYS = ("checkpoint", "stats", "kinematics", "gmt_policy", "compat_profile", "isaac_contract")
FULL_DATASETS = {"AIST++", "AIOZ-GDANCE", "FineDance", "Mine"}


def validate_config(config):
    """在加载大模型前拒绝代码不能遵循的环境/时序配置，而不是静默忽略。"""
    if config.get("version") != "genmo.gmt_frozen_isaac.v1":
        raise ValueError("invalid Stage8 configuration version")
    fixed = {"timing": {"clock_hz": 600, "decision_ticks": 300, "control_ticks": 12,
                        "physics_ticks": 3, "warmup_ticks": 600, "min_prefix_frames": 12,
                        "lookahead_control_steps": 10},
             "environment": {"terrain": "plane", "ground_height_m": 0.0,
                             "initial_foot_clearance_m": .001,
                             "domain_randomization": False, "observation_noise": False,
                             "external_perturbations": False, "reset_randomization": False,
                             "actuator_delay_steps": 0, "curriculum": False}}
    for section, fields in fixed.items():
        for key, expected in fields.items():
            if config[section].get(key) != expected:
                raise ValueError(f"Stage8 requires {section}.{key}={expected!r}")
    termination = config["termination"]
    orientation_mode = termination.get("orientation_mode", "global_quaternion")
    if orientation_mode == "global_quaternion":
        thresholds = {"root_height_error_m": .4, "global_orientation_error_rad": 1.2,
                      "end_effector_relative_height_error_m": .3}
        if any(key in termination for key in ("non_yaw_orientation_error_rad", "yaw_error_rad")):
            raise ValueError("Global quaternion mode cannot include separated yaw thresholds")
    elif orientation_mode == "separated_yaw":
        thresholds = {"root_height_error_m": .2, "non_yaw_orientation_error_rad": .6,
                      "yaw_error_rad": 1.5, "end_effector_relative_height_error_m": .15}
        if "global_orientation_error_rad" in termination:
            raise ValueError("Separated yaw mode cannot also specify a global orientation threshold")
    else:
        raise ValueError("Unknown Stage8 orientation_mode")
    for key, expected in thresholds.items():
        if config["termination"].get(key) != expected:
            raise ValueError(f"Stage8 {orientation_mode} requires termination.{key}={expected}")
    vector = config['runtime'].get('backend') == 'gpu_vectorized.v1'
    if vector:
        if config.get('stage10', {}).get('version') != 'genmo.closedloop.stage10.v2':
            raise ValueError('GPU vector backend requires explicit Stage10 v2')
        n = config['runtime'].get('num_envs')
        if type(n) is not int or n < 1 or config['runtime'].get('physics_device') != 'cuda:0':
            raise ValueError('GPU worker requires positive num_envs and its visible cuda:0 device')
    if (not vector and config["runtime"].get("num_envs") != 1) or config["model"].get("history_steps") != 50:
        raise ValueError("Stage8 checkpoint/runtime requires B=1 and H=50")
    if not vector and config["runtime"].get("physics_device") != "cpu":
        raise ValueError("This Stage8 configuration requires the verified CPU PhysX backend")
    for section, key in (("runtime", "torch_threads"), ("runtime", "rpc_timeout_s"),
                         ("runtime", "worker_start_timeout_s"), ("timing", "latency_guard_s"),
                         ("evaluation", "seconds")):
        value = config[section][key]
        if isinstance(value, bool) or not math.isfinite(float(value)) or float(value) <= 0:
            raise ValueError(f"{section}.{key} must be finite and positive")
    if not isinstance(config["runtime"]["torch_threads"], int):
        raise ValueError("runtime.torch_threads must be an integer")
    steps = config["model"]["ddim_steps"]
    if isinstance(steps, bool) or not isinstance(steps, int) or not 2 <= steps <= 1000:
        raise ValueError("model.ddim_steps must be an integer in [2,1000]")
    if not math.isfinite(float(config["model"]["guidance_scale"])):
        raise ValueError("model.guidance_scale must be finite")
    for key, lower, upper in (("calibration_warmup", 0, 100), ("calibration_samples", 1, 1000)):
        value = config["timing"][key]
        if isinstance(value, bool) or not isinstance(value, int) or not lower <= value <= upper:
            raise ValueError(f"timing.{key} must be an integer in [{lower},{upper}]")
    evaluation = config["evaluation"]
    if evaluation.get("split", "val") not in ("train", "val"):
        raise ValueError("Stage8 evaluation split must be explicitly train or val")
    if not isinstance(evaluation.get("require_audio", True), bool):
        raise ValueError("evaluation.require_audio must be boolean")
    selection_seed = evaluation.get("selection_seed", 42)
    if isinstance(selection_seed, bool) or not isinstance(selection_seed, int):
        raise ValueError("evaluation.selection_seed must be an integer")
    if float(evaluation["seconds"]) > 30:
        raise ValueError("Stage8 evaluation duration is bounded at 30 seconds per episode")
    for key, allowed in (("datasets", FULL_DATASETS), ("modes", {"paused", "latency"})):
        values = evaluation[key]
        if not values or len(set(values)) != len(values) or not set(values) <= allowed:
            raise ValueError(f"evaluation.{key} contains missing, duplicate, or unknown entries")
    seeds = evaluation["seeds"]
    if not seeds or len(set(seeds)) != len(seeds) or any(isinstance(v, bool) or not isinstance(v, int) for v in seeds):
        raise ValueError("evaluation.seeds must be a nonempty unique integer list")
    groups = evaluation["groups_per_dataset"]
    if isinstance(groups, dict):
        if not set(evaluation["datasets"]) <= set(groups) or not set(groups) <= FULL_DATASETS:
            raise ValueError("Group counts must cover each selected dataset")
        counts = list(groups.values())
    else:
        counts = [groups]
    if any(isinstance(n, bool) or not isinstance(n, int) or not 1 <= n <= 100 for n in counts):
        raise ValueError("Each dataset group count must be an integer in [1,100]")


def preflight(config, selected, *, check_gpu=True):
    validate_config(config)
    paths = config["paths"]
    required = ("checkpoint", "stats", "kinematics", "gmt_policy", "compat_profile", "isaac_contract", "genmo_python", "isaac_python")
    absent = [paths[k] for k in required if not Path(paths[k]).is_file()]
    files = check_music_files(paths["data_root"], selected,
                              require_audio=config["evaluation"].get("require_audio", True))
    repo_state = {}
    for key in ("genmo_repo", "gmt_repo"):
        root = Path(paths[key]).resolve()
        repo_state[key] = {"path": str(root), "branch": _command(["git", "branch", "--show-current"], root),
                           "head": _command(["git", "rev-parse", "HEAD"], root),
                           "status": _command(["git", "status", "--short"], root),
                           "tracked_diff_sha256": hashlib.sha256(
                               subprocess.check_output(["git", "diff", "HEAD"], cwd=root)).hexdigest()}
    host = {"hostname": socket.gethostname(), "system": platform.system(),
            "machine": platform.machine(), "processor": platform.processor()}
    gpu = {"requested_device": config["runtime"]["genmo_device"],
           "sharing_authorized": bool(config["runtime"].get("allow_shared_gpu", False))}
    if check_gpu and str(gpu["requested_device"]).startswith("cuda"):
        rows = _command(["nvidia-smi", "--query-gpu=index,uuid,name,memory.used,memory.total,utilization.gpu,driver_version", "--format=csv,noheader,nounits"])
        processes = _command(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,process_name,used_memory", "--format=csv,noheader,nounits"])
        gpu.update(devices=rows, existing_compute_processes=processes)
        index = int(str(gpu["requested_device"]).partition(":")[2] or 0)
        parsed = [[field.strip() for field in row] for row in csv.reader(rows.splitlines())]
        visible = os.environ.get("CUDA_VISIBLE_DEVICES")
        physical = str(index)
        if visible is not None:
            allowed = [entry.strip() for entry in visible.split(",") if entry.strip()]
            if index >= len(allowed):
                raise RuntimeError("Requested GPU is not exposed by CUDA_VISIBLE_DEVICES")
            physical = allowed[index]
        matching = [row for row in parsed if row[0] == physical or row[1].startswith(physical)]
        if len(matching) != 1:
            raise RuntimeError("Cannot uniquely identify requested physical GPU")
        row = matching[0]
        gpu["identity"] = {"index": int(row[0]), "uuid": row[1], "model": row[2], "driver_version": row[6]}
        gpu["free_memory_mib"] = int(row[4]) - int(row[3])
        active = [r for r in csv.reader(processes.splitlines()) if r and r[0].strip() == row[1]]
        if active and not gpu["sharing_authorized"]:
            raise RuntimeError("GPU has existing processes; sharing was not authorized")
        if gpu["free_memory_mib"] < 8192:
            raise RuntimeError("Less than 8 GiB free GPU memory; refusing to contend with existing jobs")
    elif str(gpu["requested_device"]) == "cpu":
        gpu["identity"] = {"model": "cpu", "machine": host["machine"], "processor": host["processor"]}
    asset_hashes = {key: sha256_file(paths[key]) for key in ASSET_KEYS if Path(paths[key]).is_file()}
    report = {"version": config["version"], "python": sys.executable, "repositories": repo_state,
              "missing_required_files": absent, "music": files, "gpu": gpu,
              "host": host, "asset_sha256": asset_hashes, "selected": selected}
    report["ready"] = not absent and not files["missing"] and not files["invalid_sha256"]
    return report


def calibration_identity(config, check):
    """只绑定计算相关身份；录像路径、GUI及评估子集不使同机校准失效。"""
    if set(check["asset_sha256"]) != set(ASSET_KEYS) or "identity" not in check["gpu"]:
        raise ValueError("Calibration identity requires verified assets and actual compute device")
    return {"version": "genmo.gmt_frozen_isaac.calibration_identity.v1",
            "asset_sha256": check["asset_sha256"], "host": check["host"],
            "accelerator": check["gpu"]["identity"], "model": config["model"],
            "runtime": {key: config["runtime"][key] for key in
                        ("genmo_device", "physics_device", "torch_threads", "num_envs")},
            "interpreters": {key: {"invocation": str(Path(config["paths"][key]).expanduser().absolute()),
                                   "binary": str(Path(config["paths"][key]).expanduser().resolve())}
                             for key in ("genmo_python", "isaac_python")},
            "timing": {key: config["timing"][key] for key in
                       ("clock_hz", "control_ticks", "min_prefix_frames", "lookahead_control_steps", "latency_guard_s")}}


def validate_calibration(calibration, identity):
    """复用前严格比较原身份，并核验数量/P95/预算，绝不重写旧证据冒充匹配。"""
    if not isinstance(calibration, dict):
        raise ValueError("Calibration must be a JSON object")
    if calibration.get("identity") != identity:
        old = calibration.get("identity", {})
        old = old if isinstance(old, dict) else {}
        fields = sorted(key for key in set(identity) | set(old) if identity.get(key) != old.get(key))
        raise ValueError(f"Calibration identity mismatch: {fields}; measure again on this runtime")
    samples, warmup = calibration.get("measured_requests"), calibration.get("warmup_requests")
    if isinstance(samples, bool) or not isinstance(samples, int) or not 1 <= samples <= 1000:
        raise ValueError("Calibration measured_requests is invalid")
    if isinstance(warmup, bool) or not isinstance(warmup, int) or not 0 <= warmup <= 100:
        raise ValueError("Calibration warmup_requests is invalid")
    durations = np.asarray(calibration.get("end_to_end_seconds", []), dtype=float)
    if durations.shape != (samples,) or not np.isfinite(durations).all() or np.any(durations <= 0):
        raise ValueError("Calibration must contain all positive finite measured durations")
    p95 = float(np.percentile(durations, 95))
    guard = identity["timing"]["latency_guard_s"]
    for key, expected in (("p95_seconds", p95), ("guard_seconds", guard), ("latency_budget_seconds", p95 + guard)):
        actual = float(calibration.get(key, float("nan")))
        if not math.isfinite(actual) or not math.isclose(actual, expected, rel_tol=1e-9, abs_tol=1e-9):
            raise ValueError(f"Calibration {key} does not match measured durations/guard")
    if calibration.get("full_calibration") is not (warmup == 10 and samples == 100):
        raise ValueError("Calibration full_calibration flag contradicts request counts")
    return dict(calibration)


def validation_scope(config, selected, episode_count, calibration, *, video=False, preflight_only=False):
    """只有固定四库/音乐组/种子/模式/时长全部执行，才标记完整矩阵。"""
    per_dataset = Counter(sample["dataset"] for sample in selected)
    full_selection = (set(per_dataset) == FULL_DATASETS and all(n == 2 for n in per_dataset.values())
                      and len({(s["dataset"], s["group_id"]) for s in selected}) == 8)
    evaluation = config["evaluation"]
    full_matrix = (not preflight_only and not video and full_selection
                   and evaluation.get("split", "val") == "val"
                   and config["termination"].get("orientation_mode", "global_quaternion") == "global_quaternion"
                   and set(evaluation["seeds"]) == {42, 43, 44}
                   and set(evaluation["modes"]) == {"paused", "latency"}
                   and float(evaluation["seconds"]) == 30 and episode_count == 48)
    full_calibration = bool(calibration and calibration.get("full_calibration"))
    expected = len(selected) * len(evaluation["seeds"]) * len(evaluation["modes"])
    return {"episode_count": episode_count, "requested_episode_count": expected,
            "dataset_split": evaluation.get("split", "val"),
            "termination": config["termination"],
            "requested_matrix_completed": not preflight_only and episode_count == expected,
            "selected_music_groups": len(selected),
            "unique_audio_sha256_count": len({s.get("row", {}).get("source_audio_sha256") for s in selected}
                                               - {None}), "full_matrix": full_matrix,
            "full_calibration": full_calibration,
            "small_scale_validation": not (full_matrix and full_calibration),
            "video_enabled": bool(video), "preflight_only": bool(preflight_only),
            "synthetic_validation": False}


def acceptance_flags(summary, scope, shutdown, *, code, error):
    """进程退出成功与动力学完成分开；缺少证据一律不自动视为通过。"""
    episodes = [episode for episode in summary.get("episodes", []) if episode.get("mode") != "calibration"]
    traces_complete = bool(episodes) and all(episode.get("trace_integrity", {}).get("complete") is True for episode in episodes)
    no_system_errors = (code == 0 and error is None and not scope["preflight_only"] and traces_complete
                        and set(shutdown) == {"actor", "gmt"}
                        and all(not entry.get("close_error") and entry.get("process_exit_code") == 0
                                for entry in shutdown.values()))
    actor_checks = shutdown.get("actor", {}).get("frozen_checks", {})
    frozen = (all(actor_checks.get(key) is True for key in
                  ("eval_and_no_grad", "parameters_and_buffers_unchanged", "asset_files_unchanged"))
              and shutdown.get("gmt", {}).get("runtime_parameters_unchanged") is True
              and shutdown.get("gmt", {}).get("policy_unchanged") is True)
    protected = bool(episodes) and all(episode.get("protected_reference", {}).get("modification_count") == 0 for episode in episodes)
    mine_success = any(
        episode.get("sample", {}).get("dataset", episode.get("dataset")) == "Mine"
        and episode.get("failed") is False and not episode.get("startup_failure", True)
        and episode.get("music_duration_seconds", 0) >= 30.0
        and episode.get("replan_count", 0) >= 2
        and not episode.get("terminal_snapshot", {}).get("terminated", True)
        and episode.get("trace_integrity", {}).get("complete") is True
        for episode in episodes)
    flags = {"no_system_errors": no_system_errors, "frozen_models_and_environment": frozen,
             "protected_reference_unchanged": protected,
             "continuous_mine_30s_with_multiple_replans": mine_success,
             "fixed_48_episode_matrix_completed": scope["full_matrix"] and len(episodes) == 48,
             "full_10_plus_100_calibration": scope["full_calibration"]}
    flags["baseline_ready"] = all(flags.values())
    return flags


def fatal_worker_start_error(path):
    """仅识别Kit已宣告所有GPU创建失败；普通无窗口/GLFW警告不视为故障。"""
    with Path(path).open('rb') as stream:
        stream.seek(0,2)
        stream.seek(max(0,stream.tell()-16384))
        tail=stream.read().decode('utf-8',errors='replace')
    marker='[omni.gpu_foundation_factory.plugin] Failed to create any GPU devices'
    return next((line for line in tail.splitlines() if marker in line),None)


class Workers:
    def __init__(self, config, output):
        self.config, self.output = config, Path(output)
        self.entries = []
        self.temp = tempfile.TemporaryDirectory(prefix="stage8_rpc_")
        self.shutdown = {}

    def start(self, name, command, cwd, socket_path, *, environment=None, strip_distributed=False):
        log = (self.output / f"{name}_worker.log").open("w")
        env = os.environ.copy()
        if strip_distributed:
            for key in tuple(env):
                if key in {'RANK', 'LOCAL_RANK', 'WORLD_SIZE', 'LOCAL_WORLD_SIZE', 'GROUP_RANK',
                           'ROLE_RANK', 'ROLE_WORLD_SIZE', 'MASTER_ADDR', 'MASTER_PORT'} or key.startswith('TORCHELASTIC_'):
                    env.pop(key, None)
        env.update(environment or {})
        env.update(PYTHONDONTWRITEBYTECODE="1", OMP_NUM_THREADS=str(self.config["runtime"]["torch_threads"]))
        env["PYTHONPATH"] = str(ROOT) + os.pathsep + env.get("PYTHONPATH", "")
        parent_guard = None
        if strip_distributed and sys.platform == 'linux':
            # torchrun 异常终止 rank 时，其独立会话中的 GMT 也必须退出，避免孤儿占卡。
            import ctypes
            libc, parent = ctypes.CDLL(None), os.getpid()
            def parent_guard():
                # Isaac 能捕获 SIGTERM 并长时间停留在清理阶段；父 rank 已死时无法
                # 再完成任何 ACK/恢复，必须可靠终止直属 worker，避免孤儿继续占卡。
                # 正常退出仍通过 close RPC；这里只处理父进程突然死亡。
                if libc.prctl(1, signal.SIGKILL) != 0:
                    os._exit(126)
                if os.getppid() != parent:
                    os._exit(125)
        try:
            proc = subprocess.Popen(command, cwd=cwd, env=env, stdout=log, stderr=subprocess.STDOUT,
                                    start_new_session=True, preexec_fn=parent_guard)
        except BaseException:
            log.close()
            raise
        entry = {"name": name, "proc": proc, "log": log, "client": None}
        self.entries.append(entry)
        deadline = time.monotonic() + float(self.config["runtime"]["worker_start_timeout_s"])
        last_update = time.monotonic()
        last_health_check = last_update-1.
        while time.monotonic() < deadline:
            if proc.poll() is not None:
                raise RuntimeError(f"{name} worker exited {proc.returncode}; see {name}_worker.log")
            if time.monotonic()-last_health_check>=1.:
                last_health_check=time.monotonic()
                fatal=fatal_worker_start_error(log.name)
                if fatal:raise RuntimeError(f'{name} worker startup failed before RPC: {fatal}')
            if Path(socket_path).exists():
                try:
                    client = RpcClient(socket_path, timeout_s=float(self.config["runtime"]["rpc_timeout_s"]))
                    entry["client"] = client
                    hello = client.call("hello")
                    write_json(self.output / f"{name}_identity.json", hello)
                    print(f"[WORKER] {name} ready pid={proc.pid}", flush=True)
                    return client
                except (ConnectionRefusedError, FileNotFoundError):
                    pass
            if time.monotonic() - last_update >= 20:
                print(f"[WORKER] waiting for {name} pid={proc.pid}", flush=True)
                last_update = time.monotonic()
            time.sleep(.2)
        raise TimeoutError(f"{name} worker startup timeout")

    def launch(self, config_path):
        paths = self.config["paths"]
        actor_socket = str(Path(self.temp.name) / "actor.sock")
        gmt_socket = str(Path(self.temp.name) / "gmt.sock")
        actor = self.start("actor", [paths["genmo_python"], "-B", "-m", "gem.closedloop.frozen_actor",
            "--config", str(config_path), "--socket", actor_socket], paths["genmo_repo"], actor_socket)
        command = [paths["isaac_python"], "-B", str(Path(paths["gmt_repo"]) / "scripts/rsl_rl/serve_frozen_gmt.py"),
                   "--config", str(config_path), "--socket", gmt_socket]
        if self.config["runtime"].get("headless", True):
            command.append("--headless")
        backend = self.start("gmt", command, paths["gmt_repo"], gmt_socket)
        return actor, backend

    def close(self):
        for entry in reversed(self.entries):
            name, proc, client = entry["name"], entry["proc"], entry["client"]
            try:
                if client is not None and proc.poll() is None:
                    client.connection.settimeout(min(float(self.config["runtime"]["rpc_timeout_s"]), 30.0))
                    self.shutdown[name] = client.call("close")
            except Exception as exc:
                self.shutdown[name] = {"close_error": str(exc)}
            finally:
                if client is not None:
                    client.close()
                try:
                    proc.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    self.shutdown.setdefault(name, {})["forced_shutdown"] = True
                    # 只操作 start_new_session 创建并仍存活的本轮进程组。
                    try:
                        os.killpg(proc.pid, signal.SIGTERM)
                    except ProcessLookupError:
                        pass
                    try:
                        proc.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        try:
                            os.killpg(proc.pid, signal.SIGKILL)
                        except ProcessLookupError:
                            pass
                        proc.wait(timeout=10)
                entry["log"].close()
                self.shutdown.setdefault(name, {})["process_exit_code"] = proc.returncode
        write_json(self.output / "worker_shutdown.json", self.shutdown)
        self.temp.cleanup()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/closedloop/stage8_frozen_isaac.yaml")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--datasets", nargs="+")
    parser.add_argument("--max-episodes", type=int)
    parser.add_argument("--only-sample", action="append", metavar="DATASET/SAMPLE_ID",
                        help="精确补测完整确定性清单中的样本；可重复指定，不重新选样")
    parser.add_argument("--seconds", type=float)
    parser.add_argument("--modes", nargs="+", choices=("paused", "latency"))
    parser.add_argument("--calibration-warmup", type=int)
    parser.add_argument("--calibration-samples", type=int)
    parser.add_argument("--calibration", type=Path, help="显式复用已测校准JSON；须同模型/机器/配置")
    parser.add_argument("--video", action="store_true", help="逐episode真实Isaac配乐视频；不用于正常计时矩阵")
    args = parser.parse_args(argv)
    config = yaml.safe_load(args.config.read_text())
    if args.max_episodes is not None and args.max_episodes <= 0:
        parser.error("max-episodes must be positive")
    if args.seconds is not None and (not math.isfinite(args.seconds) or args.seconds <= 0):
        parser.error("seconds must be finite and positive")
    if args.calibration and (args.calibration_warmup is not None or args.calibration_samples is not None):
        parser.error("reused calibration cannot be combined with calibration count overrides")
    if args.seconds is not None:
        config["evaluation"]["seconds"] = args.seconds
    if args.modes:
        config["evaluation"]["modes"] = args.modes
    if args.datasets:
        if not set(args.datasets) <= set(config["evaluation"]["datasets"]):
            parser.error("unknown dataset")
        config["evaluation"]["datasets"] = args.datasets
    if args.calibration_warmup is not None:
        config["timing"]["calibration_warmup"] = args.calibration_warmup
    if args.calibration_samples is not None:
        config["timing"]["calibration_samples"] = args.calibration_samples
    try:
        validate_config(config)
    except (KeyError, TypeError, ValueError) as exc:
        parser.error(str(exc))
    if config["evaluation"].get("split", "val") == "train":
        selected = select_train_music(config["paths"]["data_root"], config["evaluation"]["datasets"],
                                      config["evaluation"]["groups_per_dataset"],
                                      seed=config["evaluation"].get("selection_seed", 42))
    else:
        selected = select_val_music(config["paths"]["data_root"], config["evaluation"]["datasets"],
                                    config["evaluation"]["groups_per_dataset"])
    try:
        selected, selection_scope = select_evaluation_subset(selected, args.only_sample)
    except ValueError as exc:
        parser.error(str(exc))
    config["evaluation"]["selection_scope"] = selection_scope
    output = (args.output_dir or Path(config["output_root"]) / datetime.now().strftime("%Y%m%d_%H%M%S_%f")).resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite experiment: {output}")
    output.mkdir(parents=True)
    if args.video:
        config["evaluation"]["require_audio"] = True
        config["runtime"]["video_path"] = str(output / "isaac_video.mp4")
    config_path = output / "resolved_config.yaml"
    config_path.write_text(yaml.safe_dump(config, allow_unicode=True, sort_keys=False))
    workers, recorder, code, error = None, None, 0, None
    summary, calibration, episodes = {}, None, []
    source_report = None
    try:
        check = preflight(config, selected)
        write_json(output / "preflight.json", check)
        if not check["ready"]:
            raise RuntimeError("Missing or SHA-mismatched inputs; see preflight.json")
        identity = calibration_identity(config, check)
        if args.calibration:
            # 在启动两个昂贵 worker 之前检查，保留原 identity 而非覆盖。
            calibration = validate_calibration(json.loads(args.calibration.read_text()), identity)
            calibration["reused_from"] = str(args.calibration.resolve())
        if args.preflight:
            print(json.dumps({"preflight": "passed", "output": str(output)}, ensure_ascii=False), flush=True)
            return 0
        source_report = collect_source_provenance(config["paths"], repository_state=check["repositories"])
        source_report["capture_phase"] = "before_worker_start"
        write_json(output / "source_provenance.json", source_report)
        write_json(output / "compute_context.json", collect_compute_context(check["gpu"]))
        from gem.closedloop.baseline_metrics import BaselineRecorder
        from gem.closedloop.coordinator import ClosedLoopCoordinator
        from gem.closedloop.online_conditions import OnlineConditionBuilder
        from gem.robots.bumi.feature_codec import BumiMotionFeatureCodec
        from gem.robots.bumi.kinematics import BumiKinematics
        kin = BumiKinematics(config["paths"]["kinematics"])
        builder = OnlineConditionBuilder(BumiMotionFeatureCodec(kin), history_steps=config["model"]["history_steps"])
        recorder = BaselineRecorder(output, data_root=config["paths"]["data_root"])
        workers = Workers(config, output)
        actor, backend = workers.launch(config_path)
        loop = ClosedLoopCoordinator(config, actor, backend, builder, recorder)
        calibration_sample = next((s for s in selected if s["dataset"] == "Mine"), selected[0])
        if calibration is None:
            calibration = loop.calibrate(calibration_sample, load_music_features(config["paths"]["data_root"], calibration_sample),
                                         warmup=config["timing"]["calibration_warmup"],
                                         samples=config["timing"]["calibration_samples"])
            calibration["identity"] = identity
            calibration = validate_calibration(calibration, identity)
        write_json(output / "calibration.json", calibration)
        for mode in config["evaluation"]["modes"]:
            for sample in selected:
                for seed in config["evaluation"]["seeds"]:
                    if args.max_episodes is not None and len(episodes) >= args.max_episodes:
                        break
                    music = load_music_features(config["paths"]["data_root"], sample)
                    result = loop.run_episode(sample, music, seed=seed, mode=mode,
                                              latency_budget_s=calibration["latency_budget_seconds"])
                    episodes.append(result)
                    write_json(output / "progress.json", {
                        "completed_episodes": len(episodes),
                        "requested_episodes": len(selected) * len(config["evaluation"]["modes"]) * len(config["evaluation"]["seeds"]),
                        "last_dataset": sample["dataset"], "last_sample_id": sample["row"]["sample_id"],
                        "last_reason": result.get("reason"),
                        "failures": sum(bool(item.get("failed")) for item in episodes),
                        "music_exposure_seconds": sum(item.get("music_duration_seconds", 0.) for item in episodes),
                    })
                    print(f"[EPISODE] {len(episodes)} {mode} {sample['dataset']} {sample['row']['sample_id']} seed={seed}: {result.get('reason')}", flush=True)
        summary = recorder.summarize()
    except BaseException as exc:
        code, error = 1, {"type": type(exc).__name__, "message": str(exc), "traceback": traceback.format_exc()}
        # 大批量评估的中途系统异常不能抹去已经完成的独立episode证据。
        if not summary and episodes:
            summary = {"episodes": episodes, "partial_run": True}
        write_json(output / "failure.json", error)
        traceback.print_exc()
    finally:
        if workers is not None:
            try:
                workers.close()
            except Exception as exc:
                code = 1
                shutdown_error = {"type": type(exc).__name__, "message": str(exc), "traceback": traceback.format_exc()}
                write_json(output / "shutdown_failure.json", shutdown_error)
                error = error or shutdown_error
            if any(value.get("close_error") or value.get("forced_shutdown") or value.get("process_exit_code") != 0 for value in workers.shutdown.values()):
                code = 1
        if recorder is not None and hasattr(recorder, "close"):
            recorder.close()
        if source_report is not None:
            source_verification = verify_source_provenance(source_report)
            write_json(output / "source_verification.json", source_verification)
            if not source_verification["unchanged"]:
                code = 1
            summary["source_verification"] = source_verification
        if args.video and code == 0 and episodes:
            try:
                from gem.closedloop.baseline_video import mux_episode_videos
                specs = []
                for episode in episodes:
                    destination = (output / "isaac_music.mp4" if len(episodes) == 1 else
                                   (output / episode["artifacts"]["trace"]).parent / "isaac_music.mp4")
                    specs.append({"audio_path": music_path(config["paths"]["data_root"], episode["sample"], "audio_path"),
                                  "trace_path": output / episode["artifacts"]["trace"], "output_path": destination,
                                  "expected_audio_sha256": episode["sample"]["row"]["source_audio_sha256"]})
                summary["videos"] = mux_episode_videos(config["runtime"]["video_path"], specs)
                if len(episodes) == 1:
                    summary["video"] = summary["videos"][0]
            except Exception as exc:
                code = 1
                video_error = {"type": type(exc).__name__, "message": str(exc), "traceback": traceback.format_exc()}
                write_json(output / "video_failure.json", video_error)
                error = error or video_error
        scope = validation_scope(config, selected, len(episodes), calibration, video=args.video, preflight_only=args.preflight)
        flags = acceptance_flags(summary, scope, workers.shutdown if workers is not None else {}, code=code, error=error)
        summary.update(validation_scope=scope, acceptance=flags, exit_code=code,
                       selection=selected, selection_scope=selection_scope,
                       evaluation=config["evaluation"], termination=config["termination"])
        write_json(output / "run_summary.json", summary)
        if summary.get("episodes"):
            from tools.eval.audit_closedloop_baseline import audit_experiment
            try:
                audit = audit_experiment(output)
            except Exception as exc:
                audit = {"status": "failed", "error": str(exc), "traceback": traceback.format_exc()}
            write_json(output / "audit.json", audit)
            flags["protocol_and_history_audit_passed"] = audit["status"] == "passed"
            flags["render_pose_audit_passed"] = (
                audit.get("visual_evidence", {}).get("status") == "passed" if args.video else None)
            flags["baseline_ready"] = flags["baseline_ready"] and flags["protocol_and_history_audit_passed"]
            if args.video:
                flags["baseline_ready"] = flags["baseline_ready"] and flags["render_pose_audit_passed"]
            if audit["status"] != "passed" or (args.video and not flags["render_pose_audit_passed"]):
                code = 1
            summary["exit_code"] = code
            write_json(output / "run_summary.json", summary)
        print(json.dumps({"exit_code": code, "output": str(output), "baseline_ready": flags["baseline_ready"],
                          "error": error and error["message"]}, ensure_ascii=False), flush=True)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
