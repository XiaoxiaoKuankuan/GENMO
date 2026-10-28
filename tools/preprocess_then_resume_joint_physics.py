"""受控维护流水线：完整断点恢复，或经明确授权丢弃进度后的从零训练。

本进程始终只用CPU。启动时固定指定torchrun、其8个训练rank和旧单环境collector的
/proc身份（PID、启动时钟、cwd、完整argv）；等待明确global_step的原子checkpoint。
只有CPU加载核验模型/AdamW/scheduler、课程、8rank采样游标/RNG、sidecar快照及全部
浮点张量有限，并持久化checkpoint_verified.json后，才允许发送SIGTERM。信号使用
Linux pidfd，并在发送前再次比较完整进程身份；PID被复用、命令改变或训练异常消失
立即中止，不按名字搜杀，也不强制SIGKILL。旧collector自然退出则只记录、不发信号。

确认8卡空闲后串行运行三规模真实benchmark、8卡完整source预处理和原配置完整恢复。
任一阶段失败都保留checkpoint/日志并停止后续阶段，不盲目重启。恢复输出使用新目录，
原任务、checkpoint和失败证据不覆盖；此进程留在tmux中等待恢复训练。benchmark清理
仅删除本流水线独占目录中已核验身份的worker分片，保留索引、summary及失败日志。
status.json和events.jsonl持续记录阶段/PID/日志/当前步数。脚本必须显式给定授权任务PID，
不会自动选择正在运行的其它训练；本文件的存在本身不代表流水线或训练已经启动。

显式--fresh-from-stopped-run切换到已停任务模式：必须给出用户接受丢弃未保存更新的
停机证据，并重新核验具体旧PID均已退出和八卡空闲；该模式不捕获/停止不存在的任务，
不等待或加载checkpoint。原配置必须random初始化、无pretrained/resume、seed42，
全库预处理完成后在新目录启动200000步随机训练，命令绝不包含--resume。
"""

import argparse
from contextlib import contextmanager
import ctypes
import fcntl
import hashlib
import json
import os
from pathlib import Path
import platform
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.prepare_joint_physics_parallel import atomic_json, parse_gpus, require_idle_gpus

CHECKPOINT_SCHEMA = "genmo.joint_diffusion_physics.v1.checkpoint.v3.formal_prefix_future"


def _linux_x86_64_syscall(number, *arguments):
    """仅使用Server2已核验的Linux x86_64 pidfd ABI；其它架构直接拒绝。"""
    if sys.platform != "linux" or platform.machine() != "x86_64" or ctypes.sizeof(ctypes.c_void_p) != 8:
        raise RuntimeError("缺少Python pidfd接口，syscall回退仅支持已核验Linux x86_64 ABI")
    if number not in (424, 434):
        raise ValueError("仅允许已核验pidfd_send_signal=424/pidfd_open=434")
    libc = ctypes.CDLL(None, use_errno=True)
    syscall = libc.syscall
    syscall.restype = ctypes.c_long
    syscall.argtypes = [ctypes.c_long]
    ctypes.set_errno(0)
    result = syscall(ctypes.c_long(number), *arguments)
    if result == -1:
        error = ctypes.get_errno()
        raise OSError(error, os.strerror(error))
    return int(result)


def safe_pidfd_open(pid):
    """优先Python接口；发行版缺接口时直接调用pidfd_open，不退回裸PID信号。"""
    builtin = getattr(os, "pidfd_open", None)
    if callable(builtin):
        return int(builtin(int(pid), 0))
    return _linux_x86_64_syscall(434, ctypes.c_int(int(pid)), ctypes.c_uint(0))


def safe_pidfd_send_signal(descriptor, signal_number):
    if signal_number not in (0, signal.SIGTERM):
        raise ValueError("受控流水线仅允许signal0预检及SIGTERM")
    builtin = getattr(signal, "pidfd_send_signal", None)
    if callable(builtin):
        builtin(int(descriptor), int(signal_number), None, 0)
        return
    _linux_x86_64_syscall(424, ctypes.c_int(int(descriptor)), ctypes.c_int(int(signal_number)),
                         ctypes.c_void_p(None), ctypes.c_uint(0))


def pidfd_preflight():
    """只对协调器自身发送signal0做权限/ABI检查，不停止自身或任何任务。"""
    descriptor = safe_pidfd_open(os.getpid())
    try:
        safe_pidfd_send_signal(descriptor, 0)
    finally:
        os.close(descriptor)
    return {"own_pid": os.getpid(), "signal_number": 0, "pidfd_closed": True,
            "pidfd_open_backend": "python" if callable(getattr(os, "pidfd_open", None)) else "linux_x86_64_syscall434",
            "pidfd_send_backend": "python" if callable(getattr(signal, "pidfd_send_signal", None)) else "linux_x86_64_syscall424"}


def argument_value(argv, name):
    matches = [argv[i+1] for i, item in enumerate(argv[:-1]) if item == name]
    matches += [item.split("=", 1)[1] for item in argv if item.startswith(name+"=")]
    if len(matches) != 1:
        raise ValueError(f"命令行必须仅含一个{name}")
    return matches[0]


def process_identity(pid, proc_root=Path("/proc")):
    directory = Path(proc_root)/str(pid)
    try:
        stat = (directory/"stat").read_text().rsplit(") ", 1)[1].split()
        if stat[0] == "Z":
            return None
        cwd = str((directory/"cwd").resolve(strict=True))
        argv = [part.decode() for part in (directory/"cmdline").read_bytes().split(b"\0") if part]
        second = (directory/"stat").read_text().rsplit(") ", 1)[1].split()
        if second[19] != stat[19]:
            raise RuntimeError(f"读取期间PID身份变化: {pid}")
        return {"pid": int(pid), "starttime": int(stat[19]), "cwd": cwd, "argv": argv}
    except FileNotFoundError:
        return None


def require_same_process(expected, *, allow_gone=False, reader=None):
    current = (reader or process_identity)(expected["pid"])
    if current is None and allow_gone:
        return False
    if current != expected:
        raise RuntimeError(f"指定进程消失或身份变化，禁止发信号: PID {expected['pid']}")
    return True


def capture_task(pid, repo, output, role):
    value = process_identity(pid)
    if value is None or value["cwd"] != str(repo):
        raise ValueError(f"{role} PID不存在或cwd不匹配")
    if Path(argument_value(value["argv"], "--output")).resolve() != output:
        raise ValueError(f"{role} --output与指定任务不匹配")
    entry = "tools/train_joint_physics.py" if role in ("torchrun", "rank") else "tools/prepare_joint_physics_data.py"
    if entry not in value["argv"]:
        raise ValueError(f"{role}不是预期入口")
    if role == "torchrun" and ("torch.distributed.run" not in value["argv"] or
                               argument_value(value["argv"], "--nproc_per_node") != "8"):
        raise ValueError("必须显式指定8卡torchrun主管PID")
    return value


def capture_workers(torchrun, repo, run):
    path = Path(f"/proc/{torchrun['pid']}/task/{torchrun['pid']}/children")
    pids = [int(item) for item in path.read_text().split()]
    workers = [capture_task(pid, repo, run, "rank") for pid in pids]
    if len(workers) != 8:
        raise ValueError("torchrun必须具有确切8个当前训练子rank")
    ranks = []
    for worker in workers:
        # 只读取明确需要的rank字段，不保存或输出其它环境变量。
        values = (Path(f"/proc/{worker['pid']}/environ").read_bytes().split(b"\0"))
        env = dict(item.split(b"=", 1) for item in values if item.startswith((b"RANK=", b"WORLD_SIZE=")))
        if env.get(b"WORLD_SIZE") != b"8":
            raise ValueError("训练子进程WORLD_SIZE不是8")
        ranks.append(int(env[b"RANK"]))
    if sorted(ranks) != list(range(8)):
        raise ValueError("训练rank编号不完整或重复")
    return workers


def validate_checkpoint_payload(value, step, world_size=8):
    import torch
    required = {"actor", "optimizer", "scheduler", "curriculum", "rank_states", "contract", "counters",
                "config", "sidecar_snapshot", "global_step", "initialization"}
    if value.get("schema") != CHECKPOINT_SCHEMA or not required <= value.keys() or value["global_step"] != step:
        raise ValueError("checkpoint schema/字段/global_step不符合指定完整恢复点")
    if not value["actor"] or not value["optimizer"].get("state") or not value["optimizer"].get("param_groups"):
        raise ValueError("checkpoint模型或AdamW状态为空")
    if not value["scheduler"] or value["curriculum"]["step"] != step or not value["sidecar_snapshot"]:
        raise ValueError("checkpoint缺scheduler/课程/sidecar完整状态")
    if len(value["rank_states"]) != world_size:
        raise ValueError("checkpoint没有完整8rank状态")
    required_rank = {"data", "noise", "window", "rng", "physics_data", "physics_noise", "future_data", "future_noise"}
    for rank, state in enumerate(value["rank_states"]):
        if not required_rank <= state.keys() or any(state[key] is None for key in required_rank):
            raise ValueError(f"rank{rank}缺少完整采样游标/RNG")
        if (state["data"]["rank"], state["data"]["world_size"]) != (rank, world_size):
            raise ValueError("rank数据游标编号不一致")
        if not {"rng", "draws", "identity"} <= state["data"].keys():
            raise ValueError("rank全库监督采样游标不完整")
        for name in ("physics_data", "future_data"):
            if not {"source_rng", "time_rng", "draws", "order", "cursor", "identity"} <= state[name].keys():
                raise ValueError(f"rank {name}独立source/time游标不完整")
        if not {"python", "numpy", "cpu", "cuda_current"} <= state["rng"].keys() or state["rng"]["cuda_current"] is None:
            raise ValueError("rank CPU/CUDA/NumPy/Python RNG不完整")
        for rng in (state["noise"], state["physics_noise"], state["future_noise"], state["rng"]["cpu"], state["rng"]["cuda_current"]):
            if not isinstance(rng, torch.Tensor) or rng.dtype != torch.uint8 or not rng.numel():
                raise ValueError("rank扩散/CPU/CUDA RNG状态不是完整字节tensor")
    tensors, tensor_bytes = 0, 0
    def visit(item, group=""):
        nonlocal tensors, tensor_bytes
        if isinstance(item, torch.Tensor):
            tensors += 1
            tensor_bytes += item.numel()*item.element_size()
            if item.is_floating_point() or item.is_complex():
                if not bool(torch.isfinite(item).all()):
                    raise ValueError(f"checkpoint存在非有限tensor: {group}")
                if group.startswith(("actor.", "optimizer.")) and item.dtype != torch.float32:
                    raise ValueError("GENMO/AdamW主状态不是FP32")
        elif isinstance(item, dict):
            for key, child in item.items():
                visit(child, f"{group}.{key}" if group else str(key))
        elif isinstance(item, (tuple, list)):
            for i, child in enumerate(item):
                visit(child, f"{group}.{i}")
    visit(value)
    return {"global_step": step, "world_size": world_size, "tensor_count": tensors,
            "tensor_bytes": tensor_bytes, "all_floating_tensors_finite": True,
            "assets": value["contract"]["assets"], "training_max_steps": value["config"]["training"]["max_steps"]}


def checkpoint_signature(path):
    stat = Path(path).stat()
    return {"inode": stat.st_ino, "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def verify_checkpoint(path, step, proof_path):
    import torch
    path = Path(path).resolve(strict=True)
    before = checkpoint_signature(path)
    value = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    proof = validate_checkpoint_payload(value, step)
    del value
    sha = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8*1024**2), b""):
            sha.update(block)
    if checkpoint_signature(path) != before:
        raise ValueError("核验期间checkpoint发生变化")
    proof.update(verified=True, path=str(path), schema=CHECKPOINT_SCHEMA, sha256=sha.hexdigest(),
                 file_signature=before, verified_unix_seconds=time.time())
    atomic_json(proof_path, proof, replace=False)
    return json.loads(Path(proof_path).read_text())


def require_checkpoint_proof(proof_path):
    proof = json.loads(Path(proof_path).read_text())
    if not proof.get("verified") or not proof.get("all_floating_tensors_finite") or proof.get("schema") != CHECKPOINT_SCHEMA:
        raise ValueError("尚无完整有限checkpoint证据，禁止停止任务")
    if checkpoint_signature(proof["path"]) != proof["file_signature"]:
        raise ValueError("已核验checkpoint被修改，禁止停止或恢复")
    return proof


def signal_exact_pidfd(identity):
    descriptor = safe_pidfd_open(identity["pid"])
    try:
        require_same_process(identity)
        safe_pidfd_send_signal(descriptor, signal.SIGTERM)
    finally:
        os.close(descriptor)


def terminate_verified(identity, proof_path, *, allow_gone=False):
    require_checkpoint_proof(proof_path)
    if not require_same_process(identity, allow_gone=allow_gone):
        return "already_exited"
    signal_exact_pidfd(identity)
    return "sigterm_sent"


def wait_exited(identities, timeout):
    deadline = time.monotonic()+timeout
    while True:
        live = [value for value in identities if require_same_process(value, allow_gone=True)]
        if not live:
            return
        if time.monotonic() >= deadline:
            raise TimeoutError(f"指定任务未在超时内退出；不强制kill: {[item['pid'] for item in live]}")
        time.sleep(2)


def latest_metric(run):
    path = Path(run)/"metrics.jsonl"
    if not path.exists():
        return {}
    with path.open("rb") as handle:
        handle.seek(0, os.SEEK_END)
        handle.seek(max(0, handle.tell()-131072))
        lines = handle.read().splitlines()
    for line in reversed(lines):
        try:
            return json.loads(line)
        except (ValueError, UnicodeDecodeError):
            continue
    return {}


class Journal:
    def __init__(self, output):
        self.output, self.value = Path(output), {}

    def update(self, stage, **values):
        self.value.update(stage=stage, updated_unix_seconds=time.time(), **values)
        atomic_json(self.output/"status.json", self.value)
        with (self.output/"events.jsonl").open("a") as handle:
            handle.write(json.dumps({"stage": stage, "time": time.time(), **values}, ensure_ascii=False)+"\n")
            handle.flush()
        print(json.dumps({"stage": stage, **values}, ensure_ascii=False), flush=True)


@contextmanager
def pipeline_lock(run):
    with (Path(run)/".preprocess_resume_pipeline.lock").open("a+") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def run_stage(journal, stage, command, env, repo, poll_seconds):
    log = journal.output/f"{stage}.log"
    with log.open("x") as handle:
        child = subprocess.Popen(command, cwd=repo, env=env, stdout=handle, stderr=subprocess.STDOUT,
                                 start_new_session=True)
        journal.update(stage, active_pid=child.pid, active_log=str(log), command=command)
        while child.poll() is None:
            time.sleep(poll_seconds)
            if stage in ("resumed_training", "fresh_training"):
                metric = latest_metric(argument_value(command, "--output"))
                journal.update(stage, current_global_step=metric.get("step"), data_loss=metric.get("data_loss"),
                    physics_effective_updates=metric.get("physics_effective_updates"), active_pid=child.pid,
                    active_log=str(log))
        code = child.wait()
    journal.update(stage+"_exited", returncode=code, active_pid=None)
    if code != 0:
        raise RuntimeError(f"{stage}失败，exitcode={code}；后续阶段未启动")


def validate_preprocessing(summary, index, proof, expected_sources):
    if (summary.get("status") != "completed" or summary.get("benchmark") or
            not summary.get("complete_index_verified") or not summary.get("frozen_unchanged") or
            summary.get("catalog_sources") != expected_sources or
            summary.get("covered_catalog_sources") != expected_sources or
            summary.get("uncovered_catalog_sources") != 0 or summary.get("missing_assigned_sources")):
        raise ValueError("全库预处理未完整覆盖/合同或冻结核验失败，禁止恢复训练")
    if index["assets"] != proof["assets"] or index["split"] != "train":
        raise ValueError("预处理资产/split与完整训练checkpoint不同")


def clean_benchmark_shards(benchmark):
    """只删除本流水线新建benchmark中已列入不可变worker索引的精确分片。"""
    benchmark = Path(benchmark).resolve(strict=True)
    removed = []
    for size in (256, 512, 1024):
        trial = benchmark/f"envs_{size}"
        if not trial.exists():
            continue
        for directory in trial.glob("worker_[0-9][0-9]"):
            index_path = directory/"index.json"
            index = json.loads(index_path.read_text()) if index_path.exists() else {"shards": {}}
            for path in sorted(directory.glob("shard_*.pt")):
                name = path.name
                expected_sha = index["shards"].get(name)
                resolved = path.resolve(strict=True)
                if (path.is_symlink() or resolved.parent != directory.resolve() or not resolved.is_relative_to(benchmark)
                        or Path(name).name != name or not name.startswith("shard_") or not name.endswith(".pt")):
                    raise ValueError("benchmark清理路径超出精确许可范围")
                sha = hashlib.sha256()
                with path.open("rb") as handle:
                    for block in iter(lambda: handle.read(8*1024**2), b""):
                        sha.update(block)
                if expected_sha is not None and sha.hexdigest() != expected_sha:
                    raise ValueError("benchmark分片身份改变，保留现场，不清理")
                removed.append({"path": str(resolved), "sha256": sha.hexdigest(), "bytes": resolved.stat().st_size,
                                "published_in_worker_index": expected_sha is not None})
    # 全部路径/身份先核验，通过后才开始删除，日志和索引保留为验收证据。
    for item in removed:
        Path(item["path"]).unlink()
    atomic_json(benchmark/"temporary_shards_removed.json", {"removed": removed, "summaries_and_logs_preserved": True})
    return removed


def execute(args, journal):
    run, sidecar, repo = args.run, args.sidecar, args.repo
    training = capture_task(args.torchrun_pid, repo, run, "torchrun")
    workers = capture_workers(training, repo, run)
    collector = capture_task(args.collector_pid, repo, sidecar, "collector") if args.collector_pid else None
    atomic_json(journal.output/"authorized_processes.json", {"torchrun": training, "ranks": workers, "collector": collector}, replace=False)
    config_path = run/"config.json"
    cfg = json.loads(config_path.read_text())
    sidecars = cfg["data"]["sidecar"]
    if not isinstance(sidecars, list) or Path(sidecars[-1]).resolve() != sidecar or cfg["training"]["max_steps"] != args.max_steps:
        raise ValueError("原训练配置的第三sidecar/绝对训练步数与请求不同")
    checkpoint = run/f"joint_{args.checkpoint_step:08d}.pt"
    while not checkpoint.exists():
        require_same_process(training)
        for worker in workers:
            require_same_process(worker)
        if collector is not None and not require_same_process(collector, allow_gone=True):
            journal.update("collector_finished_while_waiting", collector_pid=args.collector_pid)
            collector = None
        metric = latest_metric(run)
        remaining = max(0, args.checkpoint_step-int(metric.get("step", 0)))
        journal.update("waiting_checkpoint", current_global_step=metric.get("step"),
            expected_checkpoint_step=args.checkpoint_step, checkpoint=str(checkpoint),
            estimated_wait_seconds=remaining*metric["wall_seconds"] if metric.get("wall_seconds") else None)
        time.sleep(args.poll_seconds)
    require_same_process(training)
    proof_path = journal.output/"checkpoint_verified.json"
    proof = verify_checkpoint(checkpoint, args.checkpoint_step, proof_path)
    if proof["training_max_steps"] != args.max_steps:
        raise ValueError("checkpoint绝对训练预算与指定恢复预算不同，禁止停止")
    journal.update("checkpoint_verified", checkpoint=str(checkpoint), proof=str(proof_path), global_step=args.checkpoint_step)
    for worker in workers:
        require_same_process(worker)
    journal.update("stopping_training", result=terminate_verified(training, proof_path))
    wait_exited([training, *workers], args.exit_timeout)
    journal.update("training_stopped", stopped_pids=[training["pid"], *[w["pid"] for w in workers]])
    if collector is not None:
        journal.update("stopping_collector", result=terminate_verified(collector, proof_path, allow_gone=True))
        wait_exited([collector], args.exit_timeout)
    journal.update("original_tasks_exited")
    deadline = time.monotonic()+args.gpu_idle_timeout
    while True:
        try:
            require_idle_gpus(args.gpus)
            break
        except RuntimeError as error:
            if time.monotonic() >= deadline:
                raise
            journal.update("waiting_idle_gpus", reason=str(error))
            time.sleep(args.poll_seconds)
    cpu_env = dict(os.environ, CUDA_VISIBLE_DEVICES="", PYTHONDONTWRITEBYTECODE="1", PYTHONUNBUFFERED="1")
    # 从本次正式run的实际配置派生，仅改变离线预算/分片；恢复训练仍用原config.json。
    preparation_config = json.loads(json.dumps(cfg))
    preparation_config["prepare"].update(resume=True, max_sources=None, max_sources_per_dataset=None,
        max_control_steps=None, shard_episodes=8, balanced_sources=True)
    preparation_path = journal.output/"preparation_config.json"
    atomic_json(preparation_path, preparation_config, replace=False)
    benchmark_config = json.loads(json.dumps(preparation_config))
    excluded = benchmark_config["prepare"].setdefault("exclude_sidecars", [])
    if str(sidecar) not in excluded:
        excluded.append(str(sidecar))
    benchmark_path = journal.output/"benchmark_config.json"
    atomic_json(benchmark_path, benchmark_config, replace=False)
    benchmark = journal.output/"benchmark"
    try:
        run_stage(journal, "benchmark", [args.python, "tools/benchmark_joint_physics_preparation.py",
            "--config", str(benchmark_path), "--output", str(benchmark), "--gpu", args.gpus[0]], cpu_env, repo, args.poll_seconds)
    except RuntimeError:
        # 子流水线已经退出的失败同样清理临时分片，保留失败日志、索引及删除摘要。
        if benchmark.exists():
            clean_benchmark_shards(benchmark)
        raise
    selection = json.loads((benchmark/"summary.json").read_text())
    size = selection.get("selected_num_envs")
    if selection.get("status") != "selected" or size not in (256, 512, 1024):
        raise ValueError("真实规模benchmark没有合法选择，禁止启动全库预处理")
    journal.update("benchmark_selected", selected_num_envs=size)
    clean_benchmark_shards(benchmark)
    run_stage(journal, "preprocessing", [args.python, "tools/prepare_joint_physics_parallel.py",
        "--config", str(preparation_path), "--output", str(sidecar), "--gpus", ",".join(args.gpus),
        "--num-envs", str(size)], cpu_env, repo, args.poll_seconds)
    summary = json.loads((sidecar/"summary.json").read_text())
    index = json.loads((sidecar/"index.json").read_text())
    proof = require_checkpoint_proof(proof_path)
    validate_preprocessing(summary, index, proof, args.expected_sources)
    journal.update("preprocessing_verified", covered_sources=summary["covered_catalog_sources"],
        legal_h32_windows=summary["all_published_legal_h32"]+summary.get("excluded_legal_h32", 0),
        sidecar=str(sidecar), resume_checkpoint=str(checkpoint))
    if args.resume_output.exists():
        raise FileExistsError("恢复输出已存在，禁止重复启动或覆盖")
    require_idle_gpus(args.gpus)
    train_env = dict(cpu_env, CUDA_VISIBLE_DEVICES=",".join(args.gpus), JOINT_GPUS="8", JOINT_PYTHON=args.python)
    run_stage(journal, "resumed_training", ["bash", "scripts/train_joint_physics.sh", str(config_path),
        "--resume", str(checkpoint), "--max-steps", str(args.max_steps), "--output", str(args.resume_output)],
        train_env, repo, args.poll_seconds)
    final = json.loads((args.resume_output/"summary.json").read_text())
    if final.get("status") == "paused":
        journal.update("resumed_training_paused", resume_output=str(args.resume_output), global_step=final["global_step"])
    elif final.get("global_step") != args.max_steps:
        raise ValueError("恢复训练退出但未达到指定总步数，保留现场不自动重启")
    else:
        journal.update("completed", resume_output=str(args.resume_output), max_steps=args.max_steps)


def require_random_training_config(cfg, args):
    initialization = cfg["initialization"]
    if (initialization.get("mode") != "random" or initialization.get("pretrained_genmo") is not None
            or initialization.get("resume_checkpoint") is not None or cfg["training"]["seed"] != 42
            or cfg["training"]["max_steps"] != 200000 or args.max_steps != 200000):
        raise ValueError("丢弃模式必须随机初始化、无旧GENMO权重/恢复状态、seed42及200000步")
    sidecars = cfg["data"]["sidecar"]
    if not isinstance(sidecars, list) or Path(sidecars[-1]).resolve() != args.sidecar:
        raise ValueError("原配置最后一个sidecar不是本次已停采集目录")


def verify_stopped_evidence(args):
    evidence = json.loads(args.stop_evidence.read_text())
    if (evidence.get("authorization") != "discard_uncheckpointed_progress_and_restart_random"
            or evidence.get("all_exited") is not True or evidence.get("gpu_idle") is not True
            or Path(evidence["run"]).resolve() != args.run
            or [str(value) for value in evidence.get("gpus", [])] != args.gpus):
        raise ValueError("缺少明确丢弃授权或完整停机/GPU证据")
    identities = evidence.get("stopped_processes", [])
    if len(identities) < 9 or len({item["pid"] for item in identities}) != len(identities):
        raise ValueError("停机证据需包含独立torchrun和8个rank完整身份")
    training, supervisors = [], []
    for identity in identities:
        if not {"pid", "starttime", "cwd", "argv"} <= identity.keys() or int(identity["starttime"]) <= 0:
            raise ValueError("停机证据缺具体PID/starttime/cwd/argv")
        if process_identity(identity["pid"]) is not None:
            raise ValueError(f"旧PID仍存在或已被复用，禁止继续: {identity['pid']}")
        if "tools/train_joint_physics.py" in identity["argv"]:
            if (Path(identity["cwd"]).resolve() != args.repo
                    or Path(argument_value(identity["argv"], "--output")).resolve() != args.run):
                raise ValueError("停机证据中的训练进程不是指定原run")
            training.append(identity)
            if "torch.distributed.run" in identity["argv"]:
                supervisors.append(identity)
    if len(training) != 9 or len(supervisors) != 1 or argument_value(supervisors[0]["argv"], "--nproc_per_node") != "8":
        raise ValueError("停机证据必须明确包含1个八卡torchrun及其8个训练rank")
    require_idle_gpus(args.gpus)
    return evidence


def execute_fresh(args, journal):
    """仅用于用户明确接受丢弃且旧任务已停；整个路径不调用checkpoint/信号接口。"""
    evidence = verify_stopped_evidence(args)
    config_path = args.run/"config.json"
    config_bytes = config_path.read_bytes()
    cfg = json.loads(config_bytes)
    require_random_training_config(cfg, args)
    atomic_json(journal.output/"stop_evidence_verified.json", {"evidence": evidence,
        "evidence_path": str(args.stop_evidence), "verified_unix_seconds": time.time(),
        "all_exited_rechecked": True, "gpu_idle_rechecked": True}, replace=False)
    journal.update("stopped_tasks_verified_for_fresh_training", fresh_training=True,
        discarded_uncheckpointed_progress=True, initial_global_step=0, resume_checkpoint=None,
        pretrained_genmo=None, seed=42, training_output=str(args.resume_output))
    # 保存原真实资产合同；后续协调器重新检查实际文件和分片，不能因重训放松资产检查。
    baseline = json.loads((args.sidecar/"index.json").read_text())
    reference = {"assets": baseline["assets"]}
    cpu_env = dict(os.environ, CUDA_VISIBLE_DEVICES="", PYTHONDONTWRITEBYTECODE="1", PYTHONUNBUFFERED="1")
    preparation_config = json.loads(json.dumps(cfg))
    preparation_config["prepare"].update(resume=True, max_sources=None, max_sources_per_dataset=None,
        max_control_steps=None, shard_episodes=8, balanced_sources=True)
    preparation_path = journal.output/"preparation_config.json"
    atomic_json(preparation_path, preparation_config, replace=False)
    benchmark_config = json.loads(json.dumps(preparation_config))
    excluded = benchmark_config["prepare"].setdefault("exclude_sidecars", [])
    if str(args.sidecar) not in excluded:
        excluded.append(str(args.sidecar))
    benchmark_path = journal.output/"benchmark_config.json"
    atomic_json(benchmark_path, benchmark_config, replace=False)
    benchmark = journal.output/"benchmark"
    try:
        run_stage(journal, "benchmark", [args.python, "tools/benchmark_joint_physics_preparation.py",
            "--config", str(benchmark_path), "--output", str(benchmark), "--gpu", args.gpus[0]],
            cpu_env, args.repo, args.poll_seconds)
    except RuntimeError:
        if benchmark.exists():
            clean_benchmark_shards(benchmark)
        raise
    selection = json.loads((benchmark/"summary.json").read_text())
    size = selection.get("selected_num_envs")
    if selection.get("status") != "selected" or size not in (256, 512, 1024):
        raise ValueError("真实规模benchmark没有合法选择，禁止启动全库或随机训练")
    journal.update("benchmark_selected", selected_num_envs=size, fresh_training=True)
    clean_benchmark_shards(benchmark)
    run_stage(journal, "preprocessing", [args.python, "tools/prepare_joint_physics_parallel.py",
        "--config", str(preparation_path), "--output", str(args.sidecar), "--gpus", ",".join(args.gpus),
        "--num-envs", str(size)], cpu_env, args.repo, args.poll_seconds)
    summary = json.loads((args.sidecar/"summary.json").read_text())
    index = json.loads((args.sidecar/"index.json").read_text())
    validate_preprocessing(summary, index, reference, args.expected_sources)
    if config_path.read_bytes() != config_bytes:
        raise ValueError("预处理期间原训练配置改变，禁止静默改变网络/Loss/随机初始化合同")
    require_random_training_config(json.loads(config_path.read_text()), args)
    if args.resume_output.exists():
        raise FileExistsError("随机重训输出已存在，禁止重复启动或覆盖")
    require_idle_gpus(args.gpus)
    journal.update("preprocessing_verified_for_fresh_training", fresh_training=True,
        covered_sources=summary["covered_catalog_sources"], initial_global_step=0,
        legal_h32_windows=summary["all_published_legal_h32"]+summary.get("excluded_legal_h32", 0))
    env = dict(cpu_env, CUDA_VISIBLE_DEVICES=",".join(args.gpus), JOINT_GPUS="8", JOINT_PYTHON=args.python)
    command = ["bash", "scripts/train_joint_physics.sh", str(config_path),
               "--max-steps", str(args.max_steps), "--output", str(args.resume_output)]
    run_stage(journal, "fresh_training", command, env, args.repo, args.poll_seconds)
    final = json.loads((args.resume_output/"summary.json").read_text())
    if final.get("status") == "paused":
        journal.update("fresh_training_paused", fresh_training=True, global_step=final["global_step"])
    elif final.get("global_step") != args.max_steps:
        raise ValueError("随机训练退出但未达到200000步，保留现场不自动重启")
    else:
        journal.update("completed", fresh_training=True, training_output=str(args.resume_output), max_steps=args.max_steps)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--sidecar", type=Path, required=True)
    parser.add_argument("--torchrun-pid", type=int)
    parser.add_argument("--collector-pid", type=int)
    parser.add_argument("--output", type=Path, required=True, help="新的独立流水线日志目录")
    parser.add_argument("--training-output", "--resume-output", dest="resume_output", type=Path, required=True)
    parser.add_argument("--fresh-from-stopped-run", action="store_true", help="仅明确授权丢弃且旧任务已停时使用；预处理后从零重训")
    parser.add_argument("--stop-evidence", type=Path, help="含具体原PID身份、丢弃授权、all_exited和gpu_idle的JSON")
    parser.add_argument("--repo", type=Path, default=ROOT)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--gpus", type=parse_gpus, default=parse_gpus("0,1,2,3,4,5,6,7"))
    parser.add_argument("--checkpoint-step", type=int, default=5000)
    parser.add_argument("--max-steps", type=int, default=200000)
    parser.add_argument("--expected-sources", type=int, default=4287)
    parser.add_argument("--poll-seconds", type=float, default=30.)
    parser.add_argument("--exit-timeout", type=float, default=180.)
    parser.add_argument("--gpu-idle-timeout", type=float, default=300.)
    args = parser.parse_args()
    if args.fresh_from_stopped_run:
        if not args.stop_evidence or args.torchrun_pid or args.collector_pid:
            parser.error("从零模式要求--stop-evidence，不接受需要停止的任务PID")
        args.stop_evidence = args.stop_evidence.expanduser().resolve(strict=True)
    elif not args.torchrun_pid or args.stop_evidence:
        parser.error("完整恢复模式要求--torchrun-pid；停机证据仅用于显式从零模式")
    for key in ("run", "sidecar", "repo", "output", "resume_output"):
        setattr(args, key, getattr(args, key).expanduser().resolve())
    if args.gpus != [str(i) for i in range(8)] or not 0 < args.checkpoint_step < args.max_steps:
        parser.error("本流水线限定授权GPU0..7及合法checkpoint/总步数")
    if min(args.poll_seconds, args.exit_timeout, args.gpu_idle_timeout, args.expected_sources) <= 0:
        parser.error("间隔、超时与来源数必须为正")
    if args.resume_output == args.run or args.resume_output.exists() or args.output.exists():
        parser.error("流水线及恢复输出必须为新的独立目录")
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    import torch
    torch.set_num_threads(4)
    with pipeline_lock(args.run):
        args.output.mkdir(parents=True, exist_ok=False)
        journal = Journal(args.output)
        try:
            if args.fresh_from_stopped_run:
                execute_fresh(args, journal)
            else:
                journal.update("pidfd_preflight", **pidfd_preflight())
                execute(args, journal)
        except BaseException as error:
            journal.update("failed", error=repr(error), automatic_restart=False)
            raise
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
