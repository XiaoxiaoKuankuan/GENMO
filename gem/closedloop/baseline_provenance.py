"""采集冻结闭环的有限源码清单及校准计算环境证据，不加载任何模型。

Git HEAD 和 tracked diff 无法覆盖当前两个工作树中新写的未跟踪模块。本模块显式枚举
Stage8、复用 Actor/codec/DDIM、当前 GMT 任务及资产导入源文件，对实际磁盘内容逐个
计算 SHA256，并记录绝对路径、Git 分支/HEAD/dirty 和文件是否 tracked。机器人网格只
跟随当前 URDF 明确引用的文件，不递归扫描仓库、不读取训练数据或大 checkpoint。

应在启动 worker 前调用 collect_source_provenance，并在退出后调用
verify_source_provenance；运行开始后才采集的文件身份不能证明已导入进程的代码版本。
文件哈希与 Git 状态分别记录，提交/修改日志不会被误解释为所枚举执行源码改变。

collect_compute_context 只读取 CPU 描述、当前进程可用核和已有 preflight 的 GPU 查询
结果，不执行 GPU kernel。CPU 型号与可用核单独记录；选中 GPU 的既有进程 PID/名称、
瞬时负载和显存均为观测值，不将服务正常重启与模型配置改变混同。即使静态身份和后台
进程相同，实际负载仍可
变化，因此此证据不能把 P95 校准解释为硬实时保证。禁止采集完整命令行或环境变量。
"""

from __future__ import annotations

from collections.abc import Mapping
import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import xml.etree.ElementTree as ET


TASK = "source/NoetixRobot/NoetixRobot/tasks/mimic/mimic_noetix_bumi4340_mha_sonic"
ASSET = "source/NoetixRobot/NoetixRobot/assets/robots/bumi3_4340_lowpd"
ROBOT_URDF = ASSET + "/urdf/bumi3_4340_point.urdf"

# 明确维护可审阅的运行源码边界，不通过遍历整个 checkout 猜测依赖。
SOURCE_FILES = {
    "genmo_repo": {
        "stage8": (
            "gem/closedloop/frozen_actor.py", "gem/closedloop/online_conditions.py",
            "gem/closedloop/coordinator.py", "gem/closedloop/baseline_metrics.py",
            "gem/closedloop/evaluation_music.py", "gem/closedloop/baseline_provenance.py",
            "gem/closedloop/baseline_video.py", "tools/eval/audit_closedloop_baseline.py",
            "gem/runtime/closedloop_protocol.py", "tools/eval/run_closedloop_baseline.py",
            "configs/closedloop/stage8_frozen_isaac.yaml",
            "configs/closedloop/stage8_grounded20_isaac.yaml",
            "configs/closedloop/stage8_train200_latency_isaac.yaml",
            "configs/closedloop/stage8_train200_latency_server1.yaml",
        ),
        "reused_actor_and_sampling": (
            "gem/closedloop/__init__.py", "gem/closedloop/actor.py", "gem/closedloop/contracts.py",
            "gem/closedloop/checkpoint.py", "gem/closedloop/training.py",
            "gem/network/gem_denoiser.py", "gem/network/base_arch/embeddings/pe.py",
            "gem/network/base_arch/embeddings/rotary_embedding.py",
            "gem/network/base_arch/transformer/encoder_rope.py",
            "gem/network/base_arch/transformer/layer.py",
            "gem/diffusion_utils/gaussian_diffusion.py", "gem/diffusion_utils/respace.py",
            "gem/diffusion_utils/nn.py", "gem/diffusion_utils/losses.py", "gem/utils/net_utils.py",
        ),
        "reused_codec_and_metrics": (
            "gem/robots/bumi/feature_codec.py", "gem/robots/bumi/endecoder.py",
            "gem/robots/bumi/kinematics.py", "gem/robots/bumi/contacts.py",
            "gem/robots/bumi/metrics.py", "gem/robots/bumi/losses.py",
            "gem/utils/rotation_conversions.py", "gem/utils/music_features.py",
            "gem/runtime/bumi_music_contract.py",
        ),
    },
    "gmt_repo": {
        "stage8": (
            "scripts/rsl_rl/serve_frozen_gmt.py", "scripts/rsl_rl/bumi4340_onnx_policy.py",
            "scripts/bumi4340_policy_contract.py", "scripts/motion/closedloop_reference.py",
            "scripts/motion/bumi4340_reference_kinematics.py",
            *(f"{TASK}/closedloop/{name}.py" for name in
              ("__init__", "backend", "command", "config", "env", "history", "scene_evidence",
               "render_sync", "physical_diagnostics", "initial_ground_pose", "orientation_errors")),
        ),
        "reused_task_and_observations": (
            f"{TASK}/__init__.py", f"{TASK}/tracking_env_cfg.py",
            *(f"{TASK}/mdp/{name}.py" for name in
              ("__init__", "commands", "observations", "actions", "terminations", "events", "curriculum", "rewards")),
            "source/NoetixRobot/NoetixRobot/managers/__init__.py",
            "source/NoetixRobot/NoetixRobot/managers/observation_manager.py",
            "source/NoetixRobot/NoetixRobot/actuators/__init__.py",
            "source/NoetixRobot/NoetixRobot/actuators/delayed_implicit_actuator_pd.py",
        ),
        "asset_import": (
            "source/NoetixRobot/NoetixRobot/assets/__init__.py", f"{ASSET}/bumi.py",
            ROBOT_URDF, f"{ASSET}/mjcf/bumi3_4340.xml",
        ),
    },
}


def _git(root, *args, binary=False):
    result = subprocess.run(["git", "-C", str(root), *args], check=True, capture_output=True,
                            text=not binary, timeout=20)
    return result.stdout if binary else result.stdout.strip()


def _file_record(root, relative, category):
    declared = root / relative
    resolved = declared.resolve()
    if not resolved.is_relative_to(root):
        raise ValueError(f"Runtime source points outside its configured worktree: {declared}")
    if not resolved.is_file():
        raise FileNotFoundError(f"Required runtime provenance source is missing: {declared}")
    before = resolved.stat()
    digest = hashlib.sha256()
    with resolved.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    after = resolved.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise RuntimeError(f"Source changed during fingerprint capture: {resolved}")
    return {"relative_path": relative, "path": str(declared), "resolved_path": str(resolved),
            "category": category, "sha256": digest.hexdigest(), "bytes": after.st_size,
            "mtime_ns": after.st_mtime_ns}


def collect_source_provenance(paths: Mapping, *, repository_state: Mapping | None = None,
                              additional_files: Mapping | None = None) -> dict:
    """接受 Stage8 config.paths；可复用 preflight.repositories，返回可直接写 JSON 的证据。"""
    repositories, records = {}, []
    for repository, categories in SOURCE_FILES.items():
        root = Path(paths[repository]).expanduser().resolve()
        if repository_state is not None and repository in repository_state:
            state = dict(repository_state[repository])
            if Path(state["path"]).resolve() != root:
                raise ValueError(f"Preflight repository path mismatch: {repository}")
        else:
            state = {"path": str(root), "branch": _git(root, "branch", "--show-current"),
                     "head": _git(root, "rev-parse", "HEAD"),
                     "status": _git(root, "status", "--short", "--untracked-files=normal")}
        state["dirty"] = bool(state["status"])
        repositories[repository] = state
        entries = {relative: category for category, relatives in categories.items() for relative in relatives}
        for relative in (additional_files or {}).get(repository, ()):
            entries[relative] = "stage9_training"
        if repository == "gmt_repo":
            urdf = (root / ROBOT_URDF).resolve()
            if not urdf.is_relative_to(root):
                raise ValueError("Robot URDF escapes the configured GMT worktree")
            for mesh in ET.parse(urdf).findall(".//mesh"):
                filename = mesh.attrib.get("filename")
                if not filename or "://" in filename:
                    raise ValueError("Runtime URDF provenance requires explicit local mesh filenames")
                mesh_path = (urdf.parent / filename).resolve()
                if not mesh_path.is_relative_to(root):
                    raise ValueError("Robot mesh escapes the configured GMT worktree")
                entries.setdefault(mesh_path.relative_to(root).as_posix(), "asset_mesh")
        tracked = set(_git(root, "ls-files", "-z", "--", *sorted(entries), binary=True).decode("utf-8").split("\0"))
        for relative, category in sorted(entries.items()):
            record = _file_record(root, relative, category)
            record.update(repository=repository, git_tracked=relative in tracked)
            records.append(record)
    # 还原完整清单的捕获边界：不能在先前已哈希的文件变化之后继续报告一致快照。
    for record in records:
        current = Path(record["path"]).resolve()
        info = current.stat()
        if str(current) != record["resolved_path"] or (info.st_size, info.st_mtime_ns) != (record["bytes"], record["mtime_ns"]):
            raise RuntimeError(f"Source changed before fingerprint capture completed: {record['path']}")
    content = [(record["repository"], record["relative_path"], record["sha256"]) for record in records]
    digest = hashlib.sha256(json.dumps(content, ensure_ascii=False, separators=(",", ":")).encode("utf-8")).hexdigest()
    return {"schema": "genmo.closedloop_stage8_source_provenance.v1",
            "captured_utc": datetime.now(timezone.utc).isoformat(),
            "repositories": repositories, "files": records, "file_count": len(records),
            "untracked_file_count": sum(not record["git_tracked"] for record in records),
            "source_manifest_sha256": digest,
            "evidence_boundary": "Explicit on-disk runtime source and URDF mesh inventory; capture before worker import. External library binaries are not source-hashed here."}


def verify_source_provenance(snapshot: Mapping) -> dict:
    """复核已记录的有限清单；返回差异，不读取整个仓库或自动覆盖原快照。"""
    if snapshot.get("schema") != "genmo.closedloop_stage8_source_provenance.v1":
        raise ValueError("Unsupported source provenance schema")
    changed = []
    for original in snapshot["files"]:
        root = Path(snapshot["repositories"][original["repository"]]["path"]).resolve()
        try:
            current = _file_record(root, original["relative_path"], original["category"])
            if current["sha256"] != original["sha256"] or current["resolved_path"] != original["resolved_path"]:
                changed.append({"path": original["path"], "before_sha256": original["sha256"],
                                "after_sha256": current["sha256"], "reason": "content_or_import_path_changed"})
        except (FileNotFoundError, ValueError, RuntimeError) as exc:
            changed.append({"path": original["path"], "reason": str(exc)})
    return {"unchanged": not changed, "checked_file_count": len(snapshot["files"]),
            "changed_files": changed, "initial_manifest_sha256": snapshot["source_manifest_sha256"]}


def collect_compute_context(preflight_gpu: Mapping | None = None) -> dict:
    """补充校准的 CPU/后台进程身份；沿用已有 GPU 查询，不调用 CUDA 或加载网络。"""
    cpuinfo = Path("/proc/cpuinfo")
    cpu = {}
    if cpuinfo.is_file():
        for line in cpuinfo.read_text(encoding="utf-8").splitlines():
            key, sep, value = line.partition(":")
            if sep and key.strip() in ("model name", "vendor_id", "cpu family", "model", "stepping", "microcode"):
                cpu.setdefault(key.strip(), set()).add(value.strip())
    identity = {"cpu": {"descriptors": {key: sorted(values) for key, values in sorted(cpu.items())},
                        "machine": platform.machine(), "logical_cpu_count": os.cpu_count(),
                        "allowed_cpu_ids": sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None}}
    observed = {"load_average_1_5_15m": list(os.getloadavg()) if hasattr(os, "getloadavg") else None}
    gpu = dict(preflight_gpu or {})
    selected_uuid = gpu.get("identity", {}).get("uuid")
    processes, memory = [], []
    for row in csv.reader(gpu.get("existing_compute_processes", "").splitlines()):
        if not row or not selected_uuid or row[0].strip() != selected_uuid:
            continue
        if len(row) != 4:
            raise ValueError("Unexpected preflight GPU process row")
        gpu_uuid, pid, name, usage = (field.strip() for field in row)
        process = {"gpu_uuid": gpu_uuid, "pid": int(pid), "process_name": name}
        processes.append(process)
        memory.append({**process, "used_memory_mib": int(usage) if usage.isdigit() else usage})
    observed.update(background_gpu_processes=sorted(processes, key=lambda row: (row["gpu_uuid"], row["pid"])),
                    background_gpu_memory=memory, free_gpu_memory_mib=gpu.get("free_memory_mib"))
    return {"identity": identity, "observed": observed,
            "timing_boundary": "Matching CPU/device/process identity does not guarantee identical instantaneous contention or a hard latency bound."}
