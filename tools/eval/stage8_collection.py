"""严格读取 Stage8 显式多次运行集合，不伪造一次连续评估或改写原始证据。

本模块只依赖标准库。调用方提供带 SHA256 的 collection manifest，逐一列出原
run 目录和选中的原 episode ID。它核验每个原 run 的退出、冻结、源码未变及协议
证据，并比较模型、Isaac 资产、环境参数和实测延迟校准身份；仅允许音乐选择器与
评估入口/报告等离线编排文件不同。校准复用来源路径可以不同，实测值必须完全相同。

函数返回带原目录上下文的记录，报告继续读取原 trace/events；不复制 trace，不
改 episode/plan ID，不创建假的 merged run_summary，不自动根据成功或失败删样。
每条选中 trace 的 SHA 在报告流式读取后再次核验。独立运行中相同 episode ID 用
source_run_id 加以区分。适用于关闭视频的显式派生统计集合，保留所有原运行证据。
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re


COLLECTION_SCHEMA = "genmo.closedloop_collection.v1"
REQUIRED_ARTIFACTS = (
    "audit.json", "gmt_identity.json", "actor_identity.json", "worker_shutdown.json",
    "source_verification.json", "source_provenance.json", "calibration.json", "events.jsonl",
)
ALLOWED_CHANGED_SOURCES = {
    ("genmo_repo", "gem/closedloop/evaluation_music.py"),
    ("genmo_repo", "tools/eval/run_closedloop_baseline.py"),
    ("genmo_repo", "tools/eval/report_stage8_music_sweep.py"),
    ("genmo_repo", "tools/eval/stage8_collection.py"),
}


def _json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def _digest(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024*1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _bound_file(root, name, expected):
    _require(isinstance(expected, str) and re.fullmatch("[0-9a-f]{64}", expected), f"Invalid SHA binding: {name}")
    path = (root/name).resolve()
    _require(path.is_relative_to(root), f"Artifact escapes source run: {name}")
    _require(path.is_file() and _digest(path) == expected, f"Source artifact SHA mismatch: {path}")
    return path


def load_collection(manifest_path):
    """返回已校验源上下文及可序列化 provenance；任何不匹配明确报错，不宽松合并。"""
    path = Path(manifest_path).resolve()
    manifest = _json(path)
    _require(manifest.get("schema") == COLLECTION_SCHEMA, "Unknown Stage8 collection schema")
    _require(bool(manifest.get("sources")), "Collection sources must be nonempty")
    contexts, evidence, seen_ids, seen_roots = [], [], set(), set()
    common_identity = None
    common_core = None
    for source in manifest["sources"]:
        run_id = source["source_run_id"]
        _require(isinstance(run_id, str) and re.fullmatch(r"[A-Za-z0-9_-]+", run_id), "Invalid source_run_id")
        _require(run_id not in seen_ids, f"Duplicate source_run_id: {run_id}")
        seen_ids.add(run_id)
        root = (path.parent/source["run_dir"]).resolve()
        _require(root not in seen_roots, f"Same source run listed twice: {root}")
        seen_roots.add(root)
        summary = _json(_bound_file(root, "run_summary.json", source["run_summary_sha256"]))
        bindings = source["artifact_sha256"]
        artifact_paths = {name: _bound_file(root, name, bindings.get(name)) for name in REQUIRED_ARTIFACTS}
        artifacts = {name: _json(item) for name, item in artifact_paths.items() if name.endswith(".json")}
        audit, actor, gmt = (artifacts[name] for name in ("audit.json", "actor_identity.json", "gmt_identity.json"))
        shutdown, verification, provenance, calibration = (artifacts[name] for name in
            ("worker_shutdown.json", "source_verification.json", "source_provenance.json", "calibration.json"))
        _require(summary.get("exit_code") == 0 and not (root/"failure.json").exists(), f"Source run did not exit cleanly: {run_id}")
        _require(audit.get("status") == "passed", f"Source audit did not pass: {run_id}")
        _require(verification.get("unchanged") is True and not verification.get("changed_files"), f"Source code changed while running: {run_id}")
        _require(summary.get("source_verification") == verification, f"Summary/source verification mismatch: {run_id}")
        _require(provenance.get("source_manifest_sha256") == verification.get("initial_manifest_sha256"), f"Source manifest binding mismatch: {run_id}")
        _require(summary.get("acceptance", {}).get("frozen_models_and_environment") is True, f"Missing frozen run acceptance: {run_id}")
        for worker in ("actor", "gmt"):
            state = shutdown.get(worker, {})
            _require(state.get("closed") is True and state.get("process_exit_code") == 0, f"Worker did not close cleanly: {run_id}/{worker}")
        _require(all(shutdown["actor"].get("frozen_checks", {}).get(key) is True for key in
            ("eval_and_no_grad", "parameters_and_buffers_unchanged", "asset_files_unchanged")), f"Actor freeze mismatch: {run_id}")
        _require(shutdown["actor"].get("parameter_fingerprint") == actor.get("parameter_fingerprint") and
                 shutdown["actor"].get("sha256") == actor.get("sha256"), f"Actor initial/final identity mismatch: {run_id}")
        _require(shutdown["gmt"].get("policy_unchanged") is True and shutdown["gmt"].get("runtime_parameters_unchanged") is True and
                 shutdown["gmt"].get("policy_sha256") == gmt.get("policy_sha256") and
                 shutdown["gmt"].get("runtime_fingerprint_sha256") == gmt.get("runtime_fingerprint", {}).get("sha256"),
                 f"GMT initial/final identity mismatch: {run_id}")
        _require(calibration.get("full_calibration") is True and calibration.get("warmup_requests") == 10 and
                 calibration.get("measured_requests") == 100, f"Incomplete latency calibration: {run_id}")
        calibration_identity = {k: v for k, v in calibration.items() if k != "reused_from"}
        identity = {"actor_sha256": actor.get("sha256"), "actor_parameters": actor.get("parameter_fingerprint"),
            "actor_interface": actor.get("interface"), "gmt": {key: gmt.get(key) for key in
                ("policy_sha256", "asset_sha256", "runtime_fingerprint", "initial_ground_pose", "termination", "control_hz", "physics_hz", "physics_device")},
            "termination": summary.get("termination"), "calibration": calibration_identity}
        core = {(item["repository"], item["relative_path"]): item["sha256"] for item in provenance.get("files", [])
                if (item["repository"], item["relative_path"]) not in ALLOWED_CHANGED_SOURCES}
        _require(bool(core), f"Source code manifest is empty: {run_id}")
        if common_identity is None:
            common_identity, common_core = identity, core
        else:
            _require(identity == common_identity, f"Collection model/runtime/termination/calibration mismatch: {run_id}")
            _require(core == common_core, f"Collection core source mismatch: {run_id}")
        requested = source["episodes"]
        _require(bool(requested), f"No selected source episodes: {run_id}")
        requested_by_id = {item["episode_id"]: item for item in requested}
        _require(len(requested_by_id) == len(requested), f"Duplicate selected episode: {run_id}")
        originals = {item["episode_id"]: item for item in summary["episodes"]}
        _require(len(originals) == len(summary["episodes"]), f"Duplicate original episode identity: {run_id}")
        audit_ids = {item["episode_id"]: item for item in audit["episodes"]}
        selected = []
        for episode_id, selected_spec in requested_by_id.items():
            _require(episode_id in originals and originals[episode_id].get("mode") != "calibration", f"Unknown/non-evaluation selected episode: {run_id}/{episode_id}")
            _require(audit_ids.get(episode_id, {}).get("status") == "passed", f"Selected episode audit failed: {run_id}/{episode_id}")
            _require(bool(re.fullmatch("[0-9a-f]{64}", selected_spec.get("trace_sha256", ""))), f"Missing selected trace SHA: {run_id}/{episode_id}")
            selected.append(originals[episode_id])
        contexts.append({"source_run_id": run_id, "root": root, "summary": summary, "audit": audit,
                         "identity": gmt, "episodes": selected, "selected_by_id": requested_by_id})
        evidence.append({"source_run_id": run_id, "run_dir": str(root), "run_summary_sha256": source["run_summary_sha256"],
            "artifact_sha256": bindings, "selected_episode_count": len(selected), "excluded_episode_ids":
            [item["episode_id"] for item in summary["episodes"] if item.get("mode") != "calibration" and item["episode_id"] not in requested_by_id],
            "exit_code": 0, "frozen_and_source_checks_passed": True, "audit_status": "passed"})
    return contexts, {"schema": COLLECTION_SCHEMA, "manifest_path": str(path), "manifest_sha256": _digest(path),
        "manifest": manifest, "sources": evidence, "source_count": len(contexts), "source_checks_passed": True,
        "interpretation": "显式多次运行的派生集合：从各原始run重读所选真实episode，不是一次连续进程运行；未改写任何原episode或plan编号。"}
