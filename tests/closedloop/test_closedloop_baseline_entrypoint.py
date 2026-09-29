"""验证 Stage8 启动前检查、校准身份复用和最终验收标记，不启动真实 worker。

使用 pytest 临时目录中的小文件、模拟 nvidia-smi/Git 输出和固定清单，检查缺失输入、
GPU共享授权与显存下限、参数提前拒绝，以及校准对模型/资产/主机/设备/线程数的绑定。
录像路径不参与计算身份，不能因为录像要求无意义重校准。小规模执行和成功退出不会被
自动标成完整48组动力学基线。另以启动异常替身验证失败报告和清理路径；不占用GPU，
不运行 Isaac，不修改正式模型、数据或实验目录。
"""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import yaml

from tools.eval import run_closedloop_baseline as entry


@pytest.fixture
def settings(tmp_path, monkeypatch):
    config = yaml.safe_load((entry.ROOT / "configs/closedloop/stage8_frozen_isaac.yaml").read_text())
    config["runtime"]["genmo_device"] = "cpu"
    paths = config["paths"]
    for key in (*entry.ASSET_KEYS, "genmo_python", "isaac_python"):
        path = tmp_path / key
        path.write_bytes(key.encode())
        paths[key] = str(path)
    paths["genmo_repo"] = paths["gmt_repo"] = str(tmp_path)
    paths["data_root"] = str(tmp_path / "data")
    config["output_root"] = str(tmp_path / "outputs")
    root = Path(paths["data_root"]) / "Mine"
    root.mkdir(parents=True)
    samples = []
    for index in range(2):
        feature, audio = root / f"feature{index}.pt", root / f"audio{index}.wav"
        feature.write_bytes(b"feature")
        audio.write_bytes(b"audio")
        samples.append({"dataset": "Mine", "group_id": str(index), "row": {
            "sample_id": f"mine{index}", "music_feature_path": feature.name, "audio_path": audio.name,
            "source_music_feature_sha256": hashlib.sha256(feature.read_bytes()).hexdigest(),
            "source_audio_sha256": hashlib.sha256(audio.read_bytes()).hexdigest(),
        }})

    def command(args, cwd=None):
        if args[0] == "git":
            return "test-git"
        if "--query-gpu=" in args[1]:
            return "0, GPU-test-uuid, Test GPU Model, 10000, 24576, 3, 580.10"
        if "--query-compute-apps=" in args[1]:
            return "GPU-test-uuid, 123, existing-text-service, 10000"
        raise AssertionError(args)
    monkeypatch.setattr(entry, "_command", command)
    monkeypatch.setattr(entry.subprocess, "check_output", lambda *args, **kwargs: b"test-diff")
    monkeypatch.setattr(entry.platform, "processor", lambda: "test-cpu")
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    return config, samples


def calibration_for(identity):
    durations = [.1, .2]
    p95 = float(np.percentile(durations, 95))
    guard = identity["timing"]["latency_guard_s"]
    return {"warmup_requests": 1, "measured_requests": 2, "end_to_end_seconds": durations,
            "p95_seconds": p95, "guard_seconds": guard, "latency_budget_seconds": p95 + guard,
            "full_calibration": False, "identity": copy.deepcopy(identity)}


def test_missing_required_file_is_reported_without_starting_workers(settings):
    config, samples = settings
    Path(config["paths"]["checkpoint"]).unlink()
    report = entry.preflight(config, samples, check_gpu=False)
    assert not report["ready"]
    assert report["missing_required_files"] == [config["paths"]["checkpoint"]]
    assert "checkpoint" not in report["asset_sha256"]


def test_preflight_records_full_device_and_asset_identity(settings):
    config, samples = settings
    config["runtime"]["genmo_device"] = "cuda:0"
    report = entry.preflight(config, samples)
    assert report["ready"]
    assert report["gpu"]["identity"] == {"index": 0, "uuid": "GPU-test-uuid", "model": "Test GPU Model", "driver_version": "580.10"}
    assert report["gpu"]["free_memory_mib"] == 14576
    assert report["gpu"]["sharing_authorized"] is True
    assert set(report["asset_sha256"]) == set(entry.ASSET_KEYS)
    assert report["host"]["hostname"]


def test_preflight_does_not_override_gpu_sharing_or_memory_rules(settings, monkeypatch):
    config, samples = settings
    config["runtime"]["genmo_device"] = "cuda:0"
    config["runtime"]["allow_shared_gpu"] = False
    with pytest.raises(RuntimeError, match="sharing was not authorized"):
        entry.preflight(config, samples)
    config["runtime"]["allow_shared_gpu"] = True
    original = entry._command
    monkeypatch.setattr(entry, "_command", lambda args, cwd=None: original(args, cwd).replace("10000, 24576", "20000, 24576"))
    with pytest.raises(RuntimeError, match="8 GiB"):
        entry.preflight(config, samples)


@pytest.mark.parametrize("field", ["checkpoint", "stats", "kinematics", "gmt_policy", "compat_profile", "isaac_contract"])
def test_calibration_reuse_rejects_any_asset_change(settings, field):
    config, samples = settings
    identity = entry.calibration_identity(config, entry.preflight(config, samples))
    calibration = calibration_for(identity)
    changed = copy.deepcopy(identity)
    changed["asset_sha256"][field] = "different"
    with pytest.raises(ValueError, match="identity mismatch"):
        entry.validate_calibration(calibration, changed)
    assert calibration["identity"] == identity


@pytest.mark.parametrize("section,key", [("host", "hostname"), ("accelerator", "model"),
                                        ("model", "ddim_steps"), ("runtime", "torch_threads"),
                                        ("runtime", "genmo_device")])
def test_calibration_reuse_binds_host_gpu_model_and_compute_configuration(settings, section, key):
    config, samples = settings
    identity = entry.calibration_identity(config, entry.preflight(config, samples))
    changed = copy.deepcopy(identity)
    changed[section][key] = "changed"
    with pytest.raises(ValueError, match="identity mismatch"):
        entry.validate_calibration(calibration_for(identity), changed)


def test_video_and_gui_do_not_invalidate_same_compute_calibration(settings):
    config, samples = settings
    check = entry.preflight(config, samples)
    identity = entry.calibration_identity(config, check)
    changed = copy.deepcopy(config)
    changed["runtime"].update(video_path="/tmp/other.mp4", headless=False)
    changed["evaluation"].update(datasets=["Mine"], seconds=5)
    second = entry.calibration_identity(changed, check)
    assert second == identity
    assert entry.validate_calibration(calibration_for(identity), second)["identity"] == identity


@pytest.mark.parametrize("key,value", [("measured_requests", 0), ("warmup_requests", -1),
                                       ("end_to_end_seconds", [float("nan"), .2]),
                                       ("latency_budget_seconds", .001), ("full_calibration", True)])
def test_calibration_rejects_invalid_or_mislabelled_evidence(settings, key, value):
    config, samples = settings
    identity = entry.calibration_identity(config, entry.preflight(config, samples))
    calibration = calibration_for(identity)
    calibration[key] = value
    with pytest.raises(ValueError):
        entry.validate_calibration(calibration, identity)


@pytest.mark.parametrize("section,key,value", [("timing", "physics_ticks", 4),
                                              ("timing", "calibration_warmup", -1),
                                              ("timing", "calibration_samples", 0),
                                              ("environment", "domain_randomization", True),
                                              ("termination", "root_height_error_m", .2),
                                              ("evaluation", "seconds", float("nan"))])
def test_unsupported_or_invalid_configuration_fails_before_loading(settings, section, key, value):
    config, _ = settings
    config[section][key] = value
    with pytest.raises(ValueError):
        entry.validate_config(config)


def test_small_selection_or_incomplete_calibration_never_claims_full_baseline(settings):
    config, samples = settings
    scope = entry.validation_scope(config, samples, 48, {"full_calibration": True})
    assert not scope["full_matrix"] and scope["small_scale_validation"]
    selected = [{"dataset": name, "group_id": str(group)} for name in entry.FULL_DATASETS for group in range(2)]
    scope = entry.validation_scope(config, selected, 48, {"full_calibration": False})
    assert scope["full_matrix"] and scope["small_scale_validation"]
    scope = entry.validation_scope(config, selected, 48, {"full_calibration": True})
    assert scope["full_matrix"] and not scope["small_scale_validation"]
    assert not entry.acceptance_flags({}, scope, {}, code=0, error=None)["baseline_ready"]


def test_zero_exit_does_not_hide_trace_or_physical_acceptance_failure(settings):
    config, samples = settings
    scope = entry.validation_scope(config, samples, 1, {"full_calibration": True})
    shutdown = {"actor": {"process_exit_code": 0, "frozen_checks": {
        "eval_and_no_grad": True, "parameters_and_buffers_unchanged": True, "asset_files_unchanged": True}},
        "gmt": {"process_exit_code": 0, "runtime_parameters_unchanged": True, "policy_unchanged": True}}
    episode = {"sample": {"dataset": "Mine"}, "failed": False, "startup_failure": False,
               "music_duration_seconds": 5.0, "replan_count": 10, "mode": "paused",
               "terminal_snapshot": {"terminated": False}, "trace_integrity": {"complete": True},
               "protected_reference": {"modification_count": 0}}
    flags = entry.acceptance_flags({"episodes": [episode]}, scope, shutdown, code=0, error=None)
    assert flags["no_system_errors"] and flags["frozen_models_and_environment"]
    assert not flags["continuous_mine_30s_with_multiple_replans"] and not flags["baseline_ready"]
    episode["music_duration_seconds"] = 30.0
    episode["trace_integrity"]["complete"] = False
    flags = entry.acceptance_flags({"episodes": [episode]}, scope, shutdown, code=0, error=None)
    assert not flags["no_system_errors"]


def test_manifest_selection_is_grouped_deterministic_and_prefers_longest(tmp_path):
    root = tmp_path / "Mine" / "manifests"
    root.mkdir(parents=True)
    rows = [{"sample_id": name, "num_frames": frames, "split": "val", "fps": 30,
             "resplit_provenance": {"group_id": group}} for name, frames, group in
            (("z", 90, "g1"), ("b", 100, "g0"), ("a", 100, "g0"), ("c", 70, "g0"), ("extra", 1000, "g2"))]
    (root / "val.jsonl").write_text("\n".join(json.dumps(row) for row in rows))
    selected = entry.select_val_music(tmp_path, ["Mine"], 2)
    assert [sample["row"]["sample_id"] for sample in selected] == ["a", "z"]
    selected = entry.select_val_music(tmp_path, ["Mine"], {"Mine": 3})
    assert [sample["row"]["sample_id"] for sample in selected] == ["a", "z", "extra"]


def test_twenty_music_sweep_is_distinct_from_original_matrix(settings):
    config, _ = settings
    config['evaluation'].update(groups_per_dataset={'AIST++': 2, 'AIOZ-GDANCE': 8, 'FineDance': 5, 'Mine': 5},
                                seeds=[42])
    entry.validate_config(config)
    selected = [{'dataset': name, 'group_id': f'{name}:{i}',
                 'row': {'source_audio_sha256': f'{name}:{i}'}}
                for name, n in config['evaluation']['groups_per_dataset'].items() for i in range(n)]
    scope = entry.validation_scope(config, selected, 40, {'full_calibration': True}, video=True)
    assert scope['requested_matrix_completed']
    assert scope['unique_audio_sha256_count'] == 20
    assert not scope['full_matrix']


def test_bad_calibration_counts_are_rejected_before_preflight_or_worker_start(settings, tmp_path, monkeypatch):
    config, _ = settings
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config))
    monkeypatch.setattr(entry, "preflight", lambda *args, **kwargs: pytest.fail("preflight must not run"))
    with pytest.raises(SystemExit) as exc:
        entry.main(["--config", str(path), "--calibration-samples", "0"])
    assert exc.value.code == 2


def test_preflight_rejects_mismatched_reused_calibration_before_workers(settings, tmp_path, monkeypatch):
    config, samples = settings
    config["evaluation"]["datasets"] = ["Mine"]
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config))
    identity = entry.calibration_identity(config, entry.preflight(config, samples))
    calibration = calibration_for(identity)
    calibration["identity"]["asset_sha256"]["stats"] = "wrong-stats"
    measured = tmp_path / "calibration.json"
    measured.write_text(json.dumps(calibration))
    monkeypatch.setattr(entry, "select_val_music", lambda *args: samples)
    monkeypatch.setattr(entry, "Workers", lambda *args: pytest.fail("worker must not start"))
    output = tmp_path / "bad_calibration"
    assert entry.main(["--config", str(path), "--output-dir", str(output),
                       "--calibration", str(measured), "--preflight"]) == 1
    error = json.loads((output / "failure.json").read_text())
    assert "identity mismatch" in error["message"]


def test_worker_startup_failure_writes_report_and_cleans_owned_workers(settings, tmp_path, monkeypatch):
    config, samples = settings
    config["paths"]["kinematics"] = str(entry.ROOT / "configs/bumi/bumi_kinematics_robot_retargeter_fe934_v1.json")
    config["evaluation"]["datasets"] = ["Mine"]
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config))
    check = entry.preflight(config, samples)
    monkeypatch.setattr(entry, "preflight", lambda *args, **kwargs: check)
    monkeypatch.setattr(entry, "select_val_music", lambda *args: samples)
    monkeypatch.setattr(entry, "collect_source_provenance", lambda *args, **kwargs: {})
    monkeypatch.setattr(entry, "collect_compute_context", lambda *args, **kwargs: {})
    monkeypatch.setattr(entry, "verify_source_provenance", lambda *args, **kwargs: {"unchanged": True})
    closed = []

    class FailedWorkers:
        def __init__(self, *args):
            self.shutdown = {}

        def launch(self, *args):
            raise RuntimeError("synthetic startup failure")

        def close(self):
            closed.append(True)

    monkeypatch.setattr(entry, "Workers", FailedWorkers)
    output = tmp_path / "failure_report"
    assert entry.main(["--config", str(path), "--output-dir", str(output)]) == 1
    assert closed == [True]
    assert "synthetic startup failure" in json.loads((output / "failure.json").read_text())["message"]
    report = json.loads((output / "run_summary.json").read_text())
    assert not report["acceptance"]["baseline_ready"] and not report["acceptance"]["no_system_errors"]
