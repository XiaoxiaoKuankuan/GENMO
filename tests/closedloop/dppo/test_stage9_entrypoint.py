"""第九步 CLI 的 CPU 预检、报告和关闭失败传播测试。

这些用例通过 monkeypatch 禁用大模型加载、GPU 检查和 Isaac worker，只核验配置及
入口控制流。特别检查 preflight 不启动 worker、不消耗物理预算，以及 eval 的关闭
失败必须同时反映在最终报告和进程返回码。替身不形成真实训练验收证据，全部输出
在 pytest tmp_path 中自动清理，禁止把这些测试解释为实际机器人或策略运行成功。
"""
from __future__ import annotations

import copy
import json
from types import SimpleNamespace

import torch
import pytest
import yaml

from tools import train_closedloop_dppo as entry


def setup_entrypoint(monkeypatch, tmp_path, *, ready=True):
    config = entry.configuration(entry.ROOT / "configs/closedloop/stage9_dppo_smoke.yaml")
    config["runtime"]["genmo_device"] = "cpu"
    config["stage9"]["bc_data_root"] = str(tmp_path / "bc")
    for name in ("AIST++", "AIOZ-GDANCE", "FineDance", "Mine"):
        manifest = tmp_path / "bc" / name / "manifests/train.jsonl"
        manifest.parent.mkdir(parents=True)
        manifest.write_text("unit-test-only")
    monkeypatch.setattr(entry, "configuration", lambda _path: copy.deepcopy(config))
    class Sampler:
        def __init__(self, *_args, **_kwargs):
            self.selected = []
            self.selection_sha256 = "unit-test-only"
    monkeypatch.setattr(entry, "TrainMusicSampler", Sampler)
    calls = []
    def preflight(_config, _selected, *, check_gpu):
        calls.append(check_gpu)
        return {"ready": ready, "repositories": {}, "asset_sha256": {}}
    monkeypatch.setattr(entry, "preflight", preflight)
    monkeypatch.setattr(entry, "collect_source_provenance", lambda *_args, **_kwargs: {"test_only": True, "source_manifest_sha256": "unit-test-only"})
    monkeypatch.setattr(entry, "verify_source_provenance", lambda *_args, **_kwargs: {"unchanged": True, "test_only": True})
    return config, calls


def test_preflight_does_not_construct_actor_or_worker(monkeypatch, tmp_path):
    _config, calls = setup_entrypoint(monkeypatch, tmp_path)
    def forbidden(*_args, **_kwargs):
        raise AssertionError("preflight may not launch a model or worker")
    monkeypatch.setattr(entry, "load_actor", forbidden)
    monkeypatch.setattr(entry, "Workers", forbidden)
    output = tmp_path / "preflight"
    assert entry.main(["--mode", "preflight", "--output-dir", str(output)]) == 0
    summary = json.loads((output / "acceptance/summary.json").read_text())
    assert summary["preflight_only"] and summary["status"] == "passed"
    assert not summary["stage9_passed"]
    assert calls == [False]
    assert all(value == 0 for value in summary["budget"]["used"].values())


def test_preflight_failure_has_report_and_nonzero_exit(monkeypatch, tmp_path):
    setup_entrypoint(monkeypatch, tmp_path, ready=False)
    output = tmp_path / "failed_preflight"
    assert entry.main(["--mode", "preflight", "--output-dir", str(output)]) == 1
    summary = json.loads((output / "acceptance/summary.json").read_text())
    assert summary["status"] == "failed" and summary["exit_code"] == 1
    assert "preflight" in summary["error"]["message"]


def test_configuration_contains_resolved_reward_and_keeps_latency():
    config = entry.configuration(entry.ROOT / "configs/closedloop/stage9_dppo_smoke.yaml")
    reward = config["stage9"]["reward"]
    assert config["stage9"]["execution_mode"] == "latency"
    assert reward['version'] == 'stage9.execution_reward.v2'
    assert reward['track_weight'] == 2.5 and reward['music_weight'] == 2.
    assert reward['activity']['window_s'] == .5
    assert reward['tracking']['std']['joint_vel'] == 1.4
    assert reward['tracking']['objective'] == 'gmt.motion_tracking.v1'
    assert reward['tracking']['weights']['joint_vel'] == 0.
    assert reward['consistency']['joint_vel_rms_rad_s'] == 1e-4
    assert 'angular_velocity_rad_s' not in reward['scales']
    assert 'joint_acceleration_rad_s2' not in reward['scales']
    assert reward['diagnostics'] == {'mechanical_power_weight': 0., 'impact_weight': 0.}
    server = entry.configuration(entry.ROOT/'configs/closedloop/stage9_dppo_server1.yaml')
    assert server['stage9']['reward'] == reward
    assert server['stage9']['actor_lr_candidates'] == [1e-9,3e-9,1e-8]
    assert server['stage9']['actor_lr_selection'] == 'largest_candidate_below_joint_kl_limit'


@pytest.mark.parametrize('overrides', [
    {'actor_lr_candidates':[1e-9,1e-9]},
    {'actor_lr_candidates':[3e-9,1e-9]},
    {'actor_lr_candidates':[1e-9,float('nan')]},
    {'actor_lr_candidates':[1e-9,1e-8,1e-7,1e-6]},
    {'actor_lr_candidates':[3e-9,1e-8]},
    {'actor_lr_selection':'ignore_kl'},
    {'max_iterations':2},
    {'guidance_scale':3.},
    {'critic_steps':21},
    {'critic_batch':64},
])
def test_configuration_rejects_ambiguous_or_unbudgeted_calibration(tmp_path, overrides):
    source=entry.ROOT/'configs/closedloop/stage9_dppo_smoke.yaml'
    config=yaml.safe_load(source.read_text())
    config['base_config']=str((source.parent/config['base_config']).resolve())
    config['stage9'].update(overrides)
    path=tmp_path/'config.yaml'
    path.write_text(yaml.safe_dump(config))
    with pytest.raises(ValueError):
        entry.configuration(path)


def test_eval_shutdown_failure_is_nonzero_exit(monkeypatch, tmp_path):
    setup_entrypoint(monkeypatch, tmp_path)
    actor = torch.nn.Linear(1, 1)
    actor.endecoder = SimpleNamespace(mean=torch.zeros(30), std=torch.ones(30))
    actor.interface_config = {"test_only": True}
    train = SimpleNamespace(model=SimpleNamespace(proprio_scales=[1., 1., 1., 1.]))
    monkeypatch.setattr(entry, "load_actor", lambda _config: (actor, train, {"test_only": True}))
    monkeypatch.setattr(entry, "DPPODiffusionPolicy", lambda *_args, **_kwargs: SimpleNamespace())
    class Workers:
        def __init__(self, *_args):
            self.temp = SimpleNamespace(name=str(tmp_path))
            self.entries = [{"client": None}]
            self.shutdown = {}
        def start(self, *_args):
            return object()
        def close(self):
            self.shutdown = {"gmt": {"process_exit_code": 2, "close_error": "injected close failure"}}
    monkeypatch.setattr(entry, "Workers", Workers)
    monkeypatch.setattr(entry, "AcknowledgedBackend", lambda client, *_args, **_kwargs: SimpleNamespace(client=client))
    monkeypatch.setattr(entry, "BumiKinematics", lambda *_args: None)
    monkeypatch.setattr(entry, "BumiMotionFeatureCodec", lambda *_args: None)
    monkeypatch.setattr(entry, "OnlineConditionBuilder", lambda *_args: None)
    monkeypatch.setattr(entry, "UpperEnvironment", lambda *_args: SimpleNamespace())
    monkeypatch.setattr(entry, "calibrate", lambda *_args: {"test_only": True})
    monkeypatch.setattr(entry, "comparison", lambda *_args: {"test_only": True})
    output = tmp_path / "eval"
    result = entry.main(["--mode", "eval", "--output-dir", str(output)])
    summary = json.loads((output / "acceptance/summary.json").read_text())
    assert summary["status"] == "failed" and summary["exit_code"] == 1
    assert summary["worker_shutdown"]["gmt"]["close_error"] == "injected close failure"
    assert "error" not in summary  # 不能因更早的夹具错误而偶然通过关闭失败测试。
    assert result == 1
