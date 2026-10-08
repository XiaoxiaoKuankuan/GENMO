"""第十步完整数据池、划分隔离、可恢复采样和随机音乐窗口的 CPU 回归。

所有四库 train/val/test 清单、音频字节、音乐特征和配对动作均写入 pytest tmp_path，
运行命令使用系统临时 basetemp 并自动删除；不读生产数据、不加载 Actor、不占 GPU。
正例验证全清单而非200首子集、文件证据、来源概率与真实执行覆盖分离、随机状态恢复；
负例覆盖跨划分组/音频/动作泄漏、漏样本、文件漂移、路径越界、损坏checkpoint。
环境替身只用于核对起点同步和行政截断语义，不构成真实动力学或训练质量证据。
"""
from __future__ import annotations

import copy
import hashlib
import json

import numpy as np
import pytest
import torch

from gem.closedloop.contracts import GMT_EXPECTED_JOINT_ORDER
import gem.closedloop.dppo.full_dataset as full_dataset
from gem.closedloop.dppo.full_dataset import FullMusicCatalog, FullMusicSampler, SOURCES, SPLITS
from gem.closedloop.dppo.target_activity import load_paired_activity


def digest(value):
    return hashlib.sha256(value).hexdigest()


def make_catalog_data(tmp_path, *, frames=180):
    root = tmp_path / "full_data"
    for source in SOURCES:
        folder = root / source
        for sub in ("manifests", "meta", "motions", "music", "audio"):
            (folder / sub).mkdir(parents=True)
        info = dict(contract_version="genmo.bumi_music.v1", fps=30, robot_name="bumi",
            qpos_order="mujoco_native", quaternion_convention="wxyz", qpos_dim=28, joint_dim=21,
            quality_filter_applied=True, dataset_name=source + "_bumi", joint_names=list(GMT_EXPECTED_JOINT_ORDER),
            source_mjcf_sha256="a"*64, quality_config_sha256="b"*64, retarget_config_sha256="c"*64,
            ground_semantics="umr_foot_sole_ground_zero_v1", root_z_adjusted=False,
            split_counts=dict(train=2, val=1, test=1))
        (folder / "meta/dataset_info.json").write_text(json.dumps(info))
        for split in SPLITS:
            rows = []
            for i in range(info["split_counts"][split]):
                name = f"{split}_{i}"
                audio = (source + name).encode()
                (folder / f"audio/{name}.wav").write_bytes(audio)
                music = torch.zeros(frames, 35)
                music[:, 0] = torch.arange(frames)
                music[:, 1] = SOURCES.index(source)
                music_path = folder / f"music/{name}.pt"
                torch.save(music, music_path)
                row = dict(sample_id=name, split=split, dataset=info["dataset_name"], fps=30,
                    num_frames=frames, quality_accepted=True, motion_path=f"motions/{name}.pt",
                    music_feature_path=f"music/{name}.pt", audio_path=f"audio/{name}.wav",
                    source_audio_sha256=digest(audio), source_motion_sha256=digest(b"motion"+audio),
                    source_music_feature_sha256=digest(music_path.read_bytes()),
                    resplit_provenance={"group_id": digest(b"group"+audio)})
                q = torch.zeros(frames, 28, dtype=torch.float64)
                q[:, 3] = 1.
                q[:, 7:] = (torch.arange(frames, dtype=torch.float64) / 30.).square()[:, None]
                payload = {**info, "qpos": q, "source_motion_sha256": row["source_motion_sha256"],
                           "source_sample_id": name, "quality_accepted": True}
                torch.save(payload, folder / row["motion_path"])
                rows.append(row)
            (folder / f"manifests/{split}.jsonl").write_text("".join(json.dumps(row)+"\n" for row in rows))
    return root


def rewrite_row(root, source, split, mutate):
    path = root / source / f"manifests/{split}.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    mutate(rows)
    path.write_text("".join(json.dumps(row)+"\n" for row in rows))


def test_full_catalog_audits_every_row_with_current_payload_hash(tmp_path):
    root = make_catalog_data(tmp_path)
    catalog = FullMusicCatalog(root)
    seen = []
    report = catalog.audit_files(progress=seen.append)
    assert report["status"] == "passed" and report["complete_manifest_scan"]
    assert report["sample_count"] == 16 == len(seen)
    assert report["identity"]["sample_counts"]["train"] == dict.fromkeys(SOURCES, 2)
    for row in report["records"]:
        assert row["audio_verified"] and row["motion_payload_sha256"] == row["payload_sha256"]
        assert row["payload_sha256"] != row["source_motion_sha256"]
        assert row["source_motion_sha_validation"] == f"payload_equals_verified_{row['split']}_manifest_declaration"
    assert all(not x["pretraining_unseen_claim"] for s in SPLITS for source in SOURCES for x in catalog.samples[s][source])


@pytest.mark.parametrize("field", ["group", "audio", "motion"])
def test_cross_source_split_leakage_rejected_before_loading_payloads(tmp_path, field):
    root = make_catalog_data(tmp_path)
    train = json.loads((root / SOURCES[0] / "manifests/train.jsonl").read_text().splitlines()[0])
    def alter(rows):
        if field == "group":
            rows[0]["resplit_provenance"]["group_id"] = train["resplit_provenance"]["group_id"]
        else:
            key = "source_audio_sha256" if field == "audio" else "source_motion_sha256"
            rows[0][key] = train[key]
    rewrite_row(root, SOURCES[1], "val", alter)
    with pytest.raises(ValueError, match="cross-split leakage"):
        FullMusicCatalog(root)


def test_same_split_multiple_crops_or_dancers_remain_in_catalog(tmp_path, monkeypatch):
    root = make_catalog_data(tmp_path)
    def alter(rows):
        rows[1]["resplit_provenance"] = rows[0]["resplit_provenance"]
        rows[1]["source_audio_sha256"] = rows[0]["source_audio_sha256"]
        rows[1]["audio_path"] = rows[0]["audio_path"]
    rewrite_row(root, SOURCES[0], "train", alter)
    catalog = FullMusicCatalog(root)
    assert len(catalog.samples["train"][SOURCES[0]]) == 2
    from gem.closedloop.dppo import full_dataset
    original, audio_calls = full_dataset.sha256_file, []
    def traced(path):
        if str(path).endswith(".wav"):
            audio_calls.append(str(path))
        return original(path)
    monkeypatch.setattr(full_dataset, "sha256_file", traced)
    report = catalog.audit_files()
    assert report["unique_audio_files_hashed"] == 15 == len(audio_calls) == len(set(audio_calls))


@pytest.mark.parametrize("fault", ["missing_split", "short_manifest", "duplicate", "wrong_split", "escape"])
def test_incomplete_or_corrupt_catalog_is_not_silently_a_subset(tmp_path, fault):
    root = make_catalog_data(tmp_path)
    if fault == "missing_split":
        (root / SOURCES[0] / "manifests/val.jsonl").unlink()
    else:
        def alter(rows):
            if fault == "short_manifest":
                rows.pop()
            elif fault == "duplicate":
                rows[1]["sample_id"] = rows[0]["sample_id"]
            elif fault == "wrong_split":
                rows[0]["split"] = "val"
            else:
                rows[0]["motion_path"] = "../../unrelated.pt"
        rewrite_row(root, SOURCES[0], "train", alter)
    with pytest.raises((ValueError, FileNotFoundError)):
        FullMusicCatalog(root)


@pytest.mark.parametrize("fault", ["audio", "music", "motion", "missing_motion", "manifest_drift", "info_drift"])
def test_all_file_audit_fails_closed_on_real_content_or_identity(tmp_path, fault):
    root = make_catalog_data(tmp_path)
    catalog = FullMusicCatalog(root)
    folder = root / SOURCES[0]
    if fault in ("audio", "music"):
        path = folder / ("audio/train_0.wav" if fault == "audio" else "music/train_0.pt")
        path.write_bytes(b"changed")
    elif fault == "missing_motion":
        (folder / "motions/train_0.pt").unlink()
    elif fault == "motion":
        path = folder / "motions/train_0.pt"
        payload = torch.load(path, weights_only=False)
        payload["source_motion_sha256"] = "f"*64
        torch.save(payload, path)
    elif fault == "manifest_drift":
        rewrite_row(root, SOURCES[0], "train", lambda rows: rows[0].update(num_frames=179))
    else:
        (folder / "meta/dataset_info.json").write_text("{}")
    with pytest.raises((ValueError, FileNotFoundError)):
        catalog.audit_files()


def test_source_directory_symlinks_allowed_but_inside_file_escape_rejected(tmp_path):
    root = make_catalog_data(tmp_path)
    link = tmp_path / "linked"
    link.mkdir()
    for source in SOURCES:
        (link / source).symlink_to(root / source, target_is_directory=True)
    assert FullMusicCatalog(link).audit_files()["sample_count"] == 16
    path = root / SOURCES[0] / "music/train_0.pt"
    external = tmp_path / "outside.pt"
    external.write_bytes(path.read_bytes())
    path.unlink()
    path.symlink_to(external)
    with pytest.raises(ValueError, match="escapes"):
        FullMusicCatalog(link)


def test_audited_motion_contents_remain_bound_during_reward_loading(tmp_path):
    root = make_catalog_data(tmp_path)
    catalog = FullMusicCatalog(root)
    report = catalog.audit_files()
    sample = catalog.samples["train"][SOURCES[0]][0]
    assert catalog.identity["audited_data_content_sha256"] == report["data_content_sha256"]
    assert sample["motion_file_sha256"]
    path = root / SOURCES[0] / sample["row"]["motion_path"]
    payload = torch.load(path, weights_only=False)
    payload["qpos"][:, 7:] += .1  # 保留原来源声明与metadata，内容仍必须被拒绝。
    torch.save(payload, path)
    with pytest.raises(ValueError, match="changed after full dataset audit"):
        load_paired_activity(root, sample)


def test_broadcast_audit_binds_reconstructed_catalog_without_rescanning(tmp_path, monkeypatch):
    root = make_catalog_data(tmp_path)
    audited = FullMusicCatalog(root)
    report = audited.audit_files()
    reconstructed = FullMusicCatalog(root)
    assert reconstructed.identity != audited.identity

    def no_payload_scan(*args, **kwargs):
        raise AssertionError("apply_audit must not rescan payload files")

    with monkeypatch.context() as isolated:
        isolated.setattr(full_dataset, "load_music_features", no_payload_scan)
        isolated.setattr(full_dataset, "_load_verified_pair", no_payload_scan)
        isolated.setattr(reconstructed, "audit_files", no_payload_scan)
        assert reconstructed.apply_audit(report) is True
        assert reconstructed.apply_audit(report) is True  # root也可同值再次绑定。
    assert reconstructed.identity == audited.identity
    assert reconstructed.samples == audited.samples and reconstructed._lookup == audited._lookup
    for split in SPLITS:
        for source in SOURCES:
            sample = reconstructed.samples[split][source][0]
            np.testing.assert_array_equal(reconstructed.load_music(sample), audited.load_music(sample))
            assert reconstructed.validate_sample(sample)["motion_file_sha256"]
    first = FullMusicSampler(audited, seed=19)
    second = FullMusicSampler(reconstructed, seed=21)
    second.load_state_dict(first.state_dict())
    assert second.next_task()["sample"] == first.next_task()["sample"]
    report["identity"]["catalog_sha256"] = "0" * 64
    assert reconstructed.identity == audited.identity  # 不保留广播输入可写别名。


@pytest.mark.parametrize("fault", ["status", "identity", "content_sha", "missing", "duplicate", "unknown",
                                  "manifest", "payload", "audio", "path", "group"])
def test_broadcast_bad_audit_never_partially_changes_catalog(tmp_path, fault):
    root = make_catalog_data(tmp_path)
    report = FullMusicCatalog(root).audit_files()
    catalog = FullMusicCatalog(root)
    before = copy.deepcopy((catalog.identity, catalog.samples, catalog._lookup))
    record = report["records"][-1]
    if fault == "status":
        report["status"] = "failed"
    elif fault == "identity":
        report["identity"]["catalog_sha256"] = "0" * 64
    elif fault == "content_sha":
        report["data_content_sha256"] = "0" * 64
    elif fault == "missing":
        report["records"].pop()
    elif fault == "duplicate":
        report["records"][-1] = copy.deepcopy(report["records"][0])
    elif fault == "unknown":
        record["sample_id"] = "unknown"
    elif fault == "manifest":
        record["manifest_sha256"] = "0" * 64
    elif fault == "payload":
        record["payload_sha256"] = "0" * 64
    elif fault == "audio":
        record["audio_sha256"] = "0" * 64
    elif fault == "path":
        record["motion_path"] = "/tmp/wrong_dataset_payload.pt"
    else:
        record["group_id"] = "wrong-group"
    if fault not in ("status", "identity", "content_sha"):
        content = hashlib.sha256(json.dumps(report["records"], sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        report["data_content_sha256"] = content
        report["identity"]["audited_data_content_sha256"] = content
        report["sample_count"] = len(report["records"])
    with pytest.raises(ValueError):
        catalog.apply_audit(report)
    assert (catalog.identity, catalog.samples, catalog._lookup) == before


def test_broadcast_audit_keeps_motion_payload_drift_detection(tmp_path):
    root = make_catalog_data(tmp_path)
    report = FullMusicCatalog(root).audit_files()
    catalog = FullMusicCatalog(root)
    catalog.apply_audit(report)
    sample = catalog.samples["train"][SOURCES[0]][0]
    path = root / SOURCES[0] / sample["row"]["motion_path"]
    payload = torch.load(path, weights_only=False)
    payload["qpos"][:, 7:] += .1
    torch.save(payload, path)
    with pytest.raises(ValueError, match="changed after full dataset audit"):
        load_paired_activity(root, sample)


def test_broadcast_audit_accepts_existing_optional_null_payload_declaration(tmp_path):
    root = make_catalog_data(tmp_path)
    rewrite_row(root, SOURCES[0], "train", lambda rows: rows[0].update(motion_file_sha256=None))
    audited = FullMusicCatalog(root)
    report = audited.audit_files()
    catalog = FullMusicCatalog(root)
    catalog.apply_audit(report)
    assert catalog.samples == audited.samples and catalog.identity == audited.identity


def test_mutating_public_sample_does_not_mutate_catalog_validation_snapshot(tmp_path):
    catalog = FullMusicCatalog(make_catalog_data(tmp_path))
    sample = catalog.samples["train"][SOURCES[0]][0]
    sample["row"]["num_frames"] -= 1
    with pytest.raises(ValueError, match="manifest identity"):
        catalog.load_music(sample)


def test_random_windows_keep_complete_music_and_resume_exactly(tmp_path):
    catalog = FullMusicCatalog(make_catalog_data(tmp_path))
    first = FullMusicSampler(catalog, seed=17, window_seconds=2.)
    a = first.next_task()
    assert len(a["music"]) == 180 and 0 <= a["start_frame"] <= 120
    assert a["full_music_duration_seconds"] == 6.
    assert a["remaining_music_seconds"] == (180-a["start_frame"])/30.
    first.record_execution(a, 25)
    state = first.state_dict()
    restored = FullMusicSampler(catalog, seed=999, window_seconds=2.)
    restored.load_state_dict(state)
    for _ in range(6):
        x, y = first.next_task(), restored.next_task()
        assert x["sample"] == y["sample"] and x["start_frame"] == y["start_frame"]
        np.testing.assert_array_equal(x["music"], y["music"])
        first.record_execution(x, 15)
        restored.record_execution(y, 15)
    assert first.state_dict() == restored.state_dict()


def test_task_source_probabilities_are_not_claimed_as_control_time_mix(tmp_path, monkeypatch):
    catalog = FullMusicCatalog(make_catalog_data(tmp_path, frames=60))
    monkeypatch.setattr(catalog, "load_music", lambda sample: np.zeros((60,35)))
    sampler = FullMusicSampler(catalog, seed=42, window_seconds=1.)
    for _ in range(10000):
        task = sampler.next_task()
        sampler.record_execution(task, 1 if task["sample"]["dataset"] == "Mine" else 10)
    report = sampler.coverage()
    assert report["probability_scope"] == "tasks_not_control_time"
    for source, expected in zip(SOURCES, (.2,.35,.25,.2)):
        row = report["per_source"][source]
        assert abs(row["tasks_drawn"]/10000-expected) < .015
        assert row["unique_samples_drawn"] == 2
        assert row["control_steps"] == row["tasks_drawn"]*(1 if source == "Mine" else 10)


def test_actual_coverage_has_no_double_counted_source_time_and_rejects_stale_task(tmp_path):
    catalog = FullMusicCatalog(make_catalog_data(tmp_path))
    sampler = FullMusicSampler(catalog, random_start=False)
    a = sampler.next_task()
    sampler.record_execution(a, 25)
    sampler.record_execution(a, 25)
    row = sampler.coverage()["per_source"][a["sample"]["dataset"]]
    assert row["executed_tasks"] == 1 and row["upper_transitions"] == 2
    assert row["actual_seconds"] == row["covered_source_seconds"] == 1.
    b = sampler.next_task()
    with pytest.raises(ValueError, match="current issued task"):
        sampler.record_execution(a, 1)
    with pytest.raises(ValueError, match="exceeds"):
        sampler.record_execution(b, 301)


@pytest.mark.parametrize("field", ["split", "window_seconds", "draw_count", "orders", "cursor", "rng", "intervals", "active"])
def test_restore_rejects_corrupt_state_without_changing_existing_sampler(tmp_path, field):
    catalog = FullMusicCatalog(make_catalog_data(tmp_path))
    sampler = FullMusicSampler(catalog)
    task = sampler.next_task()
    sampler.record_execution(task, 25)
    state = sampler.state_dict()
    corrupt = copy.deepcopy(state)
    if field == "split": corrupt["split"] = "val"
    elif field == "window_seconds": corrupt[field] = 2.
    elif field == "draw_count": corrupt[field] = 99
    elif field == "orders": corrupt[field][SOURCES[0]] = [0,0]
    elif field == "cursor": corrupt["cursors"][SOURCES[0]] = -1
    elif field == "rng": corrupt[field] = {"broken": True}
    elif field == "intervals": corrupt[field] = {"Mine/missing": [[0,12]]}
    else: corrupt[field]["start_frame"] = 180
    with pytest.raises((ValueError, KeyError, TypeError)):
        sampler.load_state_dict(corrupt)
    assert sampler.state_dict() == state


@pytest.mark.parametrize("split", ["train", "val", "test"])
def test_reward_pair_window_offset_matches_original_motion_and_explicit_split(tmp_path, split):
    root = make_catalog_data(tmp_path)
    sample = FullMusicCatalog(root).samples[split][SOURCES[0]][0]
    offset = load_paired_activity(root, sample, split=split, music_start_frame=60)
    # q=t²，起点2s，在2.00→2.02的线性区间斜率为4+1/30。
    assert offset(612)["activity_rad_s"] == pytest.approx(4+1/30)
    assert offset.source["full_source_num_frames"] == 180
    assert offset.source["source_num_frames"] == 120
    assert offset.source["music_start_frame"] == 60 and offset.source["split"] == split
    if split != "train":
        with pytest.raises(ValueError):
            load_paired_activity(root, sample)


def test_environment_applies_same_offset_before_physics_and_keeps_music_end(tmp_path, monkeypatch):
    from tests.closedloop.dppo.test_env_adapter import adapter
    env, backend = adapter(tmp_path, mode="paused")
    env.config["stage9"]["bc_data_root"] = "/explicit/full-data"
    env.config["stage9"]["episode_seconds"] = 1.
    seen = {}
    def pair(*args, **kwargs):
        seen.update(kwargs)
        from tests.closedloop.dppo.test_data_learning import target_activity_fixture
        return target_activity_fixture
    monkeypatch.setattr("gem.closedloop.dppo.env_adapter.load_paired_activity", pair)
    original = backend.call
    def call(method, **payload):
        if method == "reset_episode":
            backend.tick = 0
            return backend.snapshot()
        return original(method, **payload)
    backend.call = call
    music = np.arange(180*35, dtype=float).reshape(180,35)
    sample = dict(dataset="Mine", row=dict(sample_id="x", split="val", num_frames=180))
    env.reset_task(sample, music, seed=42, music_start_frame=60)
    assert seen["music_start_frame"] == 60 and seen["split"] == "val"
    np.testing.assert_array_equal(env.music, music[60:])
    assert env.remaining_music() == 4.
    assert env.music_end_tick == 600+2400 and env.soft_end_tick == 600+600
    assert env.music_start_frame == 60 and env.full_music_num_frames == 180


@pytest.mark.parametrize("start", [-1, 179, True, 1.5])
def test_invalid_window_rejected_before_backend_reset(tmp_path, start):
    from tests.closedloop.dppo.test_env_adapter import adapter
    env, backend = adapter(tmp_path)
    with pytest.raises(ValueError):
        env.reset_task(dict(dataset="Mine", row=dict(sample_id="x")), np.zeros((180,35)), seed=42, music_start_frame=start)
    assert backend.calls == []
