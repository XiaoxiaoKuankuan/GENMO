"""检查闭环源码指纹覆盖 untracked、有限 URDF 依赖和运行结束差异。

全部写入 pytest 临时目录，用极小的两个工作树和 Git 输出替身测试；不遍历真实仓库，
不加载模型或 GPU。另用已有 preflight 文本检查后台 GPU 进程仅作负载观测，而不会进入
静态计算身份；不会采集可能含凭据的进程完整命令行或环境变量。
"""

from __future__ import annotations

import copy
from pathlib import Path

import pytest

from gem.closedloop import baseline_provenance as provenance


@pytest.fixture
def workspace(tmp_path, monkeypatch):
    paths = {}
    for repository in ("genmo_repo", "gmt_repo"):
        root = tmp_path / repository
        root.mkdir()
        paths[repository] = str(root)
    genmo, gmt = Path(paths["genmo_repo"]), Path(paths["gmt_repo"])
    (genmo / "actor.py").write_text("existing Actor\n")
    (genmo / "new.py").write_text("new untracked Stage8 module\n")
    (gmt / "asset.py").write_text("fixed asset importer\n")
    (gmt / "robot.urdf").write_text('<robot><link><collision><mesh filename="meshes/contact.stl"/></collision>'
                                    '<visual><mesh filename="meshes/contact.stl"/></visual></link></robot>')
    (gmt / "meshes").mkdir()
    (gmt / "meshes/contact.stl").write_bytes(b"test mesh")
    inventory = {"genmo_repo": {"stage8": ("new.py",), "reused": ("actor.py",)},
                 "gmt_repo": {"asset_import": ("robot.urdf", "asset.py")}}
    monkeypatch.setattr(provenance, "SOURCE_FILES", inventory)
    monkeypatch.setattr(provenance, "ROBOT_URDF", "robot.urdf")

    def git(root, *args, binary=False):
        assert args[:2] == ("ls-files", "-z") and binary
        return b"actor.py\0" if root == genmo else b"robot.urdf\0asset.py\0"

    monkeypatch.setattr(provenance, "_git", git)
    # 任何全树递归查找都会令本测试失败，防止实现悄然扫描大数据仓库。
    monkeypatch.setattr(Path, "rglob", lambda *args, **kwargs: pytest.fail("provenance must not recurse"))
    state = {name: {"path": str(Path(path).resolve()), "branch": "feature/stage8", "head": "abc123",
                    "status": "?? new.py" if name == "genmo_repo" else " M asset.py"}
             for name, path in paths.items()}
    return paths, state


def test_explicit_inventory_includes_untracked_and_deduplicates_meshes(workspace):
    paths, state = workspace
    report = provenance.collect_source_provenance(paths, repository_state=state)
    assert report["file_count"] == 5
    assert report["untracked_file_count"] == 2
    assert len(report["source_manifest_sha256"]) == 64
    assert all(Path(row["path"]).is_absolute() for row in report["files"])
    assert all(row["sha256"] and row["bytes"] > 0 for row in report["files"])
    assert all(repo["dirty"] and repo["head"] == "abc123" for repo in report["repositories"].values())
    assert len([row for row in report["files"] if row["category"] == "asset_mesh"]) == 1
    assert provenance.verify_source_provenance(report)["unchanged"]


def test_untracked_source_edit_changes_manifest_even_with_identical_git_head(workspace):
    paths, state = workspace
    first = provenance.collect_source_provenance(paths, repository_state=state)
    (Path(paths["genmo_repo"]) / "new.py").write_text("different untracked code\n")
    second = provenance.collect_source_provenance(paths, repository_state=state)
    assert first["repositories"] == second["repositories"]
    assert first["source_manifest_sha256"] != second["source_manifest_sha256"]
    verified = provenance.verify_source_provenance(first)
    assert not verified["unchanged"]
    assert len(verified["changed_files"]) == 1
    assert verified["changed_files"][0]["path"].endswith("new.py")


def test_touch_without_content_change_does_not_invalidate_source(workspace):
    paths, state = workspace
    report = provenance.collect_source_provenance(paths, repository_state=state)
    (Path(paths["genmo_repo"]) / "new.py").touch()
    assert provenance.verify_source_provenance(report)["unchanged"]


def test_missing_core_file_fails_capture_and_is_reported_at_verification(workspace):
    paths, state = workspace
    report = provenance.collect_source_provenance(paths, repository_state=state)
    (Path(paths["genmo_repo"]) / "actor.py").unlink()
    with pytest.raises(FileNotFoundError, match="source is missing"):
        provenance.collect_source_provenance(paths, repository_state=state)
    assert not provenance.verify_source_provenance(report)["unchanged"]


def test_cross_checkout_symlink_is_rejected(workspace, tmp_path):
    paths, state = workspace
    external = tmp_path / "old_checkout_actor.py"
    external.write_text("wrong import\n")
    actor = Path(paths["genmo_repo"]) / "actor.py"
    actor.unlink()
    actor.symlink_to(external)
    with pytest.raises(ValueError, match="outside"):
        provenance.collect_source_provenance(paths, repository_state=state)


def test_external_or_remote_mesh_is_rejected(workspace):
    paths, state = workspace
    urdf = Path(paths["gmt_repo"]) / "robot.urdf"
    urdf.write_text('<robot><mesh filename="https://example.invalid/robot.stl"/></robot>')
    with pytest.raises(ValueError, match="explicit local"):
        provenance.collect_source_provenance(paths, repository_state=state)


def test_reused_git_state_must_belong_to_the_same_worktree(workspace):
    paths, state = workspace
    changed = copy.deepcopy(state)
    changed["genmo_repo"]["path"] = paths["gmt_repo"]
    with pytest.raises(ValueError, match="repository path mismatch"):
        provenance.collect_source_provenance(paths, repository_state=changed)


def test_compute_context_reports_processes_as_observed_load_only():
    report = provenance.collect_compute_context({
        "identity": {"uuid": "GPU-selected"}, "free_memory_mib": 14000,
        "existing_compute_processes": "GPU-other, 10, other-service, 7000\n"
                                      "GPU-selected, 20, text-service, 11000\n",
    })
    assert report["identity"]["cpu"]["logical_cpu_count"] > 0
    assert "background_gpu_processes" not in report["identity"]
    assert report["observed"]["background_gpu_processes"] == [
        {"gpu_uuid": "GPU-selected", "pid": 20, "process_name": "text-service"}]
    assert report["observed"]["background_gpu_memory"][0]["used_memory_mib"] == 11000
    assert report["observed"]["free_gpu_memory_mib"] == 14000
    assert "instantaneous contention" in report["timing_boundary"]


def test_compute_context_requires_no_gpu_query_or_model_import(monkeypatch):
    monkeypatch.setattr(provenance.subprocess, "run", lambda *args, **kwargs: pytest.fail("no subprocess needed"))
    result = provenance.collect_compute_context()
    assert result["observed"]["background_gpu_processes"] == []
