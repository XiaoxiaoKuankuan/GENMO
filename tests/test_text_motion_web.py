"""本地文本动作网页的接口、持久化与进程故障回归测试。

测试只在 pytest 分配的临时目录写入小型 checkpoint 和媒体夹具，不加载真实 GPU。
覆盖输入边界、文件变更、跨请求并发、阶段失败、进程重建、历史恢复与媒体路径约束；
真实动作质量与浏览器解码另由显式 GPU 验收确认，不以这些接口测试代替。
"""

import copy
import os
import queue
import time
from pathlib import Path

import pytest
import torch

from gem.runtime.text_motion_web import worker
from gem.runtime.text_motion_web.app import create_app
from gem.runtime.text_motion_web.models import ModelRegistry, inspect_checkpoint
from gem.runtime.text_motion_web.service import JobService
from gem.runtime.text_motion_web.storage import atomic_json


def fake_worker(commands, events):
    while True:
        job = commands.get()
        if job is None:
            return
        if job["prompt"] == "crash":
            return
        events.put(dict(id=job["id"], status="loading"))
        if job["prompt"] == "loading_fail":
            events.put(
                dict(id=job["id"], status="failed", failed_stage="loading", error="load failed")
            )
            continue
        events.put(dict(id=job["id"], status="generating"))
        if job["prompt"] == "slow":
            time.sleep(0.5)
        events.put(dict(id=job["id"], status="rendering"))
        if job["prompt"] == "render_fail":
            events.put(
                dict(id=job["id"], status="failed", failed_stage="rendering", error="no video")
            )
            continue
        output = Path(job["task_dir"]) / "media"
        output.mkdir(parents=True)
        (output / "video.mp4").write_bytes(b"video-range-fixture")
        (output / "thumbnail.jpg").write_bytes(b"thumbnail")
        events.put(dict(id=job["id"], status="done", output_dir=str(output)))


@pytest.fixture
def web(tmp_path):
    registry = ModelRegistry(
        tmp_path,
        roots=[],
        inspector=lambda _: {"global_step": 190000, "contract": {"max_text_len": 150}},
    )
    ckpt = tmp_path / "model.ckpt"
    ckpt.write_bytes(b"fixture")
    model = registry.add(str(ckpt))
    service = JobService(tmp_path, registry=registry, worker_target=fake_worker, discover=False)
    app = create_app(service)
    app.config["TESTING"] = True
    try:
        yield service, app.test_client(), model
    finally:
        service.close()


def payload(model, **changes):
    return dict(model_id=model["id"], prompt="walk", num_frames=120, ddim_steps=50, **{}) | changes


def wait_job(service, job_id):
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        job = service.get(job_id)
        if job["status"] in {"done", "failed"}:
            return job
        time.sleep(0.03)
    raise AssertionError("fixture worker did not complete")


@pytest.mark.parametrize(
    "change",
    [
        {"prompt": " \n"},
        {"prompt": 4},
        {"num_frames": 0},
        {"num_frames": 901},
        {"num_frames": 1.5},
        {"num_frames": True},
        {"ddim_steps": 1},
        {"ddim_steps": 1001},
        {"ddim_steps": "50"},
        {"model_id": "missing"},
        {"model_id": []},
        {"fps": 60},
    ],
)
def test_invalid_request(web, change):
    _, client, model = web
    assert client.post("/api/jobs", json=payload(model, **change)).status_code == 400


def test_concurrency_frozen_prompt_and_range(web):
    service, client, model = web
    response = client.post("/api/jobs", json=payload(model, prompt="slow"))
    assert response.status_code == 202
    assert client.post("/api/jobs", json=payload(model)).status_code == 409
    job = wait_job(service, response.json["id"])
    assert job["status"] == "done"
    assert job["prompt"] == "slow" and job["fixed"]["fps"] == 30
    media = client.get(job["video_url"], headers={"Range": "bytes=0-4"})
    assert media.status_code == 206 and media.data == b"video"
    assert media.headers["Content-Range"].startswith("bytes 0-4/")
    media.close()
    full = client.get(job["video_url"])
    assert full.status_code == 200
    full.close()
    assert client.get("/api/jobs/missing/video").status_code == 404
    assert client.get("/api/jobs/../../etc/passwd/video").status_code == 404


@pytest.mark.parametrize(
    "prompt,stage", [("loading_fail", "loading"), ("render_fail", "rendering"), ("crash", "queued")]
)
def test_failure_preserves_success_and_allows_retry(web, prompt, stage):
    service, client, model = web
    previous = wait_job(service, service.submit(payload(model))["id"])
    failed = wait_job(service, service.submit(payload(model, prompt=prompt))["id"])
    assert failed["status"] == "failed" and failed["failed_stage"] == stage
    history = client.get("/api/history").json
    assert history["active_id"] is None
    assert any(job["id"] == previous["id"] and job["status"] == "done" for job in history["jobs"])
    assert client.get(f"/api/jobs/{failed['id']}/video").status_code == 404
    assert wait_job(service, service.submit(payload(model))["id"])["status"] == "done"


def test_history_recovers_interrupted_and_done(web):
    service, _, model = web
    done = wait_job(service, service.submit(payload(model, prompt="  Keep my text\n "))["id"])
    assert done["prompt"] == "  Keep my text\n "
    pending = copy.deepcopy(done)
    pending.update(id="a" * 32, status="rendering")
    atomic_json(service.root / "tasks" / pending["id"] / "task.json", pending)
    service.close()
    restored = JobService(service.root, registry=service.registry, discover=False)
    try:
        assert restored.get(done["id"])["status"] == "done"
        assert restored.get(pending["id"])["status"] == "failed"
        assert restored.process is None and restored.active_id is None
    finally:
        restored.close()


def test_local_origin_json_and_media_containment(web, tmp_path):
    service, client, model = web
    assert (
        client.post(
            "/api/jobs", json=payload(model), headers={"Origin": "https://outside.example"}
        ).status_code
        == 403
    )
    assert client.post("/api/jobs", data="hello").status_code == 415
    assert client.get("/api/models", headers={"Host": "outside.example"}).status_code == 400
    assert client.post("/api/models", json={"path": "/missing/model.ckpt"}).status_code == 400
    assert (
        client.post("/api/models", json={"path": model["path"]}).json["model"]["id"] == model["id"]
    )
    job = wait_job(service, service.submit(payload(model))["id"])
    service.jobs[job["id"]]["output_dir"] = str(tmp_path.parent)
    assert client.get(job["video_url"]).status_code == 404


def test_registry_invalidates_changed_file_and_deduplicates(tmp_path):
    calls = []

    def inspect(path):
        calls.append(path)
        if path.read_text() == "bad":
            raise ValueError("invalid checkpoint")
        return {"global_step": 1, "contract": {"max_text_len": int(path.read_text())}}

    registry = ModelRegistry(tmp_path, roots=[], inspector=inspect)
    path = tmp_path / "small.ckpt"
    path.write_text("50")
    first = registry.add(str(path))
    link = tmp_path / "link.ckpt"
    link.symlink_to(path)
    assert registry.add(str(link))["id"] == first["id"]
    assert len(calls) == 1
    path.write_text("150")
    assert registry.get(first["id"])["contract"]["max_text_len"] == 150
    path.write_text("bad")
    # 某些文件系统连续同长度写入可能处于同一时钟 tick，显式推进文件修改时间。
    stat = path.stat()
    os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000_000))
    with pytest.raises(ValueError):
        registry.get(first["id"])
    assert registry.snapshot()["models"] == []


def test_real_checkpoint_content_contract_filter(tmp_path):
    prefix = "pipeline.denoiser3d.denoiser."
    state = {
        prefix + "embed_text.weight": torch.zeros(4, 1024),
        prefix + "text_encoder_layers.gate_cross_attn": torch.zeros(1),
        prefix + "final_layer.fc2.weight": torch.zeros(151, 4),
        prefix + "add_cond_linear.weight": torch.zeros(4, 155),
    }
    path = tmp_path / "contract.ckpt"
    checkpoint = {"state_dict": state, "global_step": 123}
    torch.save(checkpoint, path)
    assert inspect_checkpoint(path)["contract"]["max_text_len"] == 50
    checkpoint["genmo_text_contract"] = dict(
        schema_version=1, max_text_len=150, encoded_text_dim=1024, text_only=True
    )
    torch.save(checkpoint, path)
    assert inspect_checkpoint(path)["contract"]["max_text_len"] == 150
    checkpoint["hyper_parameters"] = {"network": {"regression_only": True}}
    torch.save(checkpoint, path)
    with pytest.raises(ValueError, match="回归"):
        inspect_checkpoint(path)
    del checkpoint["hyper_parameters"]
    state[prefix + "final_layer.fc2.weight"] = torch.zeros(30, 4)
    torch.save(checkpoint, path)
    with pytest.raises(ValueError, match="BUMI"):
        inspect_checkpoint(path)
    del state[prefix + "embed_text.weight"]
    torch.save(checkpoint, path)
    with pytest.raises(RuntimeError, match="text-conditioned"):
        inspect_checkpoint(path)


def test_worker_reuses_engine_updates_ddim_and_switches_contract(tmp_path, monkeypatch):
    import gem.runtime.resident_text_motion as resident

    calls, created = [], []

    class Engine:
        def __init__(self, **kwargs):
            self.max_text_len = 150 if "new" in kwargs["ckpt_path"] else 50
            self.path = kwargs["ckpt_path"]
            created.append(kwargs)

        def initialize(self):
            calls.append(("initialize", self.max_text_len))

        def set_ddim_steps(self, steps):
            calls.append(("ddim", steps))

        def generate(self, request):
            return {"ok": True, "output_dir": request["output_root"], "timing": {}}

        def close(self):
            calls.append(("close", self.max_text_len))

    monkeypatch.setattr(resident, "ResidentTextMotionEngine", Engine)
    monkeypatch.setattr(worker, "follow_parent", lambda: None)
    monkeypatch.setattr(worker, "run_renderer", lambda *_: {"fully_decoded": True})
    commands, events = queue.Queue(), queue.Queue()
    for index, (name, steps) in enumerate([("new", 20), ("new", 50), ("old", 50)]):
        path = tmp_path / name
        if not path.exists():
            path.write_bytes(b"fixture")
        commands.put(
            dict(
                id=str(index),
                model={"path": str(path), "fingerprint": worker.fingerprint(path)},
                prompt="walk",
                num_frames=120,
                ddim_steps=steps,
                task_dir=str(tmp_path / str(index)),
            )
        )
    commands.put(None)
    worker.worker_main(commands, events)
    assert len(created) == 2
    assert calls == [
        ("initialize", 150),
        ("ddim", 50),
        ("close", 150),
        ("initialize", 50),
        ("close", 50),
    ]
    assert [events.get()["status"] for _ in range(events.qsize())].count("done") == 3


def test_renderer_warning_without_video_is_failure(tmp_path, monkeypatch):
    from scripts.demo import demo_smpl_text

    monkeypatch.setattr(worker, "follow_parent", lambda: None)
    monkeypatch.setattr(worker, "check_motion", lambda *_: None)
    monkeypatch.setattr(torch, "load", lambda *_, **__: {"body_params_global": {}})
    monkeypatch.setattr(demo_smpl_text, "render_global_video", lambda *_: None)
    with pytest.raises(RuntimeError, match="未输出有效视频"):
        worker.render_job(tmp_path, 120)
