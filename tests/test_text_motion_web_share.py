"""匿名共享入口的真实 HTTP 转发与访问边界测试。

在 pytest 临时目录写入一个小型媒体夹具，并启动仅监听回环地址的假推理后端；
共享入口通过真实 urllib HTTP 请求访问它，验证跨入口复用、参数冻结、并发拒绝、
路径/内部错误脱敏、同源约束、禁止本地模型注册及视频 Range/HEAD/条件请求。
不加载模型、不启动 GPU、不写正式历史。服务器在每个夹具退出时关闭，外部执行器
负责清理统一测试临时根目录；这些测试证明接口契约，不代表生成质量验收。
"""

from __future__ import annotations

import copy
import threading

import pytest
from flask import Flask, jsonify, redirect, request, send_file
from werkzeug.serving import make_server

from gem.runtime.text_motion_web.share import LoopbackBackend, create_share_app

ORIGIN = "https://genmo.example.test"
MODEL_ID = "a" * 20
JOB_ID = "b" * 32
MODEL = {
    "id": MODEL_ID, "name": "MotionMillion / s210000.ckpt", "global_step": 210000,
    "is_default": False, "path": "/home/private/checkpoint.ckpt", "fingerprint": [12, 34],
    "contract": {"schema_version": 1, "max_text_len": 150, "encoded_text_dim": 1024, "text_only": True},
}
PAYLOAD = {"model_id": MODEL_ID, "prompt": "  A person walks forward.  ", "num_frames": 120, "ddim_steps": 50}
JOB = {
    **PAYLOAD, "id": JOB_ID, "model": MODEL, "status": "done", "elapsed_seconds": 3.5,
    "created_at": "2026-09-16T01:00:00+00:00", "task_dir": "/home/private/task",
    "output_dir": "/home/private/result", "metadata_path": "/home/private/metadata.json",
    "fixed": {"fps": 30, "seed": 42, "t5_model": "/home/private/t5"},
}


@pytest.fixture
def gateway(tmp_path):
    backend = Flask("fake-share-backend")
    state = {"busy": False, "payload": None, "headers": None, "mode": "normal"}
    media = tmp_path / "video.mp4"
    media.write_bytes(bytes(range(256)) * 8)

    @backend.get("/api/models")
    def models():
        if state["mode"] == "redirect":
            return redirect("http://127.0.0.1:1/private")
        if state["mode"] == "error":
            return jsonify(error="/home/private/internal traceback"), 500
        return jsonify(models=[MODEL], fixed=JOB["fixed"], scanning=False, error=None)

    @backend.get("/api/history")
    def history():
        job = copy.deepcopy(JOB)
        if state["mode"] == "failed":
            job.update(status="failed", failed_stage="rendering", error="/home/private/error.log")
        return jsonify(jobs=[job], active_id=JOB_ID if state["busy"] else None, recovery_errors=[])

    @backend.get("/api/jobs/<job_id>")
    def get_job(job_id):
        return jsonify(JOB) if job_id == JOB_ID else (jsonify(error="/home/private/missing"), 404)

    @backend.post("/api/jobs")
    def submit():
        state["headers"] = dict(request.headers)
        if state["busy"]:
            return jsonify(error="busy"), 409
        state["payload"] = request.get_json()
        state["busy"] = True
        return jsonify({**JOB, **state["payload"], "status": "queued"}), 202

    @backend.get("/api/jobs/<job_id>/<kind>")
    def get_media(job_id, kind):
        if job_id != JOB_ID or kind not in ("video", "thumbnail"):
            return jsonify(error="missing"), 404
        return send_file(media, conditional=True, mimetype="video/mp4")

    server = make_server("127.0.0.1", 0, backend, threaded=True)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    url = f"http://127.0.0.1:{server.server_port}"
    app = create_share_app(url, ORIGIN)
    try:
        yield app.test_client(), state, url
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def test_public_page_and_models_hide_management_and_paths(gateway):
    client, _, _ = gateway
    page = client.get("/", base_url=ORIGIN)
    assert page.status_code == 200
    assert "公开共享" in page.text and "对所有访客可见" in page.text
    assert 'id="add-model"' not in page.text and "/home/" not in page.text
    response = client.get("/api/models", base_url=ORIGIN)
    assert response.status_code == 200
    assert response.json["models"][0]["contract"]["max_text_len"] == 150
    assert response.json["capabilities"]["register_models"] is False
    assert "fingerprint" not in response.text and "/home/private" not in response.text
    history = client.get("/api/history", base_url=ORIGIN)
    assert history.json["jobs"][0]["prompt"] == PAYLOAD["prompt"]
    assert history.json["jobs"][0]["video_url"] == f"/api/jobs/{JOB_ID}/video"
    assert "/home/private" not in history.text and "output_dir" not in history.text


def test_submission_uses_existing_backend_and_does_not_forward_credentials(gateway):
    client, state, upstream = gateway
    response = client.post("/api/jobs", base_url=ORIGIN, json=PAYLOAD, headers={
        "Origin": ORIGIN, "Authorization": "Bearer external-secret", "Cookie": "private=abc",
        "X-Forwarded-Host": "attacker.example", "X-Forwarded-For": "1.2.3.4",
    })
    assert response.status_code == 202 and response.json["status"] == "queued"
    assert state["payload"] == PAYLOAD
    assert state["headers"]["Origin"] == upstream
    assert "Authorization" not in state["headers"] and "Cookie" not in state["headers"]
    assert "X-Forwarded-Host" not in state["headers"] and "X-Forwarded-For" not in state["headers"]
    assert "/home/private" not in response.text
    assert client.post("/api/jobs", base_url=ORIGIN, json=PAYLOAD).status_code == 409
    assert client.get("/api/history", base_url=ORIGIN).json["active_id"] == JOB_ID


@pytest.mark.parametrize("path", ["/api/models", "/etc/passwd", "/api/jobs/../models", f"/api/jobs/{JOB_ID}/motion.npz"])
def test_admin_and_unlisted_paths_are_not_forwarded(gateway, path):
    client, state, _ = gateway
    response = client.post(path, base_url=ORIGIN, json={"path": "/home/private/new.ckpt"})
    assert response.status_code in (403, 404, 405)
    assert state["payload"] is None


@pytest.mark.parametrize("patch", [
    {"num_frames": 0}, {"num_frames": 901}, {"num_frames": True}, {"ddim_steps": 1},
    {"ddim_steps": 1001}, {"prompt": "  "}, {"model_id": "../private"}, {"output_dir": "/tmp/evil"},
])
def test_invalid_generation_parameters_never_reach_backend(gateway, patch):
    client, state, _ = gateway
    response = client.post("/api/jobs", base_url=ORIGIN, json={**PAYLOAD, **patch})
    assert response.status_code == 400 and state["payload"] is None


def test_origin_content_type_size_host_and_query_boundaries(gateway):
    client, state, _ = gateway
    assert client.post("/api/jobs", base_url=ORIGIN, json=PAYLOAD, headers={"Origin": "https://other.example"}).status_code == 403
    assert client.post("/api/jobs", base_url=ORIGIN, data="text").status_code == 415
    assert client.post("/api/jobs", base_url=ORIGIN, json={**PAYLOAD, "prompt": "x" * 4097}).status_code == 413
    assert client.post("/api/jobs", base_url=ORIGIN, json={**PAYLOAD, "prompt": "x" * 20000}).status_code == 413
    assert client.get("/api/models", base_url="https://other.example").status_code == 400
    assert client.get("/api/models?url=http://other.example", base_url=ORIGIN).status_code == 400
    assert state["payload"] is None


def test_video_range_head_and_conditional_get(gateway):
    client, _, _ = gateway
    url = f"/api/jobs/{JOB_ID}/video"
    response = client.get(url, base_url=ORIGIN, headers={"Range": "bytes=128-255"})
    assert response.status_code == 206 and response.data == bytes(range(128, 256))
    assert response.headers["Content-Range"] == "bytes 128-255/2048"
    assert response.headers["Content-Length"] == "128"
    assert response.headers["Accept-Ranges"] == "bytes"
    head = client.head(url, base_url=ORIGIN)
    assert head.status_code == 200 and head.data == b"" and head.headers["Content-Length"] == "2048"
    cached = client.get(url, base_url=ORIGIN, headers={"If-None-Match": head.headers["ETag"]})
    assert cached.status_code == 304 and not cached.data
    invalid = client.get(url, base_url=ORIGIN, headers={"Range": "bytes=3000-4000"})
    assert invalid.status_code == 416 and invalid.headers["Content-Range"] == "bytes */2048"
    assert client.get(f"/api/jobs/{JOB_ID}/thumbnail", base_url=ORIGIN).status_code == 200


def test_backend_failures_and_redirects_do_not_expose_internal_details(gateway):
    client, state, _ = gateway
    state["mode"] = "failed"
    response = client.get("/api/history", base_url=ORIGIN)
    assert response.json["jobs"][0]["status"] == "failed"
    assert "/home/private" not in response.text
    for mode in ("error", "redirect"):
        state["mode"] = mode
        response = client.get("/api/models", base_url=ORIGIN)
        assert response.status_code == 502 and "/home/private" not in response.text


@pytest.mark.parametrize("upstream", [
    "https://127.0.0.1:8766", "http://example.com", "http://user:pass@127.0.0.1", "http://127.0.0.1/private",
])
def test_upstream_must_be_loopback_without_credentials_or_paths(upstream):
    with pytest.raises(ValueError):
        LoopbackBackend(upstream)
