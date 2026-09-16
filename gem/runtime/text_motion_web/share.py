"""文本动作工作台的匿名公网入口，复用已经运行的本机推理服务。

本模块只转发模型列表、生成任务、历史以及按任务 ID 获取的媒体，不加载模型或
创建第二个 GPU worker。本机 8766 端口继续负责 checkpoint 注册、单任务互斥、
动作生成、渲染及持久化；公网入口删除本地文件路径、权重指纹和内部错误详情，
禁止访客注册本地 checkpoint。共享页面明确说明历史对访客公开。

上游地址必须是回环 HTTP 服务，不使用环境中的 HTTP 代理，也不跟随重定向；
公网请求只能到达显式列出的路由。POST 校验公开站点 Origin、JSON 字段与参数，
不把访客的 Host、Cookie、Authorization 或代理头传入上游。视频流保留 Range、
ETag 和条件请求语义，支持浏览器拖动及 HEAD，避免一次性读入整段视频。
"""

from __future__ import annotations

import ipaddress
import json
import re
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import HTTPRedirectHandler, ProxyHandler, Request, build_opener

from flask import Flask, Response, abort, jsonify, render_template, request, send_from_directory
from werkzeug.exceptions import HTTPException

PACKAGE = Path(__file__).resolve().parent
JOB_ID = re.compile(r"[a-f0-9]{32}\Z")
MODEL_ID = re.compile(r"[a-f0-9]{20}\Z")
FIXED_KEYS = ("fps", "seed", "guidance_scale", "shape_mode", "width", "height", "postproc")
MEDIA_HEADERS = (
    "Content-Type", "Content-Length", "Content-Range", "Accept-Ranges", "ETag", "Last-Modified",
)
SAFE_ERRORS = {
    400: "输入参数无效，请检查模型、文本、帧数和 DDIM 步数。",
    403: "此操作只允许在维护者的本机入口执行。",
    404: "模型、任务或视频不存在。",
    409: "已有任务正在生成，请等待完成后再提交。",
    413: "请求内容过长，请缩短动作描述。",
    415: "请求需要使用 JSON 格式。",
    416: "视频请求范围无效，请刷新后重试。",
    502: "生成服务暂时不可用，请稍后重试。",
    503: "生成服务暂时不可用，请稍后重试。",
}


class NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


class UpstreamError(Exception):
    def __init__(self, status):
        self.status = status


class LoopbackBackend:
    def __init__(self, url):
        parsed = urlsplit(url)
        try:
            loopback = parsed.hostname == "localhost" or ipaddress.ip_address(parsed.hostname).is_loopback
        except ValueError:
            loopback = False
        if (
            parsed.scheme != "http" or not loopback or parsed.username or parsed.password
            or parsed.path not in ("", "/") or parsed.query or parsed.fragment
        ):
            raise ValueError("上游必须是无凭据、无路径的本机回环 HTTP 地址")
        self.url = url.rstrip("/")
        self.opener = build_opener(ProxyHandler({}), NoRedirect())

    def open(self, path, *, method="GET", payload=None, headers=None):
        outgoing = {"Accept": "application/json", **(headers or {})}
        data = None
        if payload is not None:
            data = json.dumps(payload, ensure_ascii=False).encode()
            outgoing.update({"Content-Type": "application/json", "Origin": self.url})
        try:
            return self.opener.open(
                Request(self.url + path, data=data, headers=outgoing, method=method), timeout=30
            )
        except HTTPError as exc:
            return exc
        except (URLError, TimeoutError, OSError) as exc:
            raise UpstreamError(502) from exc

    def json(self, path, *, payload=None):
        with self.open(path, method="POST" if payload is not None else "GET", payload=payload) as response:
            status = response.status
            if not 200 <= status < 300:
                raise UpstreamError(status if status in SAFE_ERRORS else 502)
            try:
                return json.load(response), status
            except (ValueError, OSError) as exc:
                raise UpstreamError(502) from exc


def public_model(model):
    result = {key: model.get(key) for key in ("id", "name", "global_step", "is_default")}
    contract = model.get("contract", {})
    result["contract"] = {
        key: contract.get(key)
        for key in ("schema_version", "max_text_len", "encoded_text_dim", "text_only")
    }
    return result


def public_job(job):
    result = {
        key: job[key] for key in (
            "id", "model_id", "prompt", "num_frames", "ddim_steps", "status", "created_at",
            "completed_at", "elapsed_seconds", "failed_stage",
        ) if key in job
    }
    result["model"] = public_model(job["model"])
    result["fixed"] = {key: job.get("fixed", {}).get(key) for key in FIXED_KEYS}
    if result["status"] == "done":
        result["video_url"] = f"/api/jobs/{result['id']}/video"
        result["thumbnail_url"] = f"/api/jobs/{result['id']}/thumbnail"
    if result["status"] == "failed":
        error = str(job.get("error", "")).lower()
        result["error"] = (
            "显存不足，请减少动作帧数后重试。" if "out of memory" in error
            else "本次生成失败，请重试；详细诊断已留给站点维护者。"
        )
    return result


def validate_payload(payload):
    if not isinstance(payload, dict) or set(payload) != {
        "model_id", "prompt", "num_frames", "ddim_steps",
    }:
        abort(400)
    if not isinstance(payload["model_id"], str) or not MODEL_ID.fullmatch(payload["model_id"]):
        abort(400)
    if not isinstance(payload["prompt"], str) or not payload["prompt"].strip():
        abort(400)
    if len(payload["prompt"]) > 4096:
        abort(413)
    for key, low, high in (("num_frames", 1, 900), ("ddim_steps", 2, 1000)):
        if type(payload[key]) is not int or not low <= payload[key] <= high:
            abort(400)
    return payload


def create_share_app(upstream_url, public_origin):
    parsed = urlsplit(public_origin)
    if (
        parsed.scheme not in ("http", "https") or not parsed.hostname
        or parsed.username or parsed.password or parsed.path not in ("", "/")
        or parsed.query or parsed.fragment
    ):
        raise ValueError("public_origin 必须是公开站点的完整 http(s) origin")
    public_origin = public_origin.rstrip("/")
    backend = LoopbackBackend(upstream_url)
    app = Flask(__name__, template_folder=str(PACKAGE / "templates"), static_folder=None)
    app.config.update(
        MAX_CONTENT_LENGTH=16 * 1024,
        TRUSTED_HOSTS=[parsed.hostname, "127.0.0.1", "localhost", "[::1]"],
    )

    @app.before_request
    def check_request():
        if request.query_string:
            abort(400)
        if request.method == "POST":
            origin = request.headers.get("Origin")
            if origin and origin != public_origin:
                abort(403)
            if not request.is_json:
                abort(415)

    @app.errorhandler(Exception)
    def error(exc):
        if isinstance(exc, UpstreamError):
            code = exc.status
        elif isinstance(exc, HTTPException):
            code = exc.code
        else:
            app.logger.exception("公网入口处理失败")
            code = 502
        return jsonify(error=SAFE_ERRORS.get(code, "请求无法处理，请刷新页面后重试。")), code

    @app.get("/")
    def index():
        return render_template("index.html", public_share=True)

    @app.get("/static/<filename>", endpoint="static")
    def static_file(filename):
        if filename not in ("app.js", "style.css"):
            abort(404)
        return send_from_directory(PACKAGE / "static", filename, conditional=True)

    @app.get("/api/models")
    def models():
        data, _ = backend.json("/api/models")
        return jsonify(
            models=[public_model(m) for m in data["models"]],
            fixed={key: data.get("fixed", {}).get(key) for key in FIXED_KEYS},
            scanning=data.get("scanning", False),
            error="模型列表暂时不可用。" if data.get("error") else None,
            capabilities={"register_models": False, "public_share": True},
        )

    @app.post("/api/models")
    def no_registration():
        abort(403)

    @app.post("/api/jobs")
    def submit():
        payload = validate_payload(request.get_json())
        data, status = backend.json("/api/jobs", payload=payload)
        return jsonify(public_job(data)), status

    @app.get("/api/history")
    def history():
        data, _ = backend.json("/api/history")
        return jsonify(
            jobs=[public_job(job) for job in data["jobs"]], active_id=data.get("active_id"),
            recovery_errors=["部分历史暂不可用，请联系站点维护者。"] if data.get("recovery_errors") else [],
        )

    @app.get("/api/jobs/<job_id>")
    def status(job_id):
        if not JOB_ID.fullmatch(job_id):
            abort(404)
        data, _ = backend.json(f"/api/jobs/{job_id}")
        return jsonify(public_job(data))

    @app.get("/api/jobs/<job_id>/<media>")
    def media_file(job_id, media):
        if not JOB_ID.fullmatch(job_id) or media not in ("video", "thumbnail"):
            abort(404)
        headers = {key: request.headers[key] for key in (
            "Range", "If-Range", "If-None-Match", "If-Modified-Since",
        ) if key in request.headers}
        response = backend.open(
            f"/api/jobs/{job_id}/{media}", method=request.method, headers=headers
        )
        outgoing = {key: response.headers[key] for key in MEDIA_HEADERS if key in response.headers}
        if response.status not in (200, 206, 304):
            code = response.status
            response.close()
            result = jsonify(error=SAFE_ERRORS.get(code, SAFE_ERRORS[502]))
            if "Content-Range" in outgoing:
                result.headers["Content-Range"] = outgoing["Content-Range"]
            return result, code if code in SAFE_ERRORS else 502
        if request.method == "HEAD" or response.status == 304:
            code = response.status
            response.close()
            return Response(status=code, headers=outgoing)

        def stream():
            try:
                while chunk := response.read(64 * 1024):
                    yield chunk
            finally:
                response.close()

        result = Response(stream(), status=response.status, headers=outgoing)
        result.call_on_close(response.close)
        return result

    @app.after_request
    def headers(response):
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Referrer-Policy"] = "same-origin"
        response.headers["Content-Security-Policy"] = (
            "default-src 'self'; img-src 'self'; media-src 'self'; "
            "object-src 'none'; frame-ancestors 'none'; base-uri 'self'; form-action 'self'"
        )
        response.headers["Cache-Control"] = "no-store"
        return response

    return app
