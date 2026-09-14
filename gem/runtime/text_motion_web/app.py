"""GENMO 本机 HTTP 页面与 JSON 接口。

Flask 仅提供固定前端、模型注册、任务提交、历史查询和已完成任务的媒体文件。
视频使用条件响应支持 HTTP Range；请求限制为回环 Host 与同源 JSON，防止其他
网页借本机服务发起模型加载。接口不接受任意输出路径，也不提供通用文件读取。
"""

from flask import Flask, abort, jsonify, render_template, request, send_file
from werkzeug.exceptions import HTTPException

from .service import BusyError
from .storage import FIXED


def create_app(service):
    app = Flask(__name__, template_folder="templates", static_folder="static")
    app.config.update(
        MAX_CONTENT_LENGTH=64 * 1024, TRUSTED_HOSTS=["127.0.0.1", "localhost", "[::1]"]
    )

    @app.before_request
    def local_request():
        if request.method == "POST":
            origin = request.headers.get("Origin")
            if origin and origin != request.host_url.rstrip("/"):
                abort(403, "仅允许同源本地网页提交")
            if not request.is_json:
                abort(415, "接口需要 application/json")

    @app.errorhandler(Exception)
    def error(exc):
        if isinstance(exc, BusyError):
            code = 409
        elif isinstance(exc, (ValueError, FileNotFoundError)):
            code = 400
        elif isinstance(exc, KeyError):
            code = 404
        elif isinstance(exc, HTTPException):
            code = exc.code
        else:
            app.logger.exception("本地工作台接口错误")
            code = 500
        return jsonify(error=str(exc)), code

    @app.get("/")
    def index():
        return render_template("index.html")

    @app.get("/api/models")
    def models():
        return jsonify(**service.registry.snapshot(), fixed=FIXED)

    @app.post("/api/models")
    def register():
        payload = request.get_json()
        if not isinstance(payload, dict) or set(payload) != {"path"}:
            raise ValueError("模型注册仅接受 path 字段")
        return jsonify(model=service.registry.add(payload["path"])), 201

    @app.post("/api/jobs")
    def submit():
        return jsonify(service.submit(request.get_json())), 202

    @app.get("/api/jobs/<job_id>")
    def status(job_id):
        return jsonify(service.get(job_id))

    @app.get("/api/history")
    def history():
        return jsonify(service.history())

    @app.get("/api/jobs/<job_id>/video")
    def video(job_id):
        return send_file(
            service.media_path(job_id, "video.mp4"),
            mimetype="video/mp4",
            conditional=True,
            download_name=f"genmo-{job_id}.mp4",
        )

    @app.get("/api/jobs/<job_id>/thumbnail")
    def thumbnail(job_id):
        return send_file(
            service.media_path(job_id, "thumbnail.jpg"), mimetype="image/jpeg", conditional=True
        )

    @app.after_request
    def headers(response):
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Content-Security-Policy"] = (
            "default-src 'self'; img-src 'self'; media-src 'self'; object-src 'none'; frame-ancestors 'none'"
        )
        if request.path.startswith("/api/") and response.mimetype == "application/json":
            response.headers["Cache-Control"] = "no-store"
        return response

    return app
