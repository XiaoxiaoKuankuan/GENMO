"""正式联合训练的文件式协作暂停，不使用会丢失优化状态的进程信号。

操作者在当前输出目录原子写入 ``pause-request.json``，内容必须包含非空 ``reason``
和与该目录一致的绝对 ``output``。rank0每十个已完成的optimizer步读取一次；请求位
随训练原本已有的非有限检查张量归约同步，未请求暂停时不增加DDP通信。所有rank在相同
步数保存原有完整checkpoint，关闭预取并核验冻结资产后，才发布 ``paused.json``。
这里仅负责请求身份和暂停凭据，不修改训练Loss、数据/RNG、参数或checkpoint格式。
无效/半写入的请求只诊断且继续训练；此入口仅对加载该代码的新进程生效，不能让旧进程
通过修改磁盘源码获得在线保存能力。恢复使用原CLI的 --resume，并指定新的输出目录。
"""

import json
import os
from pathlib import Path


class PauseControl:
    """rank0验证外部请求；全rank共享的暂停决定由现有collective传回。"""

    def __init__(self, output, rank, check_every=10):
        self.output = Path(output).resolve()
        self.rank = rank
        self.check_every = check_every
        if check_every < 1:
            raise ValueError("暂停检查间隔必须为正整数")
        self.request = None
        self.requested = False
        self.last_error = None
        self._seen_signature = None

    def local_flag(self, step):
        if self.rank != 0 or step % self.check_every:
            return 0
        if self.request is not None:
            return 1
        path = self.output / "pause-request.json"
        try:
            stat = path.stat()
        except FileNotFoundError:
            return 0
        except OSError as error:
            self.last_error = str(error)
            print(json.dumps({"pause_request_rejected": str(path), "reason": self.last_error}, ensure_ascii=False), flush=True)
            return 0
        signature = (stat.st_ino, stat.st_mtime_ns, stat.st_size)
        if signature == self._seen_signature:
            return 0
        self._seen_signature = signature
        try:
            if stat.st_size > 65536:
                raise ValueError("暂停请求过大")
            value = json.loads(path.read_text())
            if not isinstance(value, dict):
                raise ValueError("暂停请求必须为JSON对象")
            if not isinstance(value.get("reason"), str) or not value["reason"].strip():
                raise ValueError("暂停请求缺少非空reason")
            target = value.get("output")
            if not isinstance(target, str) or not Path(target).is_absolute() or Path(target).resolve() != self.output:
                raise ValueError("暂停请求output必须精确指向当前run绝对目录")
            self.request = {"reason": value["reason"].strip(), "output": str(self.output)}
            self.last_error = None
        except (OSError, ValueError, TypeError) as error:
            self.last_error = str(error)
            print(json.dumps({"pause_request_rejected": str(path), "reason": self.last_error}, ensure_ascii=False), flush=True)
            return 0
        return 1

    def accept_global_flag(self, flag):
        self.requested = bool(flag)

    def publish(self, step, checkpoint, world_size):
        """只在所有rank已完成完整保存和退出检查后由rank0发布，不覆盖旧凭据。"""
        if self.rank != 0 or not self.requested or self.request is None:
            raise RuntimeError("只有收到合法请求的rank0可以发布暂停凭据")
        checkpoint = Path(checkpoint).resolve()
        if checkpoint.parent != self.output or not checkpoint.is_file():
            raise ValueError("暂停凭据必须引用当前run已保存的完整checkpoint")
        value = {"status": "paused", "global_step": step, "checkpoint": str(checkpoint),
                 "world_size": world_size, "request": self.request}
        path = self.output / "paused.json"
        temporary = self.output / f".paused-{os.getpid()}.partial"
        created = False
        try:
            with temporary.open("x") as handle:
                created = True
                json.dump(value, handle, ensure_ascii=False, indent=2)
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.link(temporary, path)
            directory_fd = os.open(self.output, os.O_RDONLY | os.O_DIRECTORY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
        finally:
            if created:
                temporary.unlink(missing_ok=True)
        return value
