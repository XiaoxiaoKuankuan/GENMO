"""宏迭代、阶段与完整优化步的原子恢复记录。

自有PyTorch断点保存网络/优化器/调度器及各rank RNG，旁路JSON绑定配置、文件摘要
与阶段进度。先fsync断点再原子更新latest指针；中断只回退到最后完整优化步。
物理世界不在此序列化，恢复采集必须重新建世界并重做未完成任务，不能拼接半条Episode。
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import random
import tempfile
import numpy as np
import torch

from .io import atomic_json, cpu_copy, digest, file_sha256, sync_directory


def capture_rng(device):
    return dict(torch=torch.get_rng_state(), numpy=np.random.get_state(), python=random.getstate(),
                cuda=torch.cuda.get_rng_state(device) if torch.device(device).type == "cuda" else None)


def restore_rng(state, device):
    torch.set_rng_state(state["torch"]); np.random.set_state(state["numpy"]); random.setstate(state["python"])
    if state["cuda"] is not None:
        torch.cuda.set_rng_state(state["cuda"], device)


class Checkpoints:
    def __init__(self, root, identity):
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.identity = identity

    def save(self, name, payload, *, progress):
        if Path(name).name != name:
            raise ValueError("Checkpoint name must be local")
        target = self.root / (name+".pt")
        fd, temporary = tempfile.mkstemp(prefix="."+name, dir=self.root)
        try:
            with os.fdopen(fd, "wb") as stream:
                torch.save(dict(schema="genmo.awr.checkpoint.v1", identity=self.identity,
                                progress=progress, payload=cpu_copy(payload)), stream)
                stream.flush(); os.fsync(stream.fileno())
            sha = file_sha256(temporary)
            if target.exists():
                raise FileExistsError("Never overwrite a published AWR checkpoint")
            os.rename(temporary, target)
            sync_directory(self.root)
            reference = dict(file=target.name, sha256=sha, identity=self.identity, progress=progress)
            atomic_json(self.root / (name+".json"), reference)
            atomic_json(self.root / "latest.json", reference)
            return reference
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)

    def load(self, reference=None):
        ref = json.loads((self.root / "latest.json").read_text()) if reference is None else reference
        path = (self.root / ref["file"]).resolve()
        if path.parent != self.root or digest(ref["identity"]) != digest(self.identity):
            raise ValueError("Checkpoint path or run identity differs")
        if file_sha256(path) != ref["sha256"]:
            raise ValueError("Checkpoint checksum mismatch")
        state = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
        if state["identity"] != self.identity or state["progress"] != ref["progress"]:
            raise ValueError("Checkpoint metadata binding changed")
        return state


def stop_requested(run_dir):
    return (Path(run_dir) / "STOP").is_file()
