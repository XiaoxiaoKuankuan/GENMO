"""AWR同步数据并行的全局分母与SUM梯度通信。

各rank对本地真实样本的损失除以同一个全局样本数，累积完微批后执行一次分桶SUM，
因此不再除world_size。空rank与尾批仍参与同一通信顺序；全局未使用的参数保留
grad=None，避免AdamW对无监督参数产生伪更新。CPU/Gloo和CUDA/NCCL共用算术。
"""
from __future__ import annotations

import os
import torch
import torch.distributed as dist


class Distributed:
    def __init__(self, device="cpu", tensor_group=None):
        self.device = torch.device(device)
        self.rank = dist.get_rank() if dist.is_initialized() else 0
        self.world_size = dist.get_world_size() if dist.is_initialized() else 1
        self.tensor_group = tensor_group

    def gather(self, value):
        if self.world_size == 1:
            return [value]
        result = [None] * self.world_size
        dist.all_gather_object(result, value)
        return result

    def broadcast(self, value):
        if self.world_size == 1:
            return value
        values = [value if self.rank == 0 else None]
        dist.broadcast_object_list(values, src=0)
        return values[0]

    def local(self, function):
        value, error = None, None
        try:
            value = function()
        except Exception as exc:
            error = f"rank {self.rank}: {type(exc).__name__}: {exc}"
        errors = [e for e in self.gather(error) if e]
        if errors:
            raise RuntimeError("AWR synchronized failure: " + " | ".join(errors))
        return value

    def root(self, function):
        value = self.local(function if self.rank == 0 else lambda: None)
        return self.broadcast(value)

    def sum(self, value):
        tensor = torch.as_tensor(value, device=self.device).clone()
        if self.world_size > 1:
            dist.all_reduce(tensor, group=self.tensor_group)
        return tensor

    def sync_model(self, model):
        if self.world_size > 1:
            for value in model.state_dict().values():
                dist.broadcast(value, src=0, group=self.tensor_group)

    def gradients(self, model, bucket_elements=2**21):
        parameters = [p for p in model.parameters() if p.requires_grad]
        present = self.sum(torch.tensor([p.grad is not None for p in parameters],
                                       dtype=torch.int32, device=self.device))
        bucket, count = [], 0

        def flush():
            if not bucket:
                return
            flat = torch.cat([p.grad.reshape(-1) for p in bucket])
            if self.world_size > 1:
                dist.all_reduce(flat, group=self.tensor_group)
            offset = 0
            for p in bucket:
                p.grad.copy_(flat[offset:offset+p.numel()].view_as(p))
                offset += p.numel()
            bucket.clear()

        for active, parameter in zip(present.cpu().tolist(), parameters):
            if not active:
                parameter.grad = None
                continue
            if parameter.grad is None:
                parameter.grad = torch.zeros_like(parameter)
            if count + parameter.numel() > bucket_elements:
                flush()
                count = 0
            bucket.append(parameter)
            count += parameter.numel()
        flush()

    def assert_models_equal(self, model):
        from gem.closedloop.common.evaluation import _model_fingerprint
        values = self.gather(_model_fingerprint(model))
        if len(set(values)) != 1:
            raise RuntimeError("AWR distributed model parameters diverged")
        return values[0]
