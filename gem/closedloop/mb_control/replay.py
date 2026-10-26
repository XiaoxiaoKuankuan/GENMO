"""近期真实决策快照的紧凑内存回放，不存完整扩散链或伪物理状态。

只接收无pending真实历史快照，入库detach，固定容量淘汰旧占用分布。保存紧凑张量
便于检查点恢复采样分布；恢复回放不等于恢复物理求解器，采集器必须重新开始episode。
"""

from collections import deque
import random


class RecentReplay:
    def __init__(self, capacity=128, seed=42):
        if capacity < 1:
            raise ValueError("回放容量必须为正")
        self.items = deque(maxlen=capacity)
        self.random = random.Random(seed)

    def add(self, state):
        from .contracts import PhysicalAnchor
        if isinstance(state, PhysicalAnchor):
            state.validate(len(state.transitions))
            self.items.append(state)
            return
        state.validate(decision=True)
        if state.history_origin != "real":
            raise ValueError("真实状态回放不接收WM预测历史")
        self.items.append(state.detached())

    def sample(self, count=1, *, actor_version=None):
        eligible = list(self.items) if actor_version is None else [x for x in self.items
            if getattr(x, "generation", None) is not None and x.generation.request.actor_version == actor_version]
        if not eligible:
            raise ValueError("必须先真实采集")
        if count < 1 or len(eligible) < count:
            raise ValueError("同版本独立真实样本不足，不能重复一个anchor冒充多样本微批")
        return self.random.sample(eligible, count)

    def eligible_count(self, actor_version):
        return sum(getattr(x, "generation", None) is not None and
                   x.generation.request.actor_version == actor_version for x in self.items)

    def state_dict(self):
        return {"capacity": self.items.maxlen, "rng": self.random.getstate(), "items": list(self.items)}

    def load_state_dict(self, state):
        if state["capacity"] != self.items.maxlen:
            raise ValueError("回放容量与续训配置不一致")
        self.items.clear()
        self.items.extend(state["items"])
        self.random.setstate(state["rng"])
