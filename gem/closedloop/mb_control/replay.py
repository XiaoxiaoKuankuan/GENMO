"""近期真实决策快照的紧凑内存回放，不存完整扩散链或伪物理状态。

研究长链模式接收无pending真实决策快照；短时模式接收包含完整pending请求、原预测
条件和真实首次接管轨迹的PhysicalAnchor。固定容量保存紧凑张量，同Actor版本无放回
采样；恢复回放不等于恢复物理求解器，采集器必须重新开始episode。
"""

from collections import deque
import random


class RecentReplay:
    def __init__(self, capacity=128, seed=42, balanced=False):
        if capacity < 1:
            raise ValueError("回放容量必须为正")
        self.items = deque(maxlen=capacity)
        self.random = random.Random(seed)
        self.balanced = balanced
        self.support_classification={}
        self.sample_labels={}

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
        eligible=[x for x in eligible if self.support_classification.get(id(x),True)]
        if not eligible:
            raise ValueError("必须先真实采集")
        if count < 1 or len(eligible) < count:
            raise ValueError("同版本独立真实样本不足，不能重复一个anchor冒充多样本微批")
        if self.balanced and actor_version is not None:
            from .sample_distribution import stratified_sample
            labels=[self.sample_labels.get(id(x)) for x in eligible]
            return stratified_sample(eligible, count, self.random,labels=labels if all(x is not None for x in labels) else None)
        return self.random.sample(eligible, count)

    def eligible_count(self, actor_version):
        return sum(getattr(x, "generation", None) is not None and
                   x.generation.request.actor_version == actor_version and self.support_classification.get(id(x),True) for x in self.items)

    def classify_support(self,engine,actor_version):
        """先逐候选验证完整H步原支持域，再抽样；记录被排除的具体分量和时刻。"""
        import torch
        from .validation import UnsupportedModelState
        from .sample_distribution import describe_sample
        rows=[];current_ids={id(x) for x in self.items}
        self.support_classification={k:v for k,v in self.support_classification.items() if k in current_ids}
        self.sample_labels={k:v for k,v in self.sample_labels.items() if k in current_ids}
        for sample in self.items:
            if sample.generation.request.actor_version!=actor_version:continue
            row=describe_sample(sample,engine.adapter)
            try:
                with torch.no_grad():engine.rollout_anchor(sample,plan_override=sample.generation.plan,support_check_only=True)
                self.support_classification[id(sample)]=True;row.update(stage="B_legal_H_anchor",legal=True)
            except UnsupportedModelState as error:
                self.support_classification[id(sample)]=False
                row.update(stage="C_outside_support",legal=False,reason=str(error),support_details=error.details)
            rows.append(row)
            self.sample_labels[id(sample)]=row
        return rows

    def state_dict(self):
        return {"capacity": self.items.maxlen, "rng": self.random.getstate(), "items": list(self.items)}

    def load_state_dict(self, state):
        if state["capacity"] != self.items.maxlen:
            raise ValueError("回放容量与续训配置不一致")
        self.items.clear()
        self.support_classification.clear()
        self.sample_labels.clear()
        self.items.extend(state["items"])
        self.random.setstate(state["rng"])
