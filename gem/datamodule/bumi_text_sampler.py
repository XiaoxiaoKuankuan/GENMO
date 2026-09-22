"""BUMI四库文本训练的可复现分层采样器。

按显式概率选择数据集，再均匀选择canonical母动作和其具体镜像/子片段，最后交由
Dataset选择caption及标注区间内的120帧窗口。caption数量与可裁窗口数量不增加母动作
权重。每次抽样由seed、epoch、全局抽样序号决定，各DDP rank处理互不重叠的序号，
因此worker数量不改变抽样结果。重复抽中同一动作属于有放回采样，不冒充全库无重复。
本模块只处理索引，不加载T5或动作分片；既有音乐和SMPL采样路径保持独立。
"""

import math
from collections import defaultdict
from dataclasses import dataclass

import numpy as np
from torch.utils.data import Dataset, Sampler


@dataclass(frozen=True)
class BumiTextDraw:
    dataset_index: int
    record_index: int
    random_seed: int
    draw_index: int


class BumiTextMixture(Dataset):
    """接收分层索引，将窗口随机种子显式传给对应Dataset。"""

    def __init__(self, datasets):
        self.datasets = tuple(datasets)

    def __len__(self):
        return sum(map(len, self.datasets))

    def __getitem__(self, draw):
        if not isinstance(draw, BumiTextDraw):
            raise TypeError("BUMI文本混合数据必须使用分层采样索引")
        sample = self.datasets[draw.dataset_index].get_window(
            draw.record_index, random_seed=draw.random_seed
        )
        sample["meta"]["sampler_draw_index"] = draw.draw_index
        return sample


class BumiTextDistributedSampler(Sampler):
    """数据集→母动作→具体记录；每个epoch的抽样总数按DDP world size整除。"""

    def __init__(
        self,
        datasets,
        dataset_probabilities,
        samples_per_epoch,
        seed=20260922,
        rank=0,
        num_replicas=1,
    ):
        self.names = [ds.dataset for ds in datasets]
        if len(set(self.names)) != len(self.names) or set(self.names) != set(dataset_probabilities):
            raise ValueError("采样概率必须精确覆盖各个不同数据集")
        values = np.asarray([float(dataset_probabilities[n]) for n in self.names])
        if (
            not np.isfinite(values).all()
            or (values <= 0).any()
            or not math.isclose(float(values.sum()), 1.0, abs_tol=1e-8)
        ):
            raise ValueError("数据集采样概率必须为有限正数且总和为1")
        if (
            type(samples_per_epoch) is not int
            or samples_per_epoch < num_replicas
            or not 0 <= rank < num_replicas
        ):
            raise ValueError("抽样总数或DDP rank/world size非法")
        if any(ds.split != "train" or ds.sequence_mode != "crop" for ds in datasets):
            raise ValueError("BUMI文本分层采样仅用于crop训练集")
        self.probabilities = values
        self.rank, self.num_replicas = rank, num_replicas
        self.samples_per_epoch = samples_per_epoch - samples_per_epoch % num_replicas
        self.dropped_samples = samples_per_epoch % num_replicas
        self.seed, self.epoch = int(seed), 0
        self.groups = []
        for ds in datasets:
            groups = defaultdict(list)
            for i, key in enumerate(ds.sample_source_groups()):
                groups[key].append(i)
            if not groups:
                raise ValueError("数据集母来源分组为空")
            self.groups.append([groups[k] for k in sorted(groups)])

    def __len__(self):
        return self.samples_per_epoch // self.num_replicas

    def set_epoch(self, epoch):
        self.epoch = int(epoch)

    def __iter__(self):
        for draw in range(self.rank, self.samples_per_epoch, self.num_replicas):
            rng = np.random.default_rng(np.random.SeedSequence([self.seed, self.epoch, draw]))
            di = int(rng.choice(len(self.names), p=self.probabilities))
            group = self.groups[di][int(rng.integers(len(self.groups[di])))]
            ri = group[int(rng.integers(len(group)))]
            yield BumiTextDraw(di, ri, int(rng.integers(0, 2**32)), draw)
