"""原配对数据的独立确定性采样与紧凑真实sidecar索引。

直接复用Stage1 Dataset/codec/地面标签，不重建音乐动作配对或split。源内保留duration-aware
权重，裁剪落到30/50Hz公共网格；独立PCG64游标保证A/B和恢复的数据顺序一致。sidecar用
dataset/split/source ID连接分片，一首歌只存一次参考和每步状态/观测；历史按时间索引还原。
读取必须核对原文件内容身份，不按shuffle顺序匹配，也不将val/test缓存传给train。
"""

from collections import OrderedDict
from pathlib import Path
import copy
import json

import numpy as np
from omegaconf import OmegaConf
import torch

from gem.closedloop.stage1_dataset import BumiClosedLoopStage1Dataset, collate_stage1_training_samples
from gem.closedloop.training import batch_to_device
from .config import data_config
from .contracts import SIDECAR_SCHEMA, PhysicsRejected, digest, source_key


class SourceCatalog:
    def __init__(self, cfg, split):
        self.split = split
        dc = data_config(cfg)
        self.datasets, self.rows, self.weights = [], [], []
        selected = cfg["data"].get("source_ids")
        for entry in dc.datasets[split].values():
            options = OmegaConf.to_container(entry, resolve=True)
            options.pop("_target_")
            dataset = BumiClosedLoopStage1Dataset(**options)
            allowed = [i for i, row in enumerate(dataset.rows) if selected is None or row["sample_id"] in selected]
            if not allowed:
                continue
            self.datasets.append(dataset)
            self.rows.append(allowed)
            self.weights.append(float(dc.train_sampling_reference.get(dataset.dataset_name, 1.)))
        if not self.datasets:
            raise ValueError("所选split/source ID没有原配对数据")
        self.probabilities = np.asarray(self.weights) / sum(self.weights)
        self.identity = {d.dataset_name: {"split": split, "manifest_sha256": digest(d.reader.manifest_path),
                         "selected_ids": [d.rows[i]["sample_id"] for i in rows]}
                         for d, rows in zip(self.datasets, self.rows)}

    def sources(self):
        for dataset, indices in zip(self.datasets, self.rows):
            for row in indices:
                yield dataset, row


class JointDataStream:
    def __init__(self, cfg, split="train", rank=0, world_size=1):
        self.catalog = SourceCatalog(cfg, split)
        self.rng = np.random.default_rng(cfg["training"]["seed"] + rank * 100003)
        self.draws = 0
        self.rank, self.world_size = rank, world_size

    def next(self, count, device):
        samples = []
        for _ in range(count):
            index = int(self.rng.choice(len(self.catalog.datasets), p=self.catalog.probabilities))
            dataset, rows = self.catalog.datasets[index], self.catalog.rows[index]
            # 沿用原Dataset展开索引的时长采样权重；窗口相位在全曲范围随机选择。
            repeats = np.bincount(dataset.idx2meta, minlength=len(dataset.rows))[rows].astype(float)
            row_index = int(self.rng.choice(rows, p=repeats / repeats.sum()))
            frames = int(dataset.rows[row_index]["num_frames"])
            start = int(self.rng.integers((frames - 1) // 3 + 1)) * 3
            sample = dataset.get_window(row_index, start_frame=start)
            sample["meta"] = dict(sample["meta"], joint_draw=self.draws, joint_rank=self.rank,
                                   time_conversion="source_frame*20=600Hz_tick; common_grid=60")
            samples.append(sample)
            self.draws += 1
        return batch_to_device(collate_stage1_training_samples(samples), device)

    def state_dict(self):
        return {"rng": copy.deepcopy(self.rng.bit_generator.state), "draws": self.draws,
                "rank": self.rank, "world_size": self.world_size, "identity": self.catalog.identity}

    def load_state_dict(self, value):
        if value["identity"] != self.catalog.identity or (value["rank"], value["world_size"]) != (self.rank, self.world_size):
            raise ValueError("恢复要求同一数据划分、采样合同与rank布局")
        self.rng.bit_generator.state = value["rng"]
        self.draws = value["draws"]


class SidecarStore:
    def __init__(self, directory, split, expected_assets=None):
        self.directory = Path(directory).resolve()
        self.index = json.loads((self.directory / "index.json").read_text())
        if self.index["schema"] != SIDECAR_SCHEMA or self.index["split"] != split:
            raise ValueError("sidecar schema/split不匹配，禁止跨split训练")
        if expected_assets is not None and self.index["assets"] != expected_assets:
            raise ValueError("sidecar的SM/WM、stats、FK或物理合同不匹配")
        self.cache, self.verified = OrderedDict(), set()
        self.identity = digest(self.directory / "index.json")

    def episode(self, meta):
        key = source_key(meta)
        if key not in self.index["sources"]:
            raise PhysicsRejected("missing_real_sidecar")
        entry = self.index["sources"][key]
        shard = entry["shard"]
        if shard not in self.cache:
            if digest(self.directory / shard) != self.index["shards"][shard]:
                raise ValueError("sidecar分片内容身份已改变")
            self.cache[shard] = torch.load(self.directory / shard, weights_only=False, map_location="cpu")
            if len(self.cache) > 2:
                self.cache.popitem(last=False)
        episode = self.cache[shard][entry["episode"]]
        if episode["source_key"] != key or episode["split"] != meta["split"]:
            raise ValueError("sidecar索引指向错误来源")
        if key not in self.verified:
            for field, identity in (("motion_path", "motion_sha256"), ("music_feature_path", "music_sha256")):
                if digest(meta[field]) != episode[identity]:
                    raise ValueError("原音乐/动作与真实sidecar内容不一致")
            self.verified.add(key)
        return episode


def slice_batch(batch, index):
    return {k: (v[index:index+1] if isinstance(v, torch.Tensor) else [v[index]] if k == "meta" else 1 if k == "B" else v)
            for k, v in batch.items()}
