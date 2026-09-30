"""第九步 train 音乐任务采样和可恢复的独立随机状态。

本模块只读取指定 train manifest 与经过修正的 200 首选曲清单，不调用第八步 val
选曲函数，不读取示范动作标签。清单的行、分组与完整 train manifest 对照，特征 SHA
由既有 EDGE35 loader 校验。四来源采样概率固定为 20/35/25/20；最初四个任务逐一
覆盖四库，此后按概率抽来源、在来源内部打乱无放回选曲，不依据执行表现筛选音乐。

采样器拥有独立 NumPy Generator，checkpoint 保存 RNG、各库排列和游标以及清单内容
身份。恢复必须匹配原清单，不能在缺文件时静默改用 val 或另选一首歌。任务总时长
来自真实音乐帧数，采集 30 秒上限属于行政截断，不改变这个有限音乐任务的终止语义。
配对动作活动度由独立 ``target_activity.load_paired_activity`` 显式从 bc_data_root
加载，仅交给奖励；本采样器继续支持音乐独立部署，不把监督活动度加入网络条件。
"""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import numpy as np

from gem.closedloop.evaluation_music import load_music_features

SOURCES = ("AIST++", "AIOZ-GDANCE", "FineDance", "Mine")
SOURCE_PROBABILITIES = (.20, .35, .25, .20)


class TrainMusicSampler:
    def __init__(self, data_root, selection_path, seed=42):
        self.data_root = Path(data_root).resolve()
        self.selection_path = Path(selection_path).resolve()
        payload = self.selection_path.read_bytes()
        self.selection_sha256 = hashlib.sha256(payload).hexdigest()
        selection = json.loads(payload)
        if not isinstance(selection, list) or not selection:
            raise ValueError("corrected train selection must be a nonempty list")
        self.selected = copy.deepcopy(selection)
        self.samples = {name: [] for name in SOURCES}
        rows_by_source, self.manifest_sha256 = {}, {}
        for source in SOURCES:
            path = self.data_root / source / "manifests/train.jsonl"
            content = path.read_bytes()
            rows = [json.loads(line) for line in content.decode().splitlines() if line.strip()]
            self.manifest_sha256[source] = hashlib.sha256(content).hexdigest()
            rows_by_source[source] = {row["sample_id"]: row for row in rows}
            if len(rows_by_source[source]) != len(rows) or any(row["split"] != "train" for row in rows):
                raise ValueError("train manifest contains duplicate samples or non-train rows")
        seen_groups, seen_audio = set(), set()
        for sample in selection:
            source, row = sample["dataset"], sample["row"]
            if source not in SOURCES or row.get("split") != "train" or row.get("fps") != 30 or row != rows_by_source[source].get(row.get("sample_id")):
                raise ValueError("selected row does not exactly match its current train manifest")
            if sample.get("manifest_sha256") != self.manifest_sha256[source]:
                raise ValueError("selected train manifest SHA mismatch")
            group = row["resplit_provenance"]["group_id"]
            if group != sample["group_id"] or group in seen_groups or row["source_audio_sha256"] in seen_audio:
                raise ValueError("train selection contains overlapping music groups or audio")
            seen_groups.add(group)
            seen_audio.add(row["source_audio_sha256"])
            path = (self.data_root / source / row["music_feature_path"]).resolve()
            if not path.is_relative_to(self.data_root / source) or not path.is_file():
                raise ValueError("selected train music file is missing or outside its source root")
            if hashlib.sha256(path.read_bytes()).hexdigest() != row["source_music_feature_sha256"]:
                raise ValueError("selected train music feature SHA mismatch")
            self.samples[source].append(copy.deepcopy(sample))
        if any(not values for values in self.samples.values()):
            raise ValueError("all four train sources must be available")
        self.rng = np.random.default_rng(seed)
        self.draw_count = 0
        self.orders = {name: self.rng.permutation(len(self.samples[name])).tolist() for name in SOURCES}
        self.cursors = {name: 0 for name in SOURCES}

    def next_task(self):
        source = SOURCES[self.draw_count] if self.draw_count < len(SOURCES) else str(self.rng.choice(SOURCES, p=SOURCE_PROBABILITIES))
        if self.cursors[source] == len(self.orders[source]):
            self.orders[source] = self.rng.permutation(len(self.samples[source])).tolist()
            self.cursors[source] = 0
        sample = copy.deepcopy(self.samples[source][self.orders[source][self.cursors[source]]])
        features = load_music_features(self.data_root, sample)
        selected_sample = copy.deepcopy(sample)
        self.cursors[source] += 1
        self.draw_count += 1
        sample.update({"music_features": features, "task_index": self.draw_count - 1,
                       "sample": selected_sample, "music": features,
                       "music_duration_seconds": len(features) / 30.,
                       "music_duration_ticks": len(features) * 20,
                       "sampling_probabilities": dict(zip(SOURCES, SOURCE_PROBABILITIES))})
        return sample

    def state_dict(self):
        return copy.deepcopy({"version": "stage9.train_music.v1", "selection_sha256": self.selection_sha256,
                              "manifest_sha256": self.manifest_sha256, "rng": self.rng.bit_generator.state,
                              "draw_count": self.draw_count, "orders": self.orders, "cursors": self.cursors})

    def load_state_dict(self, state):
        state = copy.deepcopy(state)
        if state.get("version") != "stage9.train_music.v1" or state.get("selection_sha256") != self.selection_sha256 or state.get("manifest_sha256") != self.manifest_sha256:
            raise ValueError("music sampler checkpoint identity mismatch")
        if not isinstance(state["draw_count"], int) or state["draw_count"] < 0:
            raise ValueError("invalid music draw count")
        for name in SOURCES:
            order, cursor = state["orders"][name], state["cursors"][name]
            if sorted(order) != list(range(len(self.samples[name]))) or not isinstance(cursor, int) or not 0 <= cursor <= len(order):
                raise ValueError("invalid music permutation/cursor")
        restored = np.random.default_rng()
        restored.bit_generator.state = state["rng"]
        self.rng, self.draw_count = restored, state["draw_count"]
        self.orders, self.cursors = state["orders"], state["cursors"]
