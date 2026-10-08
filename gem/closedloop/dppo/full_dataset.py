"""第十步完整四库目录、配对文件审计与可恢复的来源比例任务采样。

本模块读取四个数据源的全部 train/val/test 清单，不引用第八/九步的 200 首选择文件，
不把目录名中的划分比例当作实际样本数量。构造时检查唯一样本、划分、全局音乐组、
音频及原动作身份的跨划分泄漏；同一划分内同音乐的不同舞者/裁剪合法保留。显式调用
audit_files 才扫描全部音乐、音频与动作，逐条记录当前 motion.pt 内容 SHA 和原动作
来源声明的区别。只读 CPU 审计，不加载 Actor，也不修改源数据或划分。

FullMusicSampler 先按 20/35/25/20 抽来源，再在来源完整清单中按独立 RNG 打乱、
无放回抽样。随机起点以 30 Hz 源帧计数，返回完整音乐和 start_frame；环境必须对音乐
与配对活动目标同步应用偏移，音乐末尾仍是真任务结束，window_seconds 仅为行政采集
窗口。采样概率是任务概率，不伪称实际控制时间比例。record_execution 单独统计真实
控制步、转移和已覆盖源时间区间，并保存 RNG、游标、当前任务进度与覆盖以供恢复。

此采样器面向单环境顺序执行；开始下一任务后不能再回填旧任务的执行计数。val/test
可用同一只读加载接口创建评价任务，但训练入口必须显式选择 train；任何奖励标签、
覆盖统计或数据来源字段都不会被加入 Actor 的十个条件字段。
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
from numbers import Integral
from pathlib import Path

import numpy as np

from gem.closedloop.dppo.music_tasks import SOURCES, SOURCE_PROBABILITIES
from gem.closedloop.dppo.target_activity import _load_verified_pair, _path, _sha
from gem.closedloop.evaluation_music import load_music_features, sha256_file


SPLITS = ("train", "val", "test")
CATALOG_VERSION = "genmo.closedloop.full_music_catalog.v1"
SAMPLER_VERSION = "genmo.closedloop.full_music_sampler.v1"


def _integer(value, name, minimum=0):
    if isinstance(value, bool) or not isinstance(value, Integral) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(value)


class FullMusicCatalog:
    """完整12份清单的只读快照；文件全量扫描由 audit_files 显式执行。"""

    def __init__(self, data_root):
        self.data_root = Path(data_root).expanduser().resolve()
        self.samples = {split: {} for split in SPLITS}
        self._manifest_paths, self._metadata_paths, self._lookup = {}, {}, {}
        manifest_sha, metadata_sha, seen_samples = {}, {}, set()
        ownership = {name: {} for name in ("group_id", "source_audio_sha256", "source_motion_sha256")}
        for source in SOURCES:
            root = (self.data_root / source).resolve()
            info_path = _path(root, "meta/dataset_info.json", "dataset_info")
            info_bytes = info_path.read_bytes()
            info = json.loads(info_bytes)
            if not isinstance(info, dict) or not isinstance(info.get("dataset_name"), str):
                raise ValueError(f"{source}: invalid dataset metadata")
            self._metadata_paths[source] = info_path
            metadata_sha[source] = hashlib.sha256(info_bytes).hexdigest()
            for split in SPLITS:
                path = _path(root, f"manifests/{split}.jsonl", f"{split} manifest")
                content = path.read_bytes()
                digest = hashlib.sha256(content).hexdigest()
                rows = [json.loads(line) for line in content.decode().splitlines() if line.strip()]
                if not rows:
                    raise ValueError(f"{source}/{split}: full manifest must be nonempty")
                declared_count = info.get("split_counts", {}).get(split)
                if (isinstance(declared_count, bool) or not isinstance(declared_count, int)
                        or declared_count != len(rows)):
                    raise ValueError(f"{source}/{split}: full manifest count differs from dataset_info split_counts")
                selected = []
                for row in rows:
                    if not isinstance(row, dict) or row.get("split") != split or row.get("fps") != 30:
                        raise ValueError(f"{source}/{split}: invalid row split or fps")
                    sample_id = row.get("sample_id")
                    if not isinstance(sample_id, str) or not sample_id or (source, sample_id) in seen_samples:
                        raise ValueError(f"{source}/{split}: duplicate or invalid sample_id")
                    seen_samples.add((source, sample_id))
                    _integer(row.get("num_frames"), "num_frames", 2)
                    if row.get("quality_accepted") is not True or row.get("dataset") != info["dataset_name"]:
                        raise ValueError(f"{source}/{split}: row quality or dataset mismatch")
                    group = row.get("resplit_provenance", {}).get("group_id")
                    if not isinstance(group, str) or not group:
                        raise ValueError(f"{source}/{split}: missing global music group identity")
                    identities = {"group_id": group}
                    for key in ("source_audio_sha256", "source_motion_sha256", "source_music_feature_sha256"):
                        _sha(row.get(key), key)
                        if key != "source_music_feature_sha256":
                            identities[key] = row[key]
                    for field, value in identities.items():
                        previous = ownership[field].setdefault(value, split)
                        if previous != split:
                            raise ValueError(f"cross-split leakage: {field} appears in {previous} and {split}")
                    # 路径格式在构造时检查；文件是否存在和内容在显式全量审计时检查。
                    for key in ("motion_path", "music_feature_path", "audio_path"):
                        value = row.get(key)
                        if not isinstance(value, str) or not value or Path(value).is_absolute():
                            raise ValueError(f"{source}/{split}: {key} must be relative")
                        if not (root / value).resolve().is_relative_to(root):
                            raise ValueError(f"{source}/{split}: {key} escapes dataset root")
                    sample = dict(dataset=source, split=split, group_id=group, row=copy.deepcopy(row),
                                  manifest_sha256=digest, pretraining_unseen_claim=False)
                    selected.append(sample)
                    self._lookup[(split, source, sample_id)] = copy.deepcopy(sample)
                selected.sort(key=lambda item: item["row"]["sample_id"])
                self.samples[split][source] = tuple(selected)
                self._manifest_paths[(split, source)] = path
                manifest_sha[f"{source}/{split}"] = digest
        self.identity = dict(version=CATALOG_VERSION, manifest_sha256=manifest_sha,
                             dataset_info_sha256=metadata_sha,
                             sample_counts={split: {source: len(self.samples[split][source]) for source in SOURCES}
                                            for split in SPLITS})
        self.identity["catalog_sha256"] = hashlib.sha256(json.dumps(
            self.identity, sort_keys=True, separators=(",", ":")).encode()).hexdigest()

    def _verify_identity(self):
        for (split, source), path in self._manifest_paths.items():
            if sha256_file(path) != self.identity["manifest_sha256"][f"{source}/{split}"]:
                raise ValueError("full catalog manifest changed after construction")
        for source, path in self._metadata_paths.items():
            if sha256_file(path) != self.identity["dataset_info_sha256"][source]:
                raise ValueError("full catalog dataset metadata changed after construction")

    def validate_sample(self, sample):
        if not isinstance(sample, dict) or not isinstance(sample.get("row"), dict):
            raise ValueError("full catalog sample must contain its manifest row")
        split, source = sample.get("split", sample["row"].get("split")), sample.get("dataset")
        expected = self._lookup.get((split, source, sample["row"].get("sample_id")))
        if (expected is None or sample["row"] != expected["row"] or sample.get("group_id") != expected["group_id"]
                or sample.get("manifest_sha256") != expected["manifest_sha256"]):
            raise ValueError("sample differs from complete catalog manifest identity")
        if expected.get("motion_file_sha256") is not None and sample.get("motion_file_sha256") != expected["motion_file_sha256"]:
            raise ValueError("sample is missing its audited motion file identity")
        return copy.deepcopy(expected)

    def load_music(self, sample):
        expected = self.validate_sample(sample)
        path = self._manifest_paths[(expected["split"], expected["dataset"])]
        if sha256_file(path) != expected["manifest_sha256"]:
            raise ValueError("full catalog manifest changed before music load")
        return load_music_features(self.data_root, expected)

    def audit_files(self, progress=None, *, require_audio=True):
        """扫描全部记录并返回逐条证据；任何缺失/不匹配直接报错，不静默缩小池。"""
        if progress is not None and not callable(progress):
            raise TypeError("audit progress must be callable")
        self._verify_identity()
        records, verified_audio = [], {}
        for split in SPLITS:
            for source in SOURCES:
                for sample in self.samples[split][source]:
                    self.validate_sample(sample)
                    values = load_music_features(self.data_root, sample)
                    target = _load_verified_pair(self.data_root, sample,
                        self._manifest_paths[(split, source)], sample["manifest_sha256"], split=split)
                    row, evidence = sample["row"], target.source
                    root = (self.data_root / source).resolve()
                    audio_path = (root / row["audio_path"]).resolve()
                    if not audio_path.is_relative_to(root):
                        raise ValueError("audit audio path escapes dataset root")
                    audio_hash = None
                    if require_audio or audio_path.is_file():
                        audio_path = _path(root, row["audio_path"], "audio_path")
                        stat = audio_path.stat()
                        signature = (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)
                        previous = verified_audio.get(audio_path)
                        if previous is None or previous[0] != signature:
                            verified_audio[audio_path] = (signature, sha256_file(audio_path))
                        audio_hash = verified_audio[audio_path][1]
                        if audio_hash != row["source_audio_sha256"]:
                            raise ValueError("full audit audio SHA mismatch")
                    payload_sha = evidence["motion_file_sha256"]
                    if row.get("motion_file_sha256") is not None and row["motion_file_sha256"] != payload_sha:
                        raise ValueError("full audit motion payload SHA mismatch")
                    record = dict(dataset=source, split=split, sample_id=row["sample_id"], group_id=sample["group_id"],
                        num_frames=len(values), manifest_sha256=sample["manifest_sha256"],
                        motion_path=str(_path(root, row["motion_path"], "motion_path")),
                        motion_payload_sha256=payload_sha, payload_sha256=payload_sha,
                        source_motion_sha256=row["source_motion_sha256"],
                        source_motion_sha_validation=evidence["source_motion_sha_validation"],
                        music_feature_path=str(_path(root, row["music_feature_path"], "music_feature_path")),
                        music_feature_sha256=row["source_music_feature_sha256"],
                        audio_path=str(audio_path), audio_sha256=audio_hash, audio_verified=audio_hash is not None,
                        dataset_info_sha256=evidence["dataset_info_sha256"])
                    records.append(record)
                    if progress is not None:
                        progress(copy.deepcopy(record))
        self._verify_identity()
        for path, (signature, _) in verified_audio.items():
            stat = path.stat()
            if (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns) != signature:
                raise ValueError("full audit audio changed during scan")
        content_sha = hashlib.sha256(json.dumps(records, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        # 只有全部记录通过后才发布当前 payload 身份；后续 reward load 再逐次比对。
        for record in records:
            key = (record["split"], record["dataset"], record["sample_id"])
            self._lookup[key]["motion_file_sha256"] = record["motion_payload_sha256"]
        for split in SPLITS:
            for source in SOURCES:
                self.samples[split][source] = tuple(copy.deepcopy(self._lookup[(split, source, item["row"]["sample_id"])])
                    for item in self.samples[split][source])
        self.identity["audited_data_content_sha256"] = content_sha
        return dict(version="genmo.closedloop.full_dataset_audit.v1", status="passed", complete_manifest_scan=True,
                    require_audio=bool(require_audio), sample_count=len(records), identity=copy.deepcopy(self.identity),
                    data_content_sha256=content_sha,
                    unique_audio_files_hashed=len(verified_audio),
                    records=records)


def _merge_intervals(existing, begin, end):
    result = []
    for left, right in sorted([*existing, [begin, end]]):
        if result and left <= result[-1][1]:
            result[-1][1] = max(result[-1][1], right)
        else:
            result.append([left, right])
    return result


class FullMusicSampler:
    """完整清单内来源概率采样；真实执行覆盖必须由调用方 record_execution 回填。"""

    def __init__(self, catalog, *, split="train", seed=42, window_seconds=30., random_start=True,
                 source_probabilities=SOURCE_PROBABILITIES):
        if not isinstance(catalog, FullMusicCatalog) or split not in SPLITS:
            raise ValueError("full sampler requires a complete catalog and explicit valid split")
        _integer(seed, "seed")
        if not math.isfinite(float(window_seconds)) or window_seconds <= 0 or not isinstance(random_start, bool):
            raise ValueError("window_seconds must be positive and random_start must be boolean")
        probabilities = np.asarray(source_probabilities, dtype=float)
        if probabilities.shape != (4,) or not np.isfinite(probabilities).all() or (probabilities <= 0).any() or not np.isclose(probabilities.sum(), 1., atol=1e-12, rtol=0):
            raise ValueError("four positive source probabilities must sum to one")
        self.catalog, self.split = catalog, split
        self.window_seconds, self.random_start = float(window_seconds), random_start
        self.probabilities = probabilities.copy()
        self.rng = np.random.default_rng(int(seed))
        self.draw_count = 0
        self.orders = {source: self.rng.permutation(len(catalog.samples[split][source])).tolist() for source in SOURCES}
        self.cursors = dict.fromkeys(SOURCES, 0)
        self._counts = {source: dict(tasks_drawn=0, executed_tasks=0, upper_transitions=0, control_steps=0) for source in SOURCES}
        self._drawn, self._intervals = {source: set() for source in SOURCES}, {}
        self._active = None

    def next_task(self):
        source = str(self.rng.choice(SOURCES, p=self.probabilities))
        if self.cursors[source] == len(self.orders[source]):
            self.orders[source] = self.rng.permutation(len(self.orders[source])).tolist()
            self.cursors[source] = 0
        index = self.orders[source][self.cursors[source]]
        sample = copy.deepcopy(self.catalog.samples[self.split][source][index])
        music = self.catalog.load_music(sample)
        maximum = max(0, len(music) - max(2, math.ceil(self.window_seconds * 30)))
        start = int(self.rng.integers(maximum + 1)) if self.random_start else 0
        self.cursors[source] += 1
        self.draw_count += 1
        self._counts[source]["tasks_drawn"] += 1
        self._drawn[source].add(sample["row"]["sample_id"])
        self._active = dict(task_index=self.draw_count - 1, dataset=source, sample_id=sample["row"]["sample_id"],
                            start_frame=start, control_steps=0, num_frames=len(music))
        return dict(sample=sample, music=music, start_frame=start, music_start_frame=start,
                    task_index=self.draw_count - 1, full_music_duration_seconds=len(music) / 30.,
                    music_duration_seconds=len(music) / 30., remaining_music_seconds=(len(music)-start) / 30.,
                    window_duration_seconds=min(self.window_seconds, (len(music)-start) / 30.),
                    sampling_probabilities=dict(zip(SOURCES, self.probabilities.tolist())))

    def record_execution(self, task, control_steps, *, upper_transitions=1):
        count, upper = _integer(control_steps, "actual control steps"), _integer(upper_transitions, "upper transitions")
        sample = self.catalog.validate_sample(task["sample"])
        if (self._active is None or task.get("task_index") != self._active["task_index"]
                or sample["dataset"] != self._active["dataset"] or sample["row"]["sample_id"] != self._active["sample_id"]
                or task.get("start_frame") != self._active["start_frame"]):
            raise ValueError("execution coverage must refer to the current issued task")
        previous = self._active["control_steps"]
        if previous + count > (self._active["num_frames"] - self._active["start_frame"]) * 50 // 30:
            raise ValueError("executed coverage exceeds remaining real music duration")
        source = sample["dataset"]
        stats = self._counts[source]
        stats["executed_tasks"] += int(previous == 0 and count > 0)
        stats["upper_transitions"] += upper
        stats["control_steps"] += count
        key = f"{source}/{sample['row']['sample_id']}"
        if count:
            begin = self._active["start_frame"] * 20 + previous * 12
            self._intervals[key] = _merge_intervals(self._intervals.get(key, []), begin, begin + count * 12)
        self._active["control_steps"] += count

    def coverage(self):
        result = {}
        for source in SOURCES:
            spans = [v for k, v in self._intervals.items() if k.startswith(source + "/")]
            result[source] = {**self._counts[source], "actual_seconds": self._counts[source]["control_steps"] / 50.,
                "unique_samples_drawn": len(self._drawn[source]), "unique_samples_executed": len(spans),
                "covered_source_seconds": sum(right-left for intervals in spans for left, right in intervals) / 600.,
                "catalog_samples": len(self.catalog.samples[self.split][source])}
        return dict(split=self.split, source_probabilities=dict(zip(SOURCES, self.probabilities.tolist())),
                    probability_scope="tasks_not_control_time", draw_count=self.draw_count, per_source=result)

    def state_dict(self):
        return copy.deepcopy(dict(version=SAMPLER_VERSION, catalog_identity=self.catalog.identity, split=self.split,
            window_seconds=self.window_seconds, random_start=self.random_start, source_probabilities=self.probabilities.tolist(),
            rng=self.rng.bit_generator.state, draw_count=self.draw_count, orders=self.orders, cursors=self.cursors,
            counts=self._counts, drawn={k: sorted(v) for k, v in self._drawn.items()}, intervals=self._intervals, active=self._active))

    def _validated_state(self, state):
        """仅复制轻量采样状态进行验证，不复制只读 catalog 或读取音乐数据。"""
        state = copy.deepcopy(state)
        expected = dict(version=SAMPLER_VERSION, catalog_identity=self.catalog.identity, split=self.split,
            window_seconds=self.window_seconds, random_start=self.random_start, source_probabilities=self.probabilities.tolist())
        for key in ("version", "catalog_identity", "split", "window_seconds", "random_start", "source_probabilities"):
            if state.get(key) != expected[key]:
                raise ValueError(f"full sampler checkpoint identity differs: {key}")
        count = _integer(state["draw_count"], "draw_count")
        for source in SOURCES:
            n = len(self.catalog.samples[self.split][source])
            if (any(isinstance(v, bool) or not isinstance(v, Integral) for v in state["orders"][source])
                    or sorted(state["orders"][source]) != list(range(n))
                    or not 0 <= _integer(state["cursors"][source], "cursor") <= n):
                raise ValueError("invalid full sampler permutation or cursor")
            known = {item["row"]["sample_id"] for item in self.catalog.samples[self.split][source]}
            if len(set(state["drawn"][source])) != len(state["drawn"][source]) or not set(state["drawn"][source]).issubset(known):
                raise ValueError("invalid full sampler sample coverage")
            for field in ("tasks_drawn", "executed_tasks", "upper_transitions", "control_steps"):
                _integer(state["counts"][source][field], f"coverage {field}")
            if state["counts"][source]["executed_tasks"] > state["counts"][source]["tasks_drawn"]:
                raise ValueError("executed task coverage exceeds issued tasks")
        if sum(state["counts"][s]["tasks_drawn"] for s in SOURCES) != count:
            raise ValueError("full sampler task counts do not match draw count")
        for key, intervals in state["intervals"].items():
            source, sample_id = key.split("/", 1)
            sample = self.catalog._lookup.get((self.split, source, sample_id))
            if sample is None or not isinstance(intervals, list):
                raise ValueError("coverage references an unknown full catalog sample")
            end = -1
            for pair in intervals:
                if len(pair) != 2:
                    raise ValueError("invalid coverage interval")
                left, right = (_integer(v, "coverage tick") for v in pair)
                if not end < left < right <= sample["row"]["num_frames"] * 20:
                    raise ValueError("coverage intervals must be ordered, disjoint and inside music")
                end = right
        for source in SOURCES:
            covered = sum(right-left for key, spans in state["intervals"].items() if key.startswith(source+"/") for left, right in spans)
            if covered > state["counts"][source]["control_steps"] * 12:
                raise ValueError("covered source time exceeds actual control time")
        active = state["active"]
        if active is not None:
            sample = self.catalog._lookup.get((self.split, active.get("dataset"), active.get("sample_id")))
            if sample is None or active.get("task_index") != count-1 or active.get("num_frames") != sample["row"]["num_frames"]:
                raise ValueError("invalid active full sampler task")
            start = _integer(active["start_frame"], "active start")
            steps = _integer(active["control_steps"], "active steps")
            if start > active["num_frames"]-2 or steps > (active["num_frames"]-start)*50//30:
                raise ValueError("active task time exceeds real music")
        elif count:
            raise ValueError("nonempty sampler checkpoint is missing active task")
        rng = np.random.default_rng()
        rng.bit_generator.state = state["rng"]
        return state, rng, count

    def validate_state_dict(self, state):
        """恢复前无副作用检查；不改当前游标/RNG，也不触发 catalog 深拷贝。"""
        self._validated_state(state)
        return True

    def load_state_dict(self, state):
        state, rng, count = self._validated_state(state)
        self.rng, self.draw_count = rng, count
        self.orders, self.cursors, self._counts = state["orders"], state["cursors"], state["counts"]
        self._drawn = {source: set(state["drawn"][source]) for source in SOURCES}
        self._intervals, self._active = state["intervals"], state['active']
