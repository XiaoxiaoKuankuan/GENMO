"""原配对数据的独立确定性采样与紧凑真实sidecar索引。

直接复用Stage1 Dataset/codec/地面标签，不重建音乐动作配对或split。源内保留duration-aware
权重，裁剪落到30/50Hz公共网格；独立PCG64游标保证A/B和恢复的数据顺序一致。sidecar用
dataset/split/source ID连接分片，一首歌只存一次参考和每步状态/观测；历史按时间索引还原。
读取必须核对原文件内容身份，不按shuffle顺序匹配，也不将val/test缓存传给train。

大批训练可启用有界整曲CPU缓存、多线程窗口装配和单批预取。所有抽样仍由主线程的独立
PCG64执行，后台只处理显式(source, frame)计划；checkpoint只记录已消费游标，预取计划
由该游标确定性重建，不会把尚未消费的数据跳过去。CPU缓存复用原reader、接触和地面算法，
不缓存GPU张量、不改变原条件/标签数学，也不在后台消费Python/NumPy/Torch全局RNG。
"""

from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import RLock
import copy
import hashlib
import json
import time

import numpy as np
from omegaconf import OmegaConf
import torch

from gem.closedloop.stage1_dataset import BumiClosedLoopStage1Dataset, collate_stage1_training_samples
from gem.closedloop.training import batch_to_device
from .config import data_config
from .contracts import SIDECAR_SCHEMA, RAW_SIDECAR_SCHEMA, PhysicsRejected, digest, source_key


class _CachedSequenceReader:
    """线程安全的整曲LRU，只保存通过原reader检查的CPU数据和原算法派生标签。"""

    def __init__(self, reader, dataset, max_bytes):
        self.reader, self.dataset, self.max_bytes = reader, dataset, int(max_bytes)
        self.cache, self.signatures = OrderedDict(), {}
        self.bytes = 0
        self.lock = RLock()
        self.prepared = None
        self.prepared_verified = set()

    def __getattr__(self, name):
        return getattr(self.reader, name)

    def _signature(self, row):
        from gem.datasets.music_dance.music_dance_bumi import resolve_contract_path
        values = []
        for field in ("motion_path", "music_feature_path"):
            path = resolve_contract_path(self.reader.root, row[field], field, row["sample_id"])
            stat = path.stat()
            values.append((str(path), stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns))
        return tuple(values)

    @staticmethod
    def _tensor_bytes(value):
        storage = {}
        def visit(item):
            if isinstance(item, torch.Tensor):
                data = item.untyped_storage()
                storage[data.data_ptr()] = data.nbytes()
            elif isinstance(item, dict):
                for child in item.values():
                    visit(child)
            elif isinstance(item, (tuple, list)):
                for child in item:
                    visit(child)
        visit(value)
        return sum(storage.values())

    def load_aligned_sequence(self, row):
        key, signature = str(row["sample_id"]), self._signature(row)
        with self.lock:
            previous = self.signatures.get(key)
            if previous is not None and previous != signature:
                raise ValueError(f"训练期间原音乐/动作文件发生改变: {key}")
            self.signatures[key] = signature
            if key in self.cache:
                self.cache.move_to_end(key)
                return self.cache[key][0]
        # 不持有全库锁做磁盘/FK工作；并发命中同一首时允许重复读取一次，发布只保留一份。
        if self.prepared is not None:
            directory, manifest = self.prepared
            key_source = source_key({'dataset_id': self.dataset.dataset_name, 'split': self.dataset.split,
                                     'sample_id': str(row['sample_id'])})
            record = manifest['sources'][key_source]
            path = Path(directory)/record['file']
            if key_source not in self.prepared_verified:
                if digest(path) != record['sha256'] or digest(signature[0][0]) != record['motion_sha256'] or digest(signature[1][0]) != record['music_sha256']:
                    raise ValueError('监督特征缓存或原数据SHA不一致')
                self.prepared_verified.add(key_source)
            payload = torch.load(path, map_location='cpu', weights_only=False, mmap=True)
            if payload['source'] != key_source:
                raise ValueError('监督特征缓存source不一致')
            sequence = payload['sequence']
        else:
            sequence = self.reader.load_aligned_sequence(row)
            sequence["_joint_contact"] = BumiClosedLoopStage1Dataset._contact_timeline(self.dataset, sequence)
            sequence["_joint_ground"] = BumiClosedLoopStage1Dataset._ground_supervision(self.dataset, sequence)
        if self._signature(row) != signature:
            raise ValueError(f"读取期间原音乐/动作文件发生改变: {key}")
        size = self._tensor_bytes(sequence)
        with self.lock:
            if key in self.cache:
                self.cache.move_to_end(key)
                return self.cache[key][0]
            if size <= self.max_bytes:
                while self.cache and self.bytes + size > self.max_bytes:
                    _, (_, old_size) = self.cache.popitem(last=False)
                    self.bytes -= old_size
                self.cache[key] = (sequence, size)
                self.bytes += size
        return sequence


class _CachedStage1Dataset(BumiClosedLoopStage1Dataset):
    """保持原get_window实现，仅复用同曲已验证数据、接触和固定地面。"""

    def __init__(self, *, sequence_cache_bytes, **options):
        super().__init__(**options)
        self.reader = _CachedSequenceReader(self.reader, self, sequence_cache_bytes)

    def _contact_timeline(self, sequence):
        return sequence["_joint_contact"]

    def _ground_supervision(self, sequence):
        return dict(sequence["_joint_ground"])

    def _window_features(self, sequence, decision_frame, qpos_halo):
        if '_joint_windows' in sequence and decision_frame % 3 == 0:
            return sequence['_joint_windows'][decision_frame//3]
        return super()._window_features(sequence, decision_frame, qpos_halo)

    def _window_history(self, sequence, decision_frame):
        if '_joint_proprio' not in sequence:
            return super()._window_history(sequence, decision_frame)
        ticks = decision_frame*5-torch.arange(self.history_steps-1, -1, -1)*3
        indices = (ticks.clamp_min(0)//5).clamp_max(decision_frame)
        observations, validity = sequence['_joint_proprio']
        valid = (ticks >= 0) & validity[indices]
        return torch.where(valid[:, None], observations[indices], 0.), valid, ticks.double()/150

    def get_window(self, row_index, start_frame=None):
        result = super().get_window(row_index, start_frame)
        if self.reader.prepared is not None and result['meta']['decision_frame'] % 3 == 0:
            sequence = self.reader.load_aligned_sequence(self.rows[row_index])
            result['_target_body'] = sequence['_joint_target_body'][result['meta']['decision_frame']//3]
        return result


def collate_joint_samples(samples):
    plain = [{k:v for k,v in sample.items() if k != '_target_body'} for sample in samples]
    batch = collate_stage1_training_samples(plain)
    if all('_target_body' in sample for sample in samples):
        batch['_target_body'] = torch.stack([sample['_target_body'] for sample in samples])
    return batch


class SourceCatalog:
    def __init__(self, cfg, split):
        self.split = split
        dc = data_config(cfg)
        self.datasets, self.rows, self.weights = [], [], []
        selected = cfg["data"].get("source_ids")
        cache_bytes = int(float(cfg["data"].get("loading", {}).get("sequence_cache_mb", 0)) * 1024**2)
        if cache_bytes < 0:
            raise ValueError("sequence_cache_mb不能为负数")
        cache_per_dataset = cache_bytes // max(len(dc.datasets[split]), 1)
        for entry in dc.datasets[split].values():
            options = OmegaConf.to_container(entry, resolve=True)
            options.pop("_target_")
            dataset = (_CachedStage1Dataset(sequence_cache_bytes=cache_per_dataset, **options)
                       if cache_per_dataset else BumiClosedLoopStage1Dataset(**options))
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
        prepared = cfg['data'].get('loading', {}).get('supervision_cache')
        if prepared:
            from .supervision_cache import read_manifest
            manifest = read_manifest(prepared, cfg, self)
            for dataset in self.datasets:
                if not isinstance(dataset, _CachedStage1Dataset):
                    raise ValueError('持久监督特征需要非零sequence_cache_mb预算')
                dataset.reader.prepared = (prepared, manifest)

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
        loading = cfg["data"].get("loading", {})
        self.workers = int(loading.get("workers", 0))
        self.prefetch_batches = int(loading.get("prefetch_batches", 0))
        if self.workers < 0 or self.prefetch_batches not in (0, 1):
            raise ValueError("workers必须非负；CPU预取仅支持0或1个完整批")
        self._workers = self._coordinator = self._pending = None
        self._closed = False
        self.conditioner = None
        self.pin_memory = bool(loading.get('pin_memory', False))
        self.last_wait_seconds = 0.
        self.row_probabilities, self.row_indices = [], []
        for dataset, rows in zip(self.catalog.datasets, self.catalog.rows):
            # choice若接收Python列表会每次转换全来源索引；一次缓存数组，采样概率和RNG不变。
            indices = np.asarray(rows, dtype=np.int64)
            self.row_indices.append(indices)
            weights = np.bincount(dataset.idx2meta, minlength=len(dataset.rows))[indices].astype(float)
            self.row_probabilities.append(weights / weights.sum())

    def _plan(self, count):
        # 预取使用克隆RNG，只有next成功交付后才提交新状态。保存时无需序列化队列张量。
        rng = np.random.default_rng()
        rng.bit_generator.state = copy.deepcopy(self.rng.bit_generator.state)
        plan = []
        for _ in range(count):
            index = int(rng.choice(len(self.catalog.datasets), p=self.catalog.probabilities))
            dataset, rows = self.catalog.datasets[index], self.row_indices[index]
            # 沿用原Dataset展开索引的时长采样权重；窗口相位在全曲范围随机选择。
            row_index = int(rng.choice(rows, p=self.row_probabilities[index]))
            frames = int(dataset.rows[row_index]["num_frames"])
            start = int(rng.integers((frames - 1) // 3 + 1)) * 3
            plan.append((index, row_index, start, self.draws + len(plan)))
        return plan, copy.deepcopy(rng.bit_generator.state)

    def _sample(self, item):
        index, row_index, start, draw = item
        sample = self.catalog.datasets[index].get_window(row_index, start_frame=start)
        sample["meta"] = dict(sample["meta"], joint_draw=draw, joint_rank=self.rank,
                              time_conversion="source_frame*20=600Hz_tick; common_grid=60")
        return sample

    def _assemble(self, plan):
        samples = list(self._workers.map(self._sample, plan)) if self._workers else [self._sample(p) for p in plan]
        batch = collate_joint_samples(samples)
        if self.conditioner is not None:
            batch = self.conditioner(batch)
        if self.pin_memory and torch.cuda.is_available():
            batch = {key:value.pin_memory() if isinstance(value,torch.Tensor) else value for key,value in batch.items()}
        return batch

    def _discard_pending(self):
        if self._pending is not None:
            future, _, _ = self._pending
            if not future.cancel():
                future.result()  # 不丢掉后台数据/资产异常，也不让两批并行占满内存。
            self._pending = None

    def next(self, count, device):
        wait_started = time.perf_counter()
        if self._closed or type(count) is not int or count <= 0:
            raise ValueError("数据流必须开放且batch size为正整数")
        if self.workers and self._workers is None:
            self._workers = ThreadPoolExecutor(max_workers=self.workers, thread_name_prefix="joint-data")
        if self.prefetch_batches and self._coordinator is None:
            self._coordinator = ThreadPoolExecutor(max_workers=1, thread_name_prefix="joint-prefetch")
        if self._pending is not None and self._pending[2] != count:
            self._discard_pending()
        if self._pending is None:
            plan, next_rng = self._plan(count)
            batch = self._assemble(plan)
        else:
            future, next_rng, _ = self._pending
            batch = future.result()
            self._pending = None
        output = batch_to_device(batch, device)
        self.last_wait_seconds = time.perf_counter()-wait_started
        self.rng.bit_generator.state = next_rng
        self.draws += count
        if self._coordinator is not None:
            plan, next_rng = self._plan(count)
            self._pending = (self._coordinator.submit(self._assemble, plan), next_rng, count)
        return output

    def state_dict(self):
        return {"rng": copy.deepcopy(self.rng.bit_generator.state), "draws": self.draws,
                "rank": self.rank, "world_size": self.world_size, "identity": self.catalog.identity}

    def load_state_dict(self, value):
        if value["identity"] != self.catalog.identity or (value["rank"], value["world_size"]) != (self.rank, self.world_size):
            raise ValueError("恢复要求同一数据划分、采样合同与rank布局")
        self._discard_pending()
        self.rng.bit_generator.state = copy.deepcopy(value["rng"])
        self.draws = value["draws"]

    def close(self):
        """训练正常退出或异常finally中调用，回收纯CPU预取线程。"""
        try:
            self._discard_pending()
        finally:
            if self._coordinator is not None:
                self._coordinator.shutdown(wait=True, cancel_futures=True)
            if self._workers is not None:
                self._workers.shutdown(wait=True, cancel_futures=True)
            self._closed = True


class SidecarStore:
    def __init__(self, directory, split, expected_assets=None, *, allow_raw=False, sample_contract=None, snapshot=None,
                 cache_bytes=256 * 1024**2):
        self.children = None
        self.lock = RLock()
        if int(cache_bytes) < 0:
            raise ValueError("sidecar缓存字节预算必须非负")
        self.max_cache_bytes, self.cache_bytes = int(cache_bytes), 0
        self.options = dict(split=split, expected_assets=expected_assets, allow_raw=allow_raw,
                            sample_contract=sample_contract, cache_bytes=cache_bytes)
        self.paths = directory
        if isinstance(directory, (list, tuple)):
            self.children = [SidecarStore(path, split, expected_assets, allow_raw=allow_raw,
                                         sample_contract=sample_contract, snapshot=snapshot[i] if snapshot else None,
                                         cache_bytes=cache_bytes) for i, path in enumerate(directory)]
            self.index = dict(self.children[0].index, sources={})
            self.owners = {}
            for child in self.children:
                for key,entry in child.index["sources"].items():
                    if key in self.owners:
                        raise ValueError("增量缓存不得重复保存同一首音乐")
                    self.owners[key] = child
                    self.index["sources"][key] = entry
            self.identity = [child.identity for child in self.children]
            return
        self.directory = Path(directory).resolve()
        # 原子发布可能紧接着换代；解析和身份必须来自同一次读取的字节。
        index_bytes = (self.directory / "index.json").read_bytes()
        self.index = json.loads(index_bytes)
        live_index = self.index
        if snapshot is not None:
            if snapshot["directory"] != str(self.directory):
                raise ValueError("恢复缓存目录不同")
            self._append_only(snapshot["index"], live_index)
            self.index = snapshot["index"]
        schemas = (SIDECAR_SCHEMA, RAW_SIDECAR_SCHEMA) if allow_raw else (SIDECAR_SCHEMA,)
        if self.index["schema"] not in schemas or self.index["split"] != split:
            raise ValueError("sidecar schema/split不匹配，禁止跨split训练")
        if sample_contract is not None and self.index.get("sample_contract") != sample_contract:
            raise ValueError("离线请求缓存的前缀/采样合同不匹配，不能在线重算或读取错位条件")
        if expected_assets is not None and self.index["assets"] != expected_assets:
            raise ValueError("sidecar的SM/WM、stats、FK或物理合同不匹配")
        self.cache, self.verified = OrderedDict(), set()
        self.metrics = dict(cache_hits=0, cache_misses=0, shard_load_count=0, mapped_bytes=0,
                            disk_bytes_read=0, cpu_wait_seconds=0.)
        self.shard_signatures = {}
        self.identity = snapshot["identity"] if snapshot is not None else hashlib.sha256(index_bytes).hexdigest()

    @staticmethod
    def _append_only(old, new):
        for key in ("schema", "split", "assets", "sample_contract"):
            if old.get(key) != new.get(key):
                raise ValueError("sidecar热扩展改变了资产或数据合同")
        for group in ("sources", "shards"):
            if any(new[group].get(key) != value for key, value in old[group].items()):
                raise ValueError("sidecar只能增量添加，不能覆盖已有来源或分片")

    def snapshot(self):
        if self.children is not None:
            return [child.snapshot() for child in self.children]
        return {"directory": str(self.directory), "index": self.index, "identity": self.identity}

    def refresh(self):
        """原子发布index.json后自动发现增量；已有来源和分片必须保持原身份。"""
        fresh = SidecarStore(self.paths, **self.options)
        old_children = self.children if self.children is not None else [self]
        new_children = fresh.children if fresh.children is not None else [fresh]
        for old, new in zip(old_children, new_children):
            self._append_only(old.index, new.index)
        if fresh.identity == self.identity:
            return False
        self.__dict__.update(fresh.__dict__)
        return True

    def episode(self, meta):
        with self.lock:
            return self._episode(meta)

    def _episode(self, meta):
        key = source_key(meta)
        if self.children is not None:
            if key not in self.owners:
                raise PhysicsRejected("missing_real_sidecar")
            return self.owners[key].episode(meta)
        if key not in self.index["sources"]:
            raise PhysicsRejected("missing_real_sidecar")
        entry = self.index["sources"][key]
        shard = entry["shard"]
        started = time.perf_counter()
        stat = (self.directory / shard).stat()
        signature = (stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)
        if shard in self.cache and self.shard_signatures[shard] != signature:
            raise ValueError("训练期间已缓存sidecar分片发生改变")
        if shard not in self.cache:
            self.metrics['cache_misses'] += 1
            if self.shard_signatures.get(shard) != signature:
                if digest(self.directory / shard) != self.index["shards"][shard]:
                    raise ValueError("sidecar分片内容身份已改变")
                self.shard_signatures[shard] = signature
                self.metrics['disk_bytes_read'] += stat.st_size
            payload = torch.load(self.directory / shard, weights_only=False, map_location="cpu", mmap=True)
            self.metrics['shard_load_count'] += 1
            self.metrics['mapped_bytes'] += stat.st_size
            # mmap不等于免反序列化：全库随机抽样下仅保留两个分片会每步重复解析数百次。
            # 按映射文件字节限制LRU；超过单次预算仍可读取，但不常驻。缓存不是checkpoint状态。
            if stat.st_size <= self.max_cache_bytes:
                while self.cache and self.cache_bytes + stat.st_size > self.max_cache_bytes:
                    old_shard, _ = self.cache.popitem(last=False)
                    self.cache_bytes -= self.shard_signatures[old_shard][0]
                self.cache[shard] = payload
                self.cache_bytes += stat.st_size
        else:
            self.metrics['cache_hits'] += 1
            self.cache.move_to_end(shard)
            payload = self.cache[shard]
        episode = payload[entry["episode"]]
        if episode["source_key"] != key or episode["split"] != meta["split"]:
            raise ValueError("sidecar索引指向错误来源")
        if key not in self.verified:
            for field, identity in (("motion_path", "motion_sha256"), ("music_feature_path", "music_sha256")):
                if digest(meta[field]) != episode[identity]:
                    raise ValueError("原音乐/动作与真实sidecar内容不一致")
            self.verified.add(key)
        self.metrics['cpu_wait_seconds'] += time.perf_counter()-started
        return episode

    def cache_metrics(self):
        values = [child.cache_metrics() for child in self.children] if self.children else [dict(self.metrics)]
        names = ('cache_hits','cache_misses','shard_load_count','mapped_bytes','disk_bytes_read','cpu_wait_seconds')
        result = {name:sum(value[name] for value in values) for name in names}
        count = max(1, result['cache_hits']+result['cache_misses'])
        result.update(cache_hit_rate=result['cache_hits']/count, cache_miss_rate=result['cache_misses']/count)
        return result

    def request(self, episode, frame, prefix_frames, device, support_guard=None):
        """按source/time随机索引持久化条件；训练期间禁止调用WM重做前瞻。"""
        from gem.closedloop.mb_control.contracts import ForecastContext
        requests = episode["offline_requests"]
        index = frame // 3
        if frame % 3 or index >= len(requests["frames"]) or int(requests["frames"][index]) != frame:
            raise PhysicsRejected("missing_offline_request")
        if int(requests["prefix_frames"][index]) != prefix_frames:
            raise ValueError("离线已知前缀身份不匹配")
        if not bool(requests["valid"][index]):
            raise PhysicsRejected(requests["reasons"][index])
        lo, hi = requests["state_offsets"][index:index+2].tolist()
        if support_guard is not None:
            support_guard(requests["predicted_states"][lo:hi].to(device))
        return ForecastContext(requests["features"][index:index+1].to(device),
            torch.ones(1, dtype=torch.bool, device=device), frame*20, int(requests["forecast_ticks"][index]))


def slice_batch(batch, index):
    return {k: (v[index:index+1] if isinstance(v, torch.Tensor) else [v[index]] if k == "meta" else 1 if k == "B" else v)
            for k, v in batch.items()}
