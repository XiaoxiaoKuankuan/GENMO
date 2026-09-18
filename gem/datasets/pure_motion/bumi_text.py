"""BUMI 文本完整动作 release 与 Dataset。

直接读取 qpos28 分片，不构造人体、相机或音乐条件。动作分片独立于文本特征：
引用旧 MotionMillion T5 时同时验证源 motion/embedding manifest、分片 SHA 和 caption；
原资产不改写。每个 worker 使用有界 LRU，文件身份变化才重新哈希，返回值均为副本。
训练每条完整动作随机取一条 caption，验证固定第一条；尾部补齐到300，length仍为F。
"""

from __future__ import annotations

import hashlib
import json
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, get_worker_info

from gem.runtime.bumi_text_contract import MJCF_SHA256, sha256_file

SCHEMA = "genmo.bumi_text_release.v1"
GROUND = "retargeted_text_floor_zero_v1"


def caption_hash(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class AssetCache:
    """大分片仅首次/变更时哈希；LRU 只保留有限数量的反序列化对象。"""

    def __init__(self, capacity=2):
        self.capacity = int(capacity)
        self.values = OrderedDict()
        self.json_values = {}
        self.verified = {}

    def read(self, path, expected_sha, *, json_file=False):
        path = Path(path).resolve(strict=True)
        st = path.stat()
        identity = (st.st_dev, st.st_ino, st.st_size, st.st_mtime_ns, st.st_ctime_ns)
        key = (str(path), expected_sha)
        if self.verified.get(key) != identity:
            if sha256_file(path) != expected_sha:
                raise ValueError(f"资产 SHA256 不匹配: {path}")
            self.verified[key] = identity
            self.values.pop(key, None)
            self.json_values.pop(key, None)
        if json_file:
            if key not in self.json_values:
                self.json_values[key] = json.loads(path.read_text())
            return self.json_values[key]
        if key not in self.values:
            value = (
                json.loads(path.read_text())
                if json_file
                else torch.load(path, map_location="cpu", weights_only=False)
            )
            self.values[key] = value
        self.values.move_to_end(key)
        while len(self.values) > self.capacity:
            self.values.popitem(last=False)
        return self.values[key]


def resolve_reference(path, base):
    value = Path(path).expanduser()
    return (value if value.is_absolute() else Path(base) / value).resolve()


def read_embedding(ref, caption, cache, base, *, expected_frames=None):
    """读取完整文本对应特征，拒绝把旧50-token特征补零伪装成150-token编码。"""
    if ref.get("caption_sha256") != caption_hash(caption):
        raise ValueError("caption 与 embedding 绑定不一致")
    if ref["format"] == "motionmillion_t5_v1":
        em_path = resolve_reference(ref["embedding_manifest"], base)
        mo_path = resolve_reference(ref["motion_manifest"], base)
        em = cache.read(em_path, ref["embedding_manifest_sha256"], json_file=True)
        mo = cache.read(mo_path, ref["motion_manifest_sha256"], json_file=True)
        if (em.get("schema_version"), mo.get("schema_version")) != (1, 1):
            raise ValueError("原motion/embedding schema错误")
        if not mo.get("build_fingerprint") or mo["build_fingerprint"] != em.get(
            "source_build_fingerprint"
        ):
            raise ValueError("原motion/embedding build fingerprint不一致")
        if ref.get("source_split") != mo.get("split"):
            raise ValueError("文本关联清单必须显式绑定原split")
        if (
            em.get("max_text_tokens") != 150
            or em.get("hidden_dim") != 1024
            or em.get("split") != mo.get("split")
        ):
            raise ValueError("原T5 manifest文本契约或split不符")
        sid, rid, tid = (int(ref[k]) for k in ("shard_id", "record_index", "text_index"))
        if min(sid, rid, tid) < 0 or sid >= len(em["shards"]) or sid >= len(mo["shards"]):
            raise ValueError("embedding 索引越界")
        es, ms = em["shards"][sid], mo["shards"][sid]
        if (
            (es["shard_id"], ms["shard_id"]) != (sid, sid)
            or es["source_motion_sha256"] != ms["sha256"]
            or es["source_motion_path"] != ms["path"]
        ):
            raise ValueError("原motion/embedding分片绑定错误")
        motions = cache.read(mo_path.parent.parent / ms["path"], ms["sha256"])
        embeds = cache.read(em_path.parent.parent / es["path"], es["sha256"])
        if rid >= len(motions) or rid >= len(embeds):
            raise ValueError("embedding record_index 越界")
        motion, record = motions[rid], embeds[rid]
        from tools.data.motionmillion.common import validate_motion_record

        validate_motion_record(motion)
        if motion["split"] != mo["split"] or (
            expected_frames is not None and len(motion["pose"]) != expected_frames
        ):
            raise ValueError("机器人与原完整motion的split/帧数不一致")
        texts = motion["captions"]
        if motion["motion_id"] != record["motion_id"] or str(record["motion_id"]) != str(
            ref["motion_id"]
        ):
            raise ValueError("motion_id 对应关系错误")
        if tid >= len(texts) or texts[tid] != caption:
            raise ValueError("原motion中的caption/text_index不对应")
        from tools.data.motionmillion.common import validate_embedding_record

        validate_embedding_record(record, caption_count=len(texts))
        a, b = map(int, record["offsets"][tid : tid + 2])
        values = record["embeddings"][a:b].float()
        embedding = torch.zeros(150, 1024)
        embedding[: b - a] = values
        mask = torch.arange(150) < b - a
    elif ref["format"] == "bumi_text_t5_v1":
        payload = cache.read(resolve_reference(ref["path"], base), ref["sha256"])
        rid, tid = int(ref.get("record_index", 0)), int(ref["text_index"])
        records = payload if isinstance(payload, list) else [payload]
        if rid < 0 or rid >= len(records):
            raise ValueError("T5 record_index 越界")
        record = records[rid]
        if (record.get("max_text_len"), record.get("encoder")) != (150, "t5-3b"):
            raise ValueError("要求真实 T5-3B 150-token 编码")
        if (
            tid < 0
            or tid >= len(record["captions"])
            or record["captions"][tid] != caption
            or str(record["motion_id"]) != str(ref["motion_id"])
        ):
            raise ValueError("文本特征 caption/motion_id/text_index 不匹配")
        embedding = torch.as_tensor(record["embeddings"][tid]).float().clone()
        mask = torch.as_tensor(record["attention_mask"][tid]).bool().clone()
    else:
        raise ValueError("未知 embedding format")
    if (
        embedding.shape != (150, 1024)
        or mask.shape != (150,)
        or not mask.any()
        or not torch.isfinite(embedding).all()
    ):
        raise ValueError("文本特征形状、mask或有限性错误")
    return embedding.clone(), mask.clone()


def validate_record(record, *, split=None):
    qpos = record.get("qpos")
    if not isinstance(qpos, torch.Tensor) or qpos.ndim != 2 or qpos.shape[1] != 28:
        raise ValueError("qpos 必须为 [F,28] Tensor")
    frames = len(qpos)
    if not 60 <= frames <= 300 or not torch.isfinite(qpos).all() or record.get("frames") != frames:
        raise ValueError("动作真实长度/shape/有限性错误，禁止静默裁剪")
    norm = torch.linalg.vector_norm(qpos[:, 3:7].float(), dim=-1)
    if not torch.allclose(norm, torch.ones_like(norm), atol=1e-3, rtol=0):
        raise ValueError("qpos 需要单位 wxyz 四元数")
    if record.get("split") not in {"train", "val", "test"} or (
        split is not None and split != record["split"]
    ):
        raise ValueError("记录 split 不匹配")
    if record.get("dataset") not in {"motionmillion", "humanml3d"} or not str(
        record.get("motion_id", "")
    ):
        raise ValueError("缺少数据集或motion_id")
    captions = record.get("captions", [])
    if not captions or not all(isinstance(c, str) and c.strip() for c in captions):
        raise ValueError("caption不能为空")
    if len(record.get("embeddings", [])) != len(captions) or len(
        record.get("caption_ids", [])
    ) != len(captions):
        raise ValueError("caption、ID与embedding数量不符")
    if len(set(record["caption_ids"])) != len(captions):
        raise ValueError("重复 caption ID")
    if record.get("ground_semantics") != GROUND or record.get("fps") != 30:
        raise ValueError("需要确认的 Z-up 地面零点、30FPS契约")
    ground = record.get("ground_alignment", {})
    if (
        not ground.get("reference")
        or type(ground.get("applied")) is not bool
        or not np.isfinite(ground.get("offset_z", np.nan))
    ):
        raise ValueError("缺少地面对齐依据/固定偏移记录")
    if not ground["applied"] and ground["offset_z"] != 0:
        raise ValueError("未进行地面对齐时offset_z必须为0")
    for ref in record["embeddings"]:
        if str(ref.get("motion_id", "")) != str(
            record.get("text_source_motion_id", record["motion_id"])
        ):
            raise ValueError("机器人记录与文本来源motion_id绑定错误")
        if (
            ref.get("format") == "motionmillion_t5_v1"
            and ref.get("source_split") != record["split"]
        ):
            raise ValueError("机器人与原文本split不能混用")
    provenance = record.get("provenance", {})
    if (
        not provenance.get("source_id")
        or not provenance.get("retargeter")
        or not provenance.get("retarget_version")
    ):
        raise ValueError("缺少源ID或重定向版本")
    interval = provenance.get("interval_seconds", [])
    if len(interval) != 2 or not np.isfinite(interval).all() or not 0 <= interval[0] < interval[1]:
        raise ValueError("必须记录真实来源片段时间区间")
    if abs((interval[1] - interval[0]) * 30 - frames) > 1.01:
        raise ValueError("来源区间时长与完整F帧不一致")
    if record.get("crop_start", 0) != 0 or record.get("sequence_mode", "full") != "full":
        raise ValueError("完整动作记录不得包含训练时裁剪")
    for key in ("foot_contact", "foot_contact_available"):
        if key in record and (
            tuple(record[key].shape) != (frames, 2) or not torch.isfinite(record[key]).all()
        ):
            raise ValueError(f"{key}必须为有限[F,2]")
    if (
        "foot_contact" in record
        and ((record["foot_contact"] < 0) | (record["foot_contact"] > 1)).any()
    ):
        raise ValueError("接触标签必须为[0,1]")


class BumiTextDataset(Dataset):
    def __init__(
        self,
        root,
        split,
        dataset=None,
        sequence_mode="full",
        pad_to_frames=300,
        caption_sampling="random",
        random_seed=20260909,
        shard_cache_size=2,
        **unused,
    ):
        if unused:
            raise TypeError(f"未知 Dataset 参数: {sorted(unused)}")
        if (sequence_mode, pad_to_frames) != ("full", 300) or caption_sampling not in {
            "random",
            "first",
        }:
            raise ValueError("BUMI 文本要求完整动作/pad300和独立caption策略")
        self.root = Path(root).expanduser().resolve()
        manifest_path = self.root / "manifests" / f"{split}.json"
        self.manifest = json.loads(manifest_path.read_text())
        manifest = self.manifest
        if manifest.get("schema") != SCHEMA or manifest.get("split") != split:
            raise ValueError("release schema/split错误")
        if (
            manifest.get("fps"),
            manifest.get("qpos_dim"),
            manifest.get("quaternion_convention"),
            manifest.get("coordinate_system"),
        ) != (30, 28, "wxyz", "z_up"):
            raise ValueError("release机器人数据契约错误")
        if (
            manifest.get("source_mjcf_sha256") != MJCF_SHA256
            or len(set(manifest.get("joint_names", []))) != 21
        ):
            raise ValueError("release不是fe934/21关节")
        kin_path = resolve_reference(manifest["kinematics"]["path"], self.root)
        cache = AssetCache(shard_cache_size)
        kin = cache.read(kin_path, manifest["kinematics"]["sha256"], json_file=True)
        if (
            kin["joint_order"] != manifest["joint_names"]
            or kin["source_mjcf_sha256"] != manifest["source_mjcf_sha256"]
        ):
            raise ValueError("release运动学绑定错误")
        self.cache = cache
        self.split, self.dataset = split, dataset
        self.caption_sampling, self.pad_to_frames = caption_sampling, pad_to_frames
        self.sequence_mode, self.random_seed = sequence_mode, random_seed
        self._rng, self._worker = None, None
        self.index = []
        seen = set()
        for sid, shard in enumerate(manifest["shards"]):
            if shard["shard_id"] != sid or len(shard["records"]) != shard["record_count"]:
                raise ValueError("分片顺序/索引数不一致")
            for rid, row in enumerate(shard["records"]):
                identity = (row["dataset"], row["motion_id"])
                if identity in seen or row["record_index"] != rid or not 60 <= row["frames"] <= 300:
                    raise ValueError("重复记录、非法长度或错误索引")
                seen.add(identity)
                if dataset is None or row["dataset"] == dataset:
                    self.index.append((sid, rid, row))
        if not self.index:
            raise ValueError(f"{split}/{dataset} 没有可用完整动作")
        self.data_identity = {
            "schema": SCHEMA,
            "manifest_sha256": sha256_file(manifest_path),
            "dataset": dataset,
            "kinematics_sha256": manifest["kinematics"]["sha256"],
        }

    def __len__(self):
        return len(self.index)

    def __getstate__(self):
        result = self.__dict__.copy()
        result["cache"] = AssetCache(self.cache.capacity)
        result["_rng"], result["_worker"] = None, None
        return result

    def sample_shard_ids(self):
        return np.asarray([row[0] for row in self.index], dtype=np.int64)

    def _get_rng(self):
        worker = get_worker_info()
        identity = None if worker is None else (worker.id, worker.seed)
        if self._rng is None or self._worker != identity:
            seed = self.random_seed if worker is None else worker.seed % (2**32)
            self._rng, self._worker = np.random.RandomState(seed), identity
        return self._rng

    def read_record(self, index):
        sid, rid, row = self.index[index]
        shard = self.manifest["shards"][sid]
        records = self.cache.read(self.root / shard["path"], shard["sha256"])
        if len(records) != shard["record_count"]:
            raise ValueError("实际分片记录数不符")
        record = records[rid]
        validate_record(record, split=self.split)
        if any(record[k] != row[k] for k in ("frames", "dataset", "motion_id")):
            raise ValueError("索引与实际记录不符")
        return record

    def __getitem__(self, index):
        record = self.read_record(index)
        frames = record["frames"]
        tid = (
            int(self._get_rng().randint(len(record["captions"])))
            if self.caption_sampling == "random"
            else 0
        )
        caption = record["captions"][tid]
        embedding, text_mask = read_embedding(
            record["embeddings"][tid], caption, self.cache, self.root, expected_frames=frames
        )
        qpos = record["qpos"].float().clone()
        qpos = torch.cat((qpos, qpos[-1:].expand(300 - frames, -1)))
        valid = torch.arange(300) < frames
        contact = torch.zeros(300, 2)
        available = torch.zeros(300, 2, dtype=torch.bool)
        if "foot_contact" in record:
            contact[:frames] = record["foot_contact"]
            available[:frames] = record.get(
                "foot_contact_available", torch.ones(frames, 2, dtype=torch.bool)
            )
        return dict(
            qpos=qpos,
            length=frames,
            valid_length=frames,
            caption=caption,
            has_text=True,
            text_embed=embedding,
            text_attention_mask=text_mask,
            foot_contact=contact,
            foot_contact_available=available,
            mask={"valid": valid},
            meta=dict(
                dataset_id=record["dataset"],
                motion_id=record["motion_id"],
                source_frames=frames,
                valid_length=frames,
                crop_start=0,
                sequence_mode="full",
                pad_to_frames=300,
                text_index=tid,
                caption_id=record["caption_ids"][tid],
                ground_semantics=GROUND,
            ),
        )
