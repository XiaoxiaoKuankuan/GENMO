#!/usr/bin/env python3
"""将 GENMO 的 HumanML3D 动作汇总 PTH 导出为 UMR 逐动作 NPZ 交付目录。

输入必须是本仓库构建器生成的可信训练数据：30 Hz、AMASS Z-up、SMPL-X
身体轴角 pose[T,66]、米制 trans[T,3]、固定 beta[10]、gender 和 text_data。
本工具只拆分容器与重命名字段，不旋转坐标、不重采样、不贴地或修改动作数值。
原动作、镜像、文本子片段分别按原 ID 保存，全部文本写入独立 JSON 索引。

每条 NPZ 写入后立即回读，与原始三个数值数组逐元素精确比较，并校验设备无关的
帧率、坐标系和性别字段。清单记录每条帧数和 SHA256；文本和清单也回读核验。
所有文件先在目标父目录下的独立临时目录生成，成功后整体重命名为正式目录；
失败自动清理临时目录，已有目标一律拒绝覆盖。此验证仅证明交付数据完整性，
不代表已经完成机器人重定向、动作回放、动力学或硬件验证。
"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile

import numpy as np
import torch


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, data) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def export_dataset(source: Path, output: Path, expected_sha256: str | None = None) -> dict:
    source = source.expanduser().resolve(strict=True)
    output = output.expanduser().absolute()
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"拒绝覆盖已有输出：{output}")
    source_stat = source.stat()
    source_sha = sha256(source)
    if expected_sha256 and source_sha != expected_sha256:
        raise ValueError(f"源文件 SHA256 不匹配：{source_sha}")
    data = torch.load(source, map_location="cpu", weights_only=False, mmap=True)
    if not isinstance(data, dict) or not data:
        raise ValueError("输入必须是非空动作字典")
    if any(not isinstance(key, str) or not re.fullmatch(r"[A-Za-z0-9_-]+", key) for key in data):
        raise ValueError("动作 ID 包含不安全的文件名字符")

    output.parent.mkdir(parents=True, exist_ok=True)
    texts, manifest, genders = {}, [], Counter()
    frame_lengths = []
    with tempfile.TemporaryDirectory(prefix=f".{output.name}.staging-", dir=output.parent) as temp:
        stage = Path(temp)
        motions = stage / "motions"
        motions.mkdir()
        for index, (key, record) in enumerate(data.items(), 1):
            arrays = {}
            for field in ("pose", "trans", "beta"):
                tensor = record[field]
                if not isinstance(tensor, torch.Tensor) or tensor.dtype != torch.float32:
                    raise ValueError(f"{key}/{field} 必须为 float32 Tensor")
                array = tensor.detach().cpu().numpy()
                if not np.isfinite(array).all():
                    raise ValueError(f"{key}/{field} 含非有限值")
                arrays[field] = array
            pose, trans, beta = (arrays[field] for field in ("pose", "trans", "beta"))
            if pose.ndim != 2 or pose.shape[1] != 66 or len(pose) == 0:
                raise ValueError(f"{key} 姿态形状错误：{pose.shape}")
            if trans.shape != (len(pose), 3) or beta.shape != (10,):
                raise ValueError(f"{key} 位移或体型形状错误")
            gender = record["gender"]
            if gender not in ("neutral", "male", "female"):
                raise ValueError(f"{key} 性别字段错误：{gender}")
            captions = record["text_data"]
            if not isinstance(captions, list) or not captions:
                raise ValueError(f"{key} 缺少文本")
            for item in captions:
                if not isinstance(item.get("caption"), str) or not item["caption"].strip():
                    raise ValueError(f"{key} 存在空文本")
                if not isinstance(item.get("tokens"), list) or not item["tokens"]:
                    raise ValueError(f"{key} 缺少文本 tokens")

            fields = {
                "root_orient": pose[:, :3], "pose_body": pose[:, 3:66],
                "trans": trans, "betas": beta, "gender": np.asarray(gender),
                "mocap_frame_rate": np.asarray(30.0, dtype=np.float32),
                "output_up": np.asarray("z"),
            }
            path = motions / f"{key}.npz"
            np.savez_compressed(path, **fields)
            with np.load(path, allow_pickle=False) as loaded:
                if set(loaded.files) != set(fields):
                    raise ValueError(f"{key} 回读字段不一致")
                for field, expected in fields.items():
                    if loaded[field].dtype != expected.dtype or not np.array_equal(loaded[field], expected):
                        raise ValueError(f"{key}/{field} 回读数值或类型不一致")
            texts[key] = captions
            genders[gender] += 1
            frame_lengths.append(len(pose))
            manifest.append({
                "motion_id": key, "file": f"motions/{key}.npz", "frames": len(pose),
                "caption_count": len(captions), "bytes": path.stat().st_size,
                "sha256": sha256(path),
            })
            if index % 1000 == 0 or index == len(data):
                print(f"已导出并精确回读 {index}/{len(data)} 条", flush=True)

        write_json(stage / "texts.json", texts)
        if json.loads((stage / "texts.json").read_text(encoding="utf-8")) != texts:
            raise ValueError("文本索引回读不一致")
        manifest_path = stage / "manifest.jsonl"
        manifest_path.write_text("".join(json.dumps(row) + "\n" for row in manifest), encoding="utf-8")
        if [json.loads(line) for line in manifest_path.read_text().splitlines()] != manifest:
            raise ValueError("动作清单回读不一致")
        if {p.stem for p in motions.iterdir()} != set(data):
            raise ValueError("输出动作 ID 集合与源数据不一致")
        if source.stat().st_size != source_stat.st_size or sha256(source) != source_sha:
            raise ValueError("导出期间源 PTH 内容发生变化")
        metadata = {
            "schema": "humanml3d_umr_npz_v1",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "source_file": str(source), "source_bytes": source_stat.st_size,
            "source_sha256": source_sha, "output_directory": str(output),
            "dataset": "GENMO HumanML3D training derivative",
            "body_model": "SMPL-X", "fps": 30.0,
            "coordinate_system": "right_handed_z_up", "translation_unit": "meter",
            "rotation_representation": "axis_angle", "rotation_unit": "radian",
            "motion_count": len(data), "total_frames": sum(frame_lengths),
            "min_frames": min(frame_lengths), "max_frames": max(frame_lengths),
            "caption_count": sum(len(value) for value in texts.values()),
            "genders": dict(genders), "includes_mirrors_and_subclips": True,
            "validation": "all_npz_fields_exact_roundtrip_and_all_texts_verified",
            "robot_retargeting_performed": False,
        }
        write_json(stage / "metadata.json", metadata)
        (stage / "README.md").write_text(
            "# HumanML3D UMR 重定向输入\n\n"
            "motions/ 中每个 NPZ 对应一条原动作、镜像或文本子片段，文件名保留原 ID。\n"
            "右手 Z-up，30 Hz，位移单位米，轴角单位弧度；无需旋转坐标或重采样。\n"
            "root_orient[T,3]、pose_body[T,63]、trans[T,3]、betas[10]，"
            "另含 gender、mocap_frame_rate=30、output_up=z。\n"
            "使用 SMPL-X 模型和目标机器人 UMR 配置；不包含手指/面部动作参数。\n"
            "texts.json 按动作 ID 保存所有原始 caption/tokens；manifest.jsonl 保存"
            "每条帧数、文件大小和 SHA256；metadata.json 记录来源和数据契约。\n"
            "这是训练集派生版本，不能当作官方完整 train/val/test。"
            "导出不做地面对齐，接收方应检查 UMR 地面配置并先做少量回放。\n"
            "已逐条回读核对全部字段和文本，尚未执行全量机器人重定向。\n\n"
            "可在本目录执行 `sha256sum -c SHA256SUMS` 验证所有交付文件。\n",
            encoding="utf-8",
        )
        checksums = [f"{row['sha256']}  {row['file']}\n" for row in manifest]
        for name in ("texts.json", "manifest.jsonl", "metadata.json", "README.md"):
            checksums.append(f"{sha256(stage / name)}  {name}\n")
        (stage / "SHA256SUMS").write_text("".join(checksums), encoding="utf-8")
        if output.exists() or output.is_symlink():
            raise FileExistsError(f"发布前发现目标已存在：{output}")
        os.rename(stage, output)
    print(json.dumps(metadata, ensure_ascii=False, indent=2), flush=True)
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="可信的 GENMO HumanML3D PTH")
    parser.add_argument("--output", type=Path, required=True, help="必须尚不存在的交付目录")
    parser.add_argument("--expected-sha256", help="可选：拒绝来源指纹不符的数据")
    args = parser.parse_args()
    torch.set_num_threads(1)
    export_dataset(args.input, args.output, args.expected_sha256)


if __name__ == "__main__":
    main()
