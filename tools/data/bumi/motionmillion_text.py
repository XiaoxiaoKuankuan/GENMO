"""为原生UMR MotionMillion动作绑定原始文本和官方划分。

从已完成筛选报告提取原始MotionMillion ID及机器人/人体SHA，流式读取官方文本与
split归档，仅按完整来源ID匹配，禁止用basename或镜像文本代替。SQLite保留原文本
字节、逐文件SHA、按既有MotionMillion规则解析的caption、原始来源与官方split。
不生成文本、不编码T5、不伪造缺失split；缺失覆盖明确计数。目录通过隔离staging
原子发布，绑定新筛选报告时再次核对全部动作双输入SHA和来源，质量判定独立保留。
该模块由已有prepare_bumi_text.py入口与render_bumi_motion.py共用，避免渲染另行猜配。
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import tarfile
import tempfile
from collections import Counter
from pathlib import Path
from urllib.parse import quote

from tools.data.motionmillion.build_motionmillion_genmo import (
    _normalize_split_id,
    _split_from_member,
)
from tools.data.motionmillion.common import (
    atomic_write_json,
    identifier_candidates,
    sha256_file,
)

SCHEMA = "genmo.bumi_motionmillion_text_catalog.v1"


def report_rows(root):
    root = Path(root).resolve(strict=True)
    run = json.loads((root / "run.json").read_text())
    summary = json.loads((root / "quality_summary.json").read_text())
    if (
        run["state"] != "complete"
        or run["partial_scan"]
        or sha256_file(root / "quality_summary.json") != run["summary_sha256"]
        or summary["run_fingerprint"] != run["fingerprint"]
        or run["indexed_records"] != summary["processed_records"]
    ):
        raise ValueError("需要完整且汇总身份匹配的筛选报告")
    count = 0
    for path in sorted((root / "reports").glob("*.jsonl")):
        with path.open() as stream:
            for line in stream:
                row = json.loads(line)
                if row.get("dataset") == "humanml3d" or not row.get("source_motion_id"):
                    raise ValueError("文本索引只接受可追溯的MotionMillion记录")
                count += 1
                yield row
    if count != run["indexed_records"]:
        raise ValueError("报告逐条数量不完整")


def build_catalog(quality_report, texts_archive, split_archive, output):
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError("文本目录已存在，不能覆盖未经核验的结果")
    output.parent.mkdir(parents=True, exist_ok=True)
    archives = {"texts": Path(texts_archive).resolve(), "split": Path(split_archive).resolve()}
    fingerprints = {k: sha256_file(p) for k, p in archives.items()}
    with tempfile.TemporaryDirectory(prefix=f".{output.name}.staging-", dir=output.parent) as temp:
        staged = Path(temp) / "catalog"
        staged.mkdir()
        db = sqlite3.connect(staged / "texts.sqlite")
        wanted = set()
        try:
            db.execute(
                "CREATE TABLE texts (motion_id TEXT PRIMARY KEY, robot_sha TEXT, human_sha TEXT, source_file TEXT, raw_text BLOB, text_sha TEXT, captions_json TEXT, split TEXT)"
            )
            for row in report_rows(quality_report):
                key = row["source_motion_id"]
                if len(key.split("/")) < 2:
                    raise ValueError("需要包含数据集命名空间的完整来源ID")
                if key in wanted:
                    raise ValueError(f"报告原始来源ID重复: {key}")
                wanted.add(key)
                db.execute(
                    "INSERT INTO texts(motion_id,robot_sha,human_sha,source_file) VALUES (?,?,?,?)",
                    (key, row["source_sha256"], row["human_sha256"], row["source_file"]),
                )
            db.commit()
            counts, found = Counter(), set()
            with tarfile.open(archives["texts"], "r|gz") as archive:
                for member in archive:
                    if not member.isfile() or not member.name.lower().endswith(".txt"):
                        continue
                    matches = wanted.intersection(
                        k for k in identifier_candidates(member.name) if "/" in k
                    )
                    if not matches:
                        continue
                    if len(matches) != 1:
                        raise ValueError(f"文本来源ID不唯一: {member.name}")
                    key = next(iter(matches))
                    if key in found or member.size > 4 * 1024 * 1024:
                        raise ValueError(f"重复或异常大的文本: {member.name}")
                    found.add(key)
                    payload = archive.extractfile(member).read()
                    captions = [
                        s.strip() for s in payload.decode("utf-8-sig").splitlines() if s.strip()
                    ]
                    if not captions:
                        counts["empty_text_records"] += 1
                        continue
                    db.execute(
                        "UPDATE texts SET raw_text=?,text_sha=?,captions_json=? WHERE motion_id=?",
                        (
                            payload,
                            hashlib.sha256(payload).hexdigest(),
                            json.dumps(captions, ensure_ascii=False),
                            key,
                        ),
                    )
                    counts["matched_records"] += 1
                    counts["captions"] += len(captions)
                    if counts["matched_records"] % 10000 == 0:
                        db.commit()
                        print(json.dumps(dict(stage="text_index", **counts)), flush=True)
            seen_splits, split_files = set(), 0
            with tarfile.open(archives["split"], "r|gz") as archive:
                for member in archive:
                    split = _split_from_member(member.name)
                    if split is None or not member.isfile():
                        continue
                    split_files += 1
                    text = archive.extractfile(member).read().decode("utf-8-sig")
                    for line in text.splitlines():
                        if not line.strip():
                            continue
                        key = _normalize_split_id(line)
                        if key not in wanted:
                            continue
                        if key in seen_splits:
                            raise ValueError(f"官方split重复或冲突: {key}")
                        seen_splits.add(key)
                        db.execute("UPDATE texts SET split=? WHERE motion_id=?", (split, key))
                        counts["official_" + split] += 1
            if split_files != 3:
                raise ValueError("未找到官方t2m_60_300三个split文件")
            db.commit()
            if db.execute("PRAGMA integrity_check").fetchone()[0] != "ok":
                raise ValueError("文本数据库完整性检查失败")
            missing = [
                r[0]
                for r in db.execute(
                    "SELECT motion_id FROM texts WHERE captions_json IS NULL ORDER BY motion_id"
                )
            ]
            (staged / "missing_text_ids.txt").write_text("".join(s + "\n" for s in missing))
            counts.update(
                input_records=len(wanted),
                missing_records=len(missing),
                official_unlisted=len(wanted) - len(seen_splits),
            )
        finally:
            db.close()
        if any(sha256_file(p) != fingerprints[k] for k, p in archives.items()):
            raise ValueError("索引期间源归档改变")
        metadata = dict(
            schema=SCHEMA,
            **counts,
            database_sha256=sha256_file(staged / "texts.sqlite"),
            archives={k: dict(path=str(p), sha256=fingerprints[k]) for k, p in archives.items()},
            matching="exact source_motion_id plus robot/human SHA; no basename or mirror substitution",
            caption_policy="UTF-8-sig splitlines; strip whitespace and omit blank lines; original bytes retained",
            split_policy="official version1/t2m_60_300 only; missing split is null",
        )
        atomic_write_json(staged / "metadata.json", metadata)
        staged.rename(output)
    return metadata


class TextCatalog:
    """只读文本索引；按源ID及双输入SHA核验，允许显式报告缺失文本。"""

    def __init__(self, root):
        self.root = Path(root).resolve(strict=True)
        self.metadata = json.loads((self.root / "metadata.json").read_text())
        path = self.root / "texts.sqlite"
        if (
            self.metadata["schema"] != SCHEMA
            or sha256_file(path) != self.metadata["database_sha256"]
        ):
            raise ValueError("文本目录格式或数据库SHA不符")
        self.db = sqlite3.connect(f"file:{quote(str(path), safe='/')}?mode=ro", uri=True)

    def close(self):
        self.db.close()

    def lookup(self, row):
        key = row["source_motion_id"]
        found = self.db.execute(
            "SELECT robot_sha,human_sha,source_file,raw_text,text_sha,captions_json,split FROM texts WHERE motion_id=?",
            (key,),
        ).fetchone()
        if found is None or found[:3] != (
            row["source_sha256"],
            row["human_sha256"],
            row["source_file"],
        ):
            raise ValueError(f"文本绑定来源/动作SHA不一致: {key}")
        if found[5] is None:
            return None
        if hashlib.sha256(found[3]).hexdigest() != found[4]:
            raise ValueError(f"文本原始字节SHA不符: {key}")
        return dict(captions=json.loads(found[5]), text_sha256=found[4], official_split=found[6])


def bind_report(quality_report, text_catalog):
    root = Path(quality_report).resolve(strict=True)
    catalog = TextCatalog(text_catalog)
    counts, split_counts = Counter(), Counter()
    try:
        for row in report_rows(root):
            text = catalog.lookup(row)
            counts["records"] += 1
            counts["matched_records" if text else "missing_records"] += 1
            if text:
                counts["captions"] += len(text["captions"])
            if row["training_eligible"]:
                counts["quality_length_eligible"] += 1
                counts["eligible_with_text" if text else "eligible_missing_text"] += 1
                if text:
                    split_counts[text["official_split"] or "unlisted"] += 1
                    counts["eligible_captions"] += len(text["captions"])
        run = json.loads((root / "run.json").read_text())
        result = dict(
            schema="genmo.bumi_motionmillion_text_binding.v1",
            run_fingerprint=run["fingerprint"],
            text_catalog=str(catalog.root),
            database_sha256=catalog.metadata["database_sha256"],
            metadata_sha256=sha256_file(catalog.root / "metadata.json"),
            counts=dict(counts),
            eligible_official_splits=dict(split_counts),
        )
        atomic_write_json(root / "text_binding.json", result)
        return result
    finally:
        catalog.close()
