"""为56万级 MotionMillion UMR 输出提供可恢复的全量预处理筛选。

入口由已有 prepare_bumi_text.py 的 filter-umr 子命令提供。逐目录核对 batch_summary
与实际文件集合，以相对路径建立唯一身份，检查原生 qpos、具名关节、源人体数值、
30Hz完整时间线和 Y-up 来源；不把源坐标再旋转到已是Z-up的机器人输出上。

任务目录使用进程锁与SQLite事务，常驻worker通过有限队列计算，不把全库指标装入
内存。续跑仍重新计算动作和源人体SHA；配置、资产、代码或输入清单变化时拒绝复用。
每条保存PASS/REVIEW/REJECT/INVALID/ERROR、指标和异常帧区间。报告分片和训练候选
JSONL按数据库流式原子发布，候选仅包含完整60..300帧PASS，不裁剪、不移动源文件。

训练构建器可以用完整报告验证原生UMR输入，重新核对源文件指纹后按标准顺序读取，
沿用已有完整文本/T5配对验证。质量报告本身不创造文本、split或embedding对应关系。
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import multiprocessing
import os
import sqlite3
import time
import zipfile
from collections import Counter, defaultdict
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from pathlib import Path
from urllib.parse import quote

import numpy as np
import torch

from gem.robots.bumi.kinematics import sha256_file
from gem.robots.bumi.legacy_motion import _NumpyCompatibleUnpickler
from tools.data.bumi.umr_text_quality import QualityEngine, load_rules, verify_asset_files
from tools.data.motionmillion.common import mirror_base_id

ROOT = Path(__file__).resolve().parents[3]
SCHEMA = "genmo.bumi_umr_text_quality.v1"
DEFAULT_CONFIG = ROOT / "configs/bumi/quality_filter_umr_text_30hz_v1.yaml"
DEFAULT_KINEMATICS = ROOT / "configs/bumi/bumi_kinematics_robot_retargeter_fe934_v1.json"
_ENGINE = None
_PATHS = None


class InputContractError(ValueError):
    """已确认的源文件/生成合同问题，区别于程序或环境执行错误。"""


def check(condition, message):
    if not condition:
        raise InputContractError(message)


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_suffix(path.suffix + ".pending")
    with pending.open("w") as out:
        json.dump(value, out, ensure_ascii=False, indent=2, allow_nan=False)
        out.write("\n")
        out.flush()
        os.fsync(out.fileno())
    pending.replace(path)


def _scalar(arrays, name):
    value = arrays[name]
    check(value.size == 1, f"{name}必须为单值")
    return value.item()


def read_npz(path, *, object_names=False):
    """读取每个数组并触发ZIP CRC校验，只兼容可信UMR的关节名称object数组。"""
    result = {}
    try:
        with np.load(path, allow_pickle=False) as archive:
            check(len(archive.files) == len(set(archive.files)), "NPZ存在重复字段")
            for name in archive.files:
                if name == "robot_joint_names" and object_names:
                    with (
                        zipfile.ZipFile(path) as zip_archive,
                        zip_archive.open(name + ".npy") as stream,
                    ):
                        version = np.lib.format.read_magic(stream)
                        if version == (1, 0):
                            shape, _, dtype = np.lib.format.read_array_header_1_0(stream)
                        elif version == (2, 0):
                            shape, _, dtype = np.lib.format.read_array_header_2_0(stream)
                        else:
                            raise InputContractError("不支持的关节名称NPY版本")
                        check(shape == (21,), "关节名称数组必须为[21]")
                        value = (
                            _NumpyCompatibleUnpickler(stream).load()
                            if dtype.hasobject
                            else archive[name]
                        )
                        # object reader后读取余下字节，完成底层ZIP CRC校验。
                        stream.read()
                else:
                    value = archive[name]
                check(isinstance(value, np.ndarray), f"{name}必须是数组")
                if value.dtype.kind in "fciub":
                    check(np.isfinite(value).all(), f"{name}含NaN/Inf")
                elif value.dtype.hasobject:
                    check(
                        name == "robot_joint_names"
                        and object_names
                        and all(isinstance(x, str) for x in value.tolist()),
                        "未知object字段",
                    )
                result[name] = value
    except (ValueError, KeyError, EOFError, zipfile.BadZipFile) as exc:
        raise InputContractError(f"NPZ损坏或字段非法: {exc}") from exc
    return result


def load_umr(row, paths, engine):
    """严格对应summary、qpos、原SMPL-X时间线，并保留原始来源ID。"""
    path = Path(paths["input_root"]) / row["relative_path"]
    source = Path(row["human_path"])
    z = read_npz(path, object_names=True)
    required = {
        "qpos",
        "fps",
        "frame_ids",
        "robot_xml",
        "robot_name",
        "robot_joint_names",
        "source_data",
        "source_sequence_key",
        "source_format",
        "smpl_scale",
        "ground_z",
        "zero_source_finger_pose",
    }
    check(required <= set(z), f"缺少UMR字段: {sorted(required - set(z))}")
    qpos = z["qpos"]
    check(
        qpos.dtype == np.float32 and qpos.ndim == 2 and qpos.shape[1] == 28,
        "qpos必须为float32[T,28]",
    )
    n = len(qpos)
    check(n >= engine.config.minimum_frames, "动作不足数值检查最低帧数")
    check(float(_scalar(z, "fps")) == 30.0, "输出不是30Hz")
    check(
        np.issubdtype(z["frame_ids"].dtype, np.integer)
        and np.array_equal(z["frame_ids"], np.arange(n)),
        "非完整连续frame_ids",
    )
    check(
        _scalar(z, "robot_name") == "bumi3" and _scalar(z, "source_format") == "smplx_npz",
        "机器人或输出来源格式错误",
    )
    check(Path(_scalar(z, "robot_xml")).resolve() == engine.xml, "逐条XML路径不匹配")
    check(
        Path(_scalar(z, "source_data")).resolve() == source.resolve(), "summary与源人体路径不一致"
    )
    key = str(_scalar(z, "source_sequence_key"))
    check(key == source.stem and path.stem == key + "_bumi3", "动作身份与源文件名不一致")
    names = z["robot_joint_names"].tolist()
    check(
        len(names) == 21 and len(set(names)) == 21 and set(names) == set(engine.kin.joint_order),
        "关节名称不是唯一完整21关节",
    )
    check(
        np.max(np.abs(np.linalg.norm(qpos[:, 3:7].astype(np.float64), axis=1) - 1))
        <= engine.config.quaternion_norm_error_max,
        "根四元数范数不合法",
    )
    scale, ground = float(_scalar(z, "smpl_scale")), float(_scalar(z, "ground_z"))
    check(np.isfinite([scale, ground]).all() and scale > 0, "人体比例/地面元数据非法")
    check(z["zero_source_finger_pose"].dtype == np.bool_, "手指姿态标记必须为bool")
    _scalar(z, "zero_source_finger_pose")
    human = read_npz(source)
    keys = {"pose_aa", "trans", "fps", "output_up", "source_format", "source_file"}
    check(keys <= set(human), f"源人体字段缺失: {sorted(keys - set(human))}")
    check(
        human["pose_aa"].shape == (n, 66) and human["trans"].shape == (n, 3), "源人体帧数/维度不同"
    )
    for key, shape in {
        "pose_aa": (n, 66),
        "trans": (n, 3),
        "poses": (n, 22, 3),
        "root_orient": (n, 3),
        "pose_body": (n, 63),
        "trans_orig": (n, 3),
    }.items():
        if key in human:
            check(
                human[key].shape == shape and human[key].dtype.kind == "f",
                f"源人体{key}形状或类型不符",
            )
    check(float(_scalar(human, "fps")) == 30.0, "源人体不是30Hz")
    for fps_key in ("mocap_framerate", "mocap_frame_rate"):
        if fps_key in human:
            check(float(_scalar(human, fps_key)) == 30.0, "源人体FPS字段矛盾")
    check(_scalar(human, "output_up") == engine.rules["source_up"], "源人体up轴契约不符")
    check(_scalar(human, "source_format") == engine.rules["source_format"], "非MotionMillion来源")
    source_file = str(_scalar(human, "source_file"))
    marker = engine.rules["source_id_marker"] + "/"
    check(source_file.count(marker) == 1, "原始MotionMillion来源路径缺少唯一身份锚点")
    original = Path(source_file.split(marker, 1)[1])
    check(
        not original.is_absolute() and ".." not in original.parts and original.suffix == ".npy",
        "原始MotionMillion ID非法",
    )
    source_id = original.with_suffix("").as_posix()
    ordered = np.concatenate(
        (qpos[:, :7], qpos[:, [names.index(name) + 7 for name in engine.kin.joint_order]]), axis=1
    )
    return ordered, dict(
        frames=n,
        fps=30,
        duration_seconds=n / 30,
        source_motion_id=source_id,
        canonical_source_id=mirror_base_id(source_id),
        source_file=source_file,
        source_sequence_key=key,
        source_up="y",
        output_up="z",
        smpl_scale=scale,
        source_ground_z=ground,
        ground_height_m=engine.rules["ground_height_m"],
    )


def init_worker(paths):
    global _ENGINE, _PATHS
    torch.set_num_threads(1)
    _PATHS = paths
    _ENGINE = QualityEngine(
        load_rules(paths["config"]),
        paths["robot_xml"],
        paths["kinematics"],
        paths["retarget_config"],
        paths["batch_config"],
    )


def evaluate_row(task):
    row, cached = task
    base = dict(
        row,
        schema=SCHEMA,
        status="ERROR",
        reason_codes=[],
        metrics={},
        frames=0,
        duration_seconds=0,
        training_eligible=False,
        training_exclusion_reasons=[],
    )
    try:
        path = Path(_PATHS["input_root"]) / row["relative_path"]
        human = Path(row["human_path"])
        base["source_bytes"] = path.stat().st_size if path.is_file() else 0
        base["source_sha256"] = sha256_file(path) if path.is_file() else None
        base["human_sha256"] = sha256_file(human) if human.is_file() else None
        if (
            cached
            and cached["status"] != "ERROR"
            and all(
                cached.get(k) == base.get(k)
                for k in ("source_sha256", "human_sha256", "upstream_status")
            )
        ):
            return cached, True
        check(row["upstream_status"] == "ok", "上游未成功或输出不在summary内")
        check(base["source_sha256"] and base["human_sha256"], "输出或源人体文件缺失")
        qpos, meta = load_umr(row, _PATHS, _ENGINE)
        base.update(meta)
        base.update(_ENGINE.evaluate(qpos))
        lo, hi = _ENGINE.rules["training_frames"]
        if not lo <= len(qpos) <= hi:
            base["training_exclusion_reasons"].append("LENGTH_OUTSIDE_60_300")
        base["training_eligible"] = (
            base["status"] == "PASS" and not base["training_exclusion_reasons"]
        )
        # 检测检查过程中源文件被替换，禁止绑定不属于本次计算的数据指纹。
        if sha256_file(path) != base["source_sha256"] or sha256_file(human) != base["human_sha256"]:
            raise RuntimeError("源文件在检查期间改变，请暂停生产者并重新运行")
    except (InputContractError, FileNotFoundError) as exc:
        base.update(status="INVALID", reason_codes=["INPUT_CONTRACT"], error=str(exc))
    except Exception as exc:
        base.update(
            status="ERROR",
            training_eligible=False,
            reason_codes=["EXECUTION_ERROR"],
            error=f"{type(exc).__name__}: {exc}",
        )
    # 非有限报告也应显式失败，不能把NaN写入JSON供下游误读。
    json.dumps(base, allow_nan=False)
    return base, False


def _summaries(root, folders):
    paths = sorted(root.glob("folder*/bumi3/batch_summary.json"))
    if folders:
        requested = set(folders)
        paths = [p for p in paths if p.parent.parent.name in requested]
        if {p.parent.parent.name for p in paths} != requested:
            raise ValueError("指定folder缺少batch_summary")
    if not paths:
        raise ValueError("没有找到folder*/bumi3/batch_summary.json")
    if not folders:
        missing = [
            p
            for p in root.glob("folder*/bumi3")
            if p.is_dir() and not (p / "batch_summary.json").is_file()
        ]
        if missing:
            raise ValueError(f"输出目录缺少batch_summary: {missing}")
    return paths


def index_inputs(db, paths, summaries):
    """一次仅持有一个上游分片，精确识别缺失输出和未登记输出。"""
    root, source_root = Path(paths["input_root"]), Path(paths["source_root"])
    with db:
        db.execute("DELETE FROM inputs")
        for summary in summaries:
            payload = json.loads(summary.read_text())
            folder = summary.parent.parent.name
            actual = {p.name for p in summary.parent.iterdir() if p.suffix == ".npz"}
            seen = set()
            for item in payload["results"]:
                path, human = Path(item["out"]).resolve(), Path(item["motion"]).resolve()
                if path.parent != summary.parent or not human.is_relative_to(source_root):
                    raise ValueError("summary路径越过声明的数据根目录")
                relative = path.relative_to(root).as_posix()
                if path.name in seen:
                    raise ValueError(f"summary重复输出身份: {relative}")
                seen.add(path.name)
                db.execute(
                    "INSERT INTO inputs VALUES (?,?,?,?)",
                    (relative, folder, str(human), item["status"]),
                )
            for name in sorted(actual - seen):
                path = summary.parent / name
                expected = source_root / folder / (path.stem.removesuffix("_bumi3") + ".npz")
                db.execute(
                    "INSERT INTO inputs VALUES (?,?,?,?)",
                    (path.relative_to(root).as_posix(), folder, str(expected), "unrecorded"),
                )
            print(
                json.dumps(
                    dict(
                        stage="index",
                        folder=folder,
                        summary_records=len(seen),
                        actual_npz=len(actual),
                        missing=len(seen - actual),
                        unrecorded=len(actual - seen),
                    ),
                    ensure_ascii=False,
                ),
                flush=True,
            )


def _tasks(db, limit):
    query = "SELECT i.*, r.payload FROM inputs i LEFT JOIN results r USING(relative_path) ORDER BY i.relative_path"
    cursor = db.execute(query + (" LIMIT ?" if limit else ""), (limit,) if limit else ())
    for relative, folder, human, status, cached in cursor:
        yield (
            dict(relative_path=relative, folder=folder, human_path=human, upstream_status=status),
            json.loads(cached) if cached else None,
        )


def bounded_results(tasks, paths, workers):
    """固定最多2*workers个待执行任务，避免Executor.map提前提交整库。"""
    if workers == 1:
        init_worker(paths)
        for task in tasks:
            yield evaluate_row(task)
        return
    with ProcessPoolExecutor(
        max_workers=workers,
        mp_context=multiprocessing.get_context("spawn"),
        initializer=init_worker,
        initargs=(paths,),
    ) as pool:
        pending, exhausted = set(), False
        while pending or not exhausted:
            while len(pending) < workers * 2 and not exhausted:
                try:
                    pending.add(pool.submit(evaluate_row, next(tasks)))
                except StopIteration:
                    exhausted = True
            if not pending:
                break
            ready, pending = wait(pending, return_when=FIRST_COMPLETED)
            for future in ready:
                yield future.result()


def publish_reports(db, output, run, limit):
    counts, reasons, length_reasons = Counter(), Counter(), Counter()
    by_folder, by_source = defaultdict(Counter), defaultdict(Counter)
    frames = Counter()
    total_bytes = retained_bytes = eligible_bytes = 0
    (output / "reports").mkdir(exist_ok=True)
    handles = {}
    candidates = output / "train_candidates.jsonl.pending"
    query = "SELECT r.payload FROM inputs i JOIN results r USING(relative_path) ORDER BY i.relative_path"
    try:
        with candidates.open("w") as accepted:
            for (payload,) in db.execute(
                query + (" LIMIT ?" if limit else ""), (limit,) if limit else ()
            ):
                row = json.loads(payload)
                folder, status = row["folder"], row["status"]
                if folder not in handles:
                    handles[folder] = (output / "reports" / f"{folder}.jsonl.pending").open("w")
                handles[folder].write(payload + "\n")
                counts[status] += 1
                by_folder[folder][status] += 1
                by_source[row.get("source_motion_id", "unknown").split("/", 1)[0]][status] += 1
                reasons.update(row["reason_codes"])
                length_reasons.update(row["training_exclusion_reasons"])
                frames[status] += row["frames"]
                total_bytes += row.get("source_bytes", 0)
                if status == "PASS":
                    retained_bytes += row.get("source_bytes", 0)
                if row["training_eligible"]:
                    accepted.write(payload + "\n")
                    counts["TRAIN_ELIGIBLE"] += 1
                    frames["TRAIN_ELIGIBLE"] += row["frames"]
                    eligible_bytes += row.get("source_bytes", 0)
            accepted.flush()
            os.fsync(accepted.fileno())
        for handle in handles.values():
            handle.flush()
            os.fsync(handle.fileno())
            handle.close()
        for folder in handles:
            (output / "reports" / f"{folder}.jsonl.pending").replace(
                output / "reports" / f"{folder}.jsonl"
            )
        candidates.replace(output / "train_candidates.jsonl")
    finally:
        for handle in handles.values():
            handle.close()
    total = sum(counts[k] for k in ("PASS", "REVIEW", "REJECT", "INVALID", "ERROR"))
    measured_frames = sum(frames[k] for k in ("PASS", "REVIEW", "REJECT", "INVALID", "ERROR"))
    summary = dict(
        schema=SCHEMA,
        run_fingerprint=run["fingerprint"],
        partial_scan=run["partial_scan"],
        indexed_records=run["indexed_records"],
        processed_records=total,
        status_counts=dict(counts),
        reason_counts=dict(reasons),
        training_exclusion_counts=dict(length_reasons),
        frames_by_status=dict(frames),
        hours_by_status={k: v / 30 / 3600 for k, v in frames.items()},
        by_folder=dict(by_folder),
        by_source=dict(by_source),
        source_bytes=total_bytes,
        pass_source_bytes=retained_bytes,
        eligible_source_bytes=eligible_bytes,
        pass_record_fraction=counts["PASS"] / total if total else 0,
        eligible_record_fraction=counts["TRAIN_ELIGIBLE"] / total if total else 0,
        eligible_frame_fraction=frames["TRAIN_ELIGIBLE"] / measured_frames
        if measured_frames
        else 0,
        duration_scope="仅已成功读取时间线的动作；INVALID文件可能无法统计时长",
        candidate_manifest_sha256=sha256_file(output / "train_candidates.jsonl"),
        full_sequence=True,
        crop_count=0,
        interpretation="离线数值/运动学门禁；候选尚需文本、T5、split和来源去重核验",
    )
    write_json(output / "quality_summary.json", summary)
    return summary


def run_filter(args):
    paths = {
        key: str(Path(getattr(args, key)).resolve(strict=True))
        for key in (
            "input_root",
            "source_root",
            "config",
            "robot_xml",
            "kinematics",
            "retarget_config",
            "asset_manifest",
        )
    }
    paths["batch_config"] = (
        str(args.batch_config.resolve(strict=True)) if args.batch_config else None
    )
    output = args.output.resolve()
    for root in (Path(paths["input_root"]), Path(paths["source_root"])):
        if output == root or output.is_relative_to(root) or root.is_relative_to(output):
            raise ValueError("报告目录必须与输入目录分离")
    if args.workers < 1 or (args.limit is not None and args.limit < 1):
        raise ValueError("workers/limit必须为正")
    rules = load_rules(paths["config"])
    assets = verify_asset_files(
        paths["robot_xml"], paths["kinematics"], paths["asset_manifest"], rules
    )
    # 主进程提前校验模型/配置，使系统性配置错误在提交56万条任务前失败。
    engine = QualityEngine(
        rules,
        paths["robot_xml"],
        paths["kinematics"],
        paths["retarget_config"],
        paths["batch_config"],
    )
    summaries = _summaries(Path(paths["input_root"]), args.folders)
    code_paths = [
        Path(__file__),
        Path(__file__).with_name("umr_text_quality.py"),
        Path(__file__).with_name("filter_sonic_npz_motions.py"),
        ROOT / "gem/robots/bumi/kinematics.py",
        ROOT / "gem/robots/bumi/quality_filter.py",
        ROOT / "gem/robots/bumi/legacy_motion.py",
        ROOT / "gem/utils/rotation_conversions.py",
        ROOT / "tools/data/motionmillion/common.py",
    ]
    import mujoco

    identity = dict(
        schema=SCHEMA,
        paths=paths,
        assets=assets,
        limit=args.limit,
        folders=args.folders,
        expected_records=args.expected_records,
        libraries=dict(numpy=np.__version__, torch=torch.__version__, mujoco=mujoco.__version__),
        config_sha256=sha256_file(paths["config"]),
        retarget_config_sha256=sha256_file(paths["retarget_config"]),
        batch_config_sha256=sha256_file(paths["batch_config"]) if paths["batch_config"] else None,
        summaries={str(p): sha256_file(p) for p in summaries},
        code={str(p.relative_to(ROOT)): sha256_file(p) for p in code_paths},
        effective_limits={
            n: [float(a), float(b)]
            for n, a, b in zip(
                engine.config.joint_order,
                engine.config.joint_lower_limits,
                engine.config.joint_upper_limits,
            )
        },
    )
    fingerprint = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    if output.exists() and not args.resume:
        raise FileExistsError("报告目录已存在；同一任务续跑必须显式--resume")
    output.mkdir(parents=True, exist_ok=True)
    with (output / ".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        metadata = output / "run.json"
        if metadata.exists():
            previous = json.loads(metadata.read_text())
            if previous["fingerprint"] != fingerprint:
                raise ValueError("代码/配置/资产/输入清单变化，必须使用新的报告目录")
        elif any(p.name != ".lock" for p in output.iterdir()):
            raise ValueError("拒绝接管不是本工具创建的非空目录")
        run = dict(
            identity=identity,
            fingerprint=fingerprint,
            state="running",
            partial_scan=args.limit is not None or bool(args.folders),
            schema=SCHEMA,
        )
        write_json(metadata, run)
        db = sqlite3.connect(output / "quality.sqlite")
        try:
            db.execute("PRAGMA journal_mode=WAL")
            db.execute("PRAGMA synchronous=FULL")
            db.executescript("""
                CREATE TABLE IF NOT EXISTS inputs (relative_path TEXT PRIMARY KEY, folder TEXT,
                                                   human_path TEXT, upstream_status TEXT);
                CREATE TABLE IF NOT EXISTS results (relative_path TEXT PRIMARY KEY, payload TEXT NOT NULL);
            """)
            index_inputs(db, paths, summaries)
            run["indexed_records"] = db.execute("SELECT count(*) FROM inputs").fetchone()[0]
            if (
                args.expected_records is not None
                and run["indexed_records"] != args.expected_records
            ):
                raise ValueError(
                    f"全量清单数量 {run['indexed_records']} 与预期 {args.expected_records} 不同"
                )
            run["expected_records"] = min(
                args.limit or run["indexed_records"], run["indexed_records"]
            )
            write_json(metadata, run)
            counts, resumed, start = Counter(), 0, time.monotonic()
            for number, (row, cached) in enumerate(
                bounded_results(iter(_tasks(db, args.limit)), paths, args.workers), 1
            ):
                db.execute(
                    "INSERT OR REPLACE INTO results VALUES (?,?)",
                    (row["relative_path"], json.dumps(row, ensure_ascii=False, allow_nan=False)),
                )
                counts[row["status"]] += 1
                resumed += int(cached)
                if number % 32 == 0:
                    db.commit()
                if number == 1 or number % 100 == 0:
                    print(
                        json.dumps(
                            dict(
                                processed=number,
                                total=run["expected_records"],
                                resumed=resumed,
                                status_counts=dict(counts),
                                records_per_second=number / max(time.monotonic() - start, 0.001),
                            ),
                            ensure_ascii=False,
                        ),
                        flush=True,
                    )
            db.commit()
            if sum(counts.values()) != run["expected_records"]:
                raise RuntimeError("处理条数与任务清单不符")
            if (
                verify_asset_files(
                    paths["robot_xml"], paths["kinematics"], paths["asset_manifest"], rules
                )
                != assets
            ):
                raise RuntimeError("检查期间资产发生改变")
            report = publish_reports(db, output, run, args.limit)
            run.update(
                state="complete" if not counts["ERROR"] else "complete_with_errors",
                resumed_records=resumed,
                elapsed_seconds=time.monotonic() - start,
                summary_sha256=sha256_file(output / "quality_summary.json"),
            )
            write_json(metadata, run)
            print(json.dumps(report, ensure_ascii=False, indent=2))
            return 2 if counts["ERROR"] else 0
        finally:
            db.close()


class QualityGate:
    """构建器只消费完整报告中的PASS，并在实际读取时重验两个输入SHA。"""

    def __init__(self, report_dir):
        self.root = Path(report_dir).resolve(strict=True)
        self._lock = (self.root / ".lock").open("r")
        try:
            fcntl.flock(self._lock, fcntl.LOCK_SH | fcntl.LOCK_NB)
            self._load()
        except BaseException:
            self._lock.close()
            raise

    def _load(self):
        self.run = json.loads((self.root / "run.json").read_text())
        summary = json.loads((self.root / "quality_summary.json").read_text())
        check(
            self.run["schema"] == SCHEMA
            and self.run["state"] == "complete"
            and not self.run["partial_scan"],
            "不能用部分、未完成或存在ERROR的质量报告构建正式数据",
        )
        check(
            sha256_file(self.root / "quality_summary.json") == self.run["summary_sha256"]
            and summary["run_fingerprint"] == self.run["fingerprint"],
            "报告指纹不一致",
        )
        check(
            summary["processed_records"]
            == self.run["expected_records"]
            == self.run["indexed_records"],
            "报告未覆盖完整输入清单",
        )
        check(
            sha256_file(self.root / "train_candidates.jsonl")
            == summary["candidate_manifest_sha256"],
            "训练候选清单发生改变",
        )
        self.paths = self.run["identity"]["paths"]
        for field, key in (
            ("config", "config_sha256"),
            ("retarget_config", "retarget_config_sha256"),
            ("batch_config", "batch_config_sha256"),
        ):
            if self.paths[field]:
                check(sha256_file(self.paths[field]) == self.run["identity"][key], "报告配置已改变")
        rules = load_rules(self.paths["config"])
        check(rules["ground_height_m"] == 0, "文本release仅接受世界地面零点且不自动平移")
        check(
            verify_asset_files(
                self.paths["robot_xml"],
                self.paths["kinematics"],
                self.paths["asset_manifest"],
                rules,
            )
            == self.run["identity"]["assets"],
            "报告机器人资产已改变",
        )
        self.engine = QualityEngine(
            rules,
            self.paths["robot_xml"],
            self.paths["kinematics"],
            self.paths["retarget_config"],
            self.paths["batch_config"],
        )
        dbpath = quote(str(self.root / "quality.sqlite"), safe="/")
        self.db = sqlite3.connect(f"file:{dbpath}?mode=ro", uri=True)

    def close(self):
        self.db.close()
        self._lock.close()

    def lookup(self, path):
        path = Path(path).resolve(strict=True)
        check(path.is_relative_to(Path(self.paths["input_root"])), "qpos不在已筛选的源目录内")
        relative = path.relative_to(self.paths["input_root"]).as_posix()
        item = self.db.execute(
            "SELECT payload FROM results JOIN inputs USING(relative_path) WHERE relative_path=?",
            (relative,),
        ).fetchone()
        check(item is not None, "动作没有质量记录")
        row = json.loads(item[0])
        check(
            sha256_file(path) == row["source_sha256"]
            and sha256_file(row["human_path"]) == row["human_sha256"],
            "动作或源人体在筛选后改变",
        )
        return row

    @staticmethod
    def validate_identity(row, record):
        check(record["dataset"] == "motionmillion", "UMR报告仅对应MotionMillion")
        source_id = row["source_motion_id"]
        check(
            record["provenance"]["source_id"] == source_id
            and record.get("text_source_motion_id", record["motion_id"]) == source_id,
            "caption来源ID与原始MotionMillion文件身份不匹配",
        )
        canonical = record["provenance"].get("canonical_source_id")
        check(canonical in (None, row["canonical_source_id"]), "镜像归一化来源ID不一致")

    def read_candidate(self, path, record):
        row = self.lookup(path)
        if not row["training_eligible"] or row["status"] != "PASS":
            return None, row
        self.validate_identity(row, record)
        qpos, meta = load_umr(row, self.paths, self.engine)
        check(meta["frames"] == row["frames"], "读取帧数与质量报告不同")
        check(
            sha256_file(path) == row["source_sha256"]
            and sha256_file(row["human_path"]) == row["human_sha256"],
            "源文件在构建读取期间改变",
        )
        return torch.from_numpy(qpos.copy()), row
