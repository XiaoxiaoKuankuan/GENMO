# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NVIDIA-OneWay-Noncommercial
"""AIST++ 音乐舞蹈训练数据集。

该数据集读取 AIST++ 舞蹈动作、相机/图像特征和音乐相关特征，把舞蹈视频整理成
GEM 可训练的时序样本。完整 gem_smpl 模型用它学习音乐条件下的人体动作生成能力。
"""

import os
import re
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils import data

from gem.utils.geo_transform import (
    compute_cam_angvel,
    compute_cam_tvel,
    get_bbx_xys_from_xyxy,
    normalize_T_w2c,
)
from gem.utils.ground_sidecar import load_ground_sidecar
from gem.utils.motion_utils import get_c_rootparam, get_R_c2gv
from gem.utils.net_utils import (
    get_valid_mask,
    repeat_to_max_len,
    repeat_to_max_len_dict,
)
from gem.utils.pylogger import Log
from gem.utils.smplx_utils import make_smplx

AIST_METRIC_MAX_MEDIAN_ROOT_STEP_M = 0.25
AIST_METRIC_MAX_ABS_TRANSLATION_P95_M = 20.0
_AIST_MUSIC_TOKEN_RE = re.compile(r"_(m[^_]+)_ch", re.IGNORECASE)


def get_aist_music_token(sequence_id: str) -> str:
    """Extract the official AIST++ music token, for example ``mBR1``."""
    match = _AIST_MUSIC_TOKEN_RE.search(str(sequence_id))
    if match is None:
        raise ValueError(f"AIST++ sequence has no music token: {sequence_id}")
    return match.group(1).casefold()


def aist_translation_statistics(
    translations: np.ndarray | torch.Tensor,
) -> dict[str, float]:
    """Return robust diagnostics for one AIST++ root trajectory in metres."""
    array = (
        translations.detach().cpu().numpy()
        if isinstance(translations, torch.Tensor)
        else np.asarray(translations)
    )
    if array.ndim != 2 or array.shape[1] != 3 or array.shape[0] <= 0:
        raise ValueError(f"AIST++ translation must be non-empty [F,3], got {array.shape}")
    if not np.issubdtype(array.dtype, np.number) or not np.isfinite(array).all():
        raise ValueError("AIST++ translation must be numeric and finite")
    steps = (
        np.linalg.norm(np.diff(array.astype(np.float64), axis=0), axis=1)
        if array.shape[0] > 1
        else np.zeros(1, dtype=np.float64)
    )
    return {
        "median_root_step_m": float(np.median(steps)),
        "p95_root_step_m": float(np.percentile(steps, 95)),
        "max_root_step_m": float(steps.max(initial=0.0)),
        "p95_abs_translation_m": float(
            np.percentile(np.abs(array.astype(np.float64)), 95)
        ),
    }


def validate_aist_metric_translation(
    translations: np.ndarray | torch.Tensor,
    *,
    sequence_id: str,
    max_median_root_step_m: float = AIST_METRIC_MAX_MEDIAN_ROOT_STEP_M,
    max_abs_translation_p95_m: float = AIST_METRIC_MAX_ABS_TRANSLATION_P95_M,
) -> dict[str, float]:
    """Reject stale AIST++ artifacts that still store centimetre-scale motion."""
    stats = aist_translation_statistics(translations)
    if (
        stats["median_root_step_m"] > max_median_root_step_m
        or stats["p95_abs_translation_m"] > max_abs_translation_p95_m
    ):
        raise ValueError(
            "AIST++ translation is not in GEM metric scale: "
            f"sequence_id={sequence_id}, "
            f"median_root_step={stats['median_root_step_m']:.6f}m, "
            f"p95_abs_translation={stats['p95_abs_translation_m']:.6f}m. "
            "This usually means annot_aist_30fps.pt was built before smpl_trans "
            "was divided by smpl_scaling. Rebuild the artifact; do not train or "
            "repair the target by silently clipping/zeroing translation."
        )
    return stats


def load_music_feature_tensor(path: str | Path) -> torch.Tensor:
    """Load a NumPy- or Tensor-backed ``.pt`` music feature as float32."""
    path = Path(path)
    value = torch.load(path, map_location="cpu", weights_only=False)
    try:
        tensor = torch.as_tensor(value).float()
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"Unsupported music feature payload in {path}: {type(value).__name__}"
        ) from exc
    return tensor


def load_aist_artifact(path: str | Path) -> Any:
    """Load trusted local AIST++ NumPy/Tensor artifacts across Torch versions."""
    path = Path(path)
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def validate_musicfeat_v2(features: torch.Tensor, source: str | Path = "musicfeat_v2") -> None:
    """Validate the EDGE baseline35 contract used by the v2 AIST++ loader."""
    if features.ndim != 2 or features.shape[1] != 35:
        raise ValueError(f"{source} must have shape [L, 35]; got {tuple(features.shape)}")
    if not torch.isfinite(features).all():
        raise ValueError(f"{source} contains NaN or Inf")
    for channel in (33, 34):
        values = features[:, channel]
        binary = torch.isclose(values, torch.zeros_like(values), atol=1e-4) | torch.isclose(
            values, torch.ones_like(values), atol=1e-4
        )
        if binary.float().mean().item() < 0.99:
            raise ValueError(f"{source} channel {channel} must be essentially binary (0/1)")


def load_music_beats(
    root: str | Path,
    vid: str,
    music_features: torch.Tensor,
) -> torch.Tensor:
    """Use legacy beat channel 53 when present, otherwise v2 channel 34."""
    legacy_path = Path(root) / "musicfeat" / f"{vid}_musicfeat_fps30.pt"
    if legacy_path.is_file():
        legacy = load_music_feature_tensor(legacy_path)
        if legacy.ndim != 2 or legacy.shape[1] <= 53:
            raise ValueError(
                f"Legacy music feature must expose beat channel 53: {legacy_path}, "
                f"got {tuple(legacy.shape)}"
            )
        return legacy[..., 53]
    validate_musicfeat_v2(music_features, source=f"musicfeat_v2 for {vid}")
    return music_features[..., 34]


def resolve_music_motion_alignment(
    *,
    sequence_id: str,
    motion_frames: int,
    music_frames: int,
    music_feature_path: str | Path,
    strict: bool,
    max_mismatch: int,
) -> tuple[int, dict[str, Any]]:
    """Validate/resolve the shared temporal extent of motion and music."""
    difference = abs(int(motion_frames) - int(music_frames))
    info = {
        "sequence_id": sequence_id,
        "motion_frames": int(motion_frames),
        "music_frames": int(music_frames),
        "difference": difference,
        "music_feature_path": str(music_feature_path),
        "trimmed_to_frames": min(int(motion_frames), int(music_frames)),
    }
    if strict and difference > max_mismatch:
        raise ValueError(
            "AIST++ music-motion alignment exceeds tolerance: "
            f"sequence_id={sequence_id}, F_motion={motion_frames}, "
            f"F_music={music_frames}, difference={difference}, "
            f"music_feature_path={music_feature_path}"
        )
    return info["trimmed_to_frames"], info


def select_aist_temporal_window(
    *,
    sequence_length: int,
    target_length: int,
    random_crop: bool,
    eval_clip_mode: str = "first",
) -> tuple[int, int]:
    """Select a safe random or deterministic AIST++ temporal window."""
    if sequence_length <= 0 or target_length <= 0:
        raise ValueError("sequence_length and target_length must be positive")
    if eval_clip_mode not in {"first", "center"}:
        raise ValueError("eval_clip_mode must be 'first' or 'center'")
    if sequence_length <= target_length:
        return 0, sequence_length
    if random_crop:
        start = int(np.random.randint(0, sequence_length - target_length + 1))
    elif eval_clip_mode == "center":
        start = (sequence_length - target_length) // 2
    else:
        start = 0
    return start, target_length


class AISTPlusPlusSmplDataset(data.Dataset):
    def __init__(
        self,
        root="inputs/AIST++",
        split="train",
        motion_frames=120,
        lazy_load=False,
        eval_gen_only=True,
        feat_version="v1",
        annot_file=None,
        split_file=None,
        strict_music_alignment: bool = False,
        max_music_motion_frame_mismatch: int = 2,
        load_raw_music_audio: bool = True,
        eval_motion_frames: int | None = None,
        eval_clip_mode: str = "first",
        music_only_conditioning: bool = False,
        enable_contact_supervision: bool = False,
        duration_aware_sampling: bool = False,
        validate_metric_translation: bool = False,
        aist_world_up_axis: str = "z",
        ground_sidecar_path: str | Path | None = None,
        require_ground_sidecar: bool = False,
    ):
        super().__init__()
        # Path
        self.root = Path(root)

        # Setting
        self.motion_frames = motion_frames
        self.lazy_load = lazy_load
        self.split = split
        self.eval_gen_only = eval_gen_only
        self.feat_version = feat_version
        self.annot_file = annot_file
        self.split_file = split_file
        self.strict_music_alignment = strict_music_alignment
        self.max_music_motion_frame_mismatch = max_music_motion_frame_mismatch
        self.load_raw_music_audio = load_raw_music_audio
        self.eval_motion_frames = eval_motion_frames
        self.eval_clip_mode = eval_clip_mode
        self.music_only_conditioning = music_only_conditioning
        self.validate_metric_translation = validate_metric_translation
        self.aist_world_up_axis = aist_world_up_axis
        self.ground_sidecar_path = ground_sidecar_path
        self.require_ground_sidecar = bool(require_ground_sidecar)
        # Opt-in only: old AIST++ experiments keep one item per sequence.
        # The four-dataset specialist repeats a sequence in proportion to its
        # number of complete 120-frame windows.
        self.duration_aware_sampling = duration_aware_sampling
        # Upstream AIST++ samples disable static-joint supervision.  Keep that
        # legacy default, while allowing the music-only specialist to opt in to
        # velocity-derived contact labels from its fully supervised SMPL motion.
        self.enable_contact_supervision = enable_contact_supervision
        if self.max_music_motion_frame_mismatch < 0:
            raise ValueError("max_music_motion_frame_mismatch must be non-negative")
        if self.eval_motion_frames is not None and self.eval_motion_frames <= 0:
            raise ValueError("eval_motion_frames must be positive when provided")
        if self.eval_clip_mode not in {"first", "center"}:
            raise ValueError("eval_clip_mode must be 'first' or 'center'")
        if self.aist_world_up_axis not in {"y", "z"}:
            raise ValueError("aist_world_up_axis must be 'y' or 'z'")
        self.music_alignment_stats = {
            "exact_match_count": 0,
            "within_1_count": 0,
            "within_2_count": 0,
            "trimmed_count": 0,
            "max_abs_mismatch": 0,
        }
        self._load_dataset()
        self._get_idx2meta()
        if self.ground_sidecar_path is None:
            self.ground_records = None
            if self.require_ground_sidecar:
                raise ValueError(
                    "AIST++ require_ground_sidecar=True but no path was configured"
                )
        else:
            sidecar_path = Path(self.ground_sidecar_path).expanduser()
            if not sidecar_path.is_absolute():
                sidecar_path = self.root / sidecar_path
            self.ground_records = load_ground_sidecar(sidecar_path)
            expected_ids = set(self.idx2meta)
            missing_ids = sorted(expected_ids - set(self.ground_records))
            if missing_ids:
                preview = ", ".join(missing_ids[:5])
                raise ValueError(
                    f"AIST++ ground sidecar misses {len(missing_ids)} samples: {preview}"
                )
            for sequence_id in expected_ids:
                expected_frames = int(
                    self.motion_files[sequence_id]["bbox_xyxy"].shape[0]
                )
                if int(self.ground_records[sequence_id]["num_frames"]) != expected_frames:
                    raise ValueError(
                        f"{sequence_id}: ground num_frames differs from AIST artifact"
                    )

    def _load_dataset(self):
        # smplpose
        tic = Log.time()
        annot_filename = (
            self.annot_file if self.annot_file is not None else "annot_aist_30fps.pt"
        )
        split_filename = (
            self.split_file if self.split_file is not None else f"{self.split}.pt"
        )
        fn = self.root / annot_filename
        self.annot_path = fn
        self.smpl_model = make_smplx("supermotion")
        Log.info(f"[AIST++ {self.feat_version}] Loading from {fn} ...")
        self.motion_files = load_aist_artifact(fn)
        self.split_set = load_aist_artifact(self.root / split_filename)
        if self.validate_metric_translation:
            metric_stats = []
            for sequence_id in self.split_set:
                if sequence_id not in self.motion_files:
                    continue
                metric_stats.append(
                    validate_aist_metric_translation(
                        self.motion_files[sequence_id]["smpl_trans_global"],
                        sequence_id=sequence_id,
                    )
                )
            self.metric_translation_summary = {
                "checked_sequences": len(metric_stats),
                "max_median_root_step_m": max(
                    (item["median_root_step_m"] for item in metric_stats), default=0.0
                ),
                "max_p95_abs_translation_m": max(
                    (item["p95_abs_translation_m"] for item in metric_stats), default=0.0
                ),
            }
        # Dict of {
        #          "smpl_params_glob": {'body_pose', 'global_orient', 'transl', 'betas'}, FxC
        #          "cam_Rt": tensor(F, 3),
        #          "cam_K": tensor(1, 10),
        #         }
        self.seqs = list(self.motion_files.keys())
        Log.info(
            f"[AIST++ {self.feat_version}] {len(self.seqs)} sequences. Elapsed: {Log.time() - tic:.2f}s"
        )

    def _get_idx2meta(self):
        # We expect to see the entire sequence during one epoch,
        # so each sequence will be sampled max(SeqLength // MotionFrames, 1) times
        seq_lengths = []
        self.idx2meta = []
        for vid in self.motion_files:
            if vid not in self.split_set:
                continue
            motion_length = self.motion_files[vid]["bbox_xyxy"].shape[0]
            if self.duration_aware_sampling:
                music_path = (
                    self.root
                    / f"musicfeat_{self.feat_version}/{vid}_musicfeat_fps30.pt"
                )
                music_length = int(load_music_feature_tensor(music_path).shape[0])
                seq_length, _ = resolve_music_motion_alignment(
                    sequence_id=vid,
                    motion_frames=motion_length,
                    music_frames=music_length,
                    music_feature_path=music_path,
                    strict=self.strict_music_alignment,
                    max_mismatch=self.max_music_motion_frame_mismatch,
                )
            else:
                seq_length = motion_length
            seq_lengths.append(seq_length)
            repeat_count = (
                max(seq_length // self.motion_frames, 1)
                if self.duration_aware_sampling
                else 1
            )
            self.idx2meta.extend([vid] * repeat_count)
        hours = sum(seq_lengths) / 30 / 3600
        self.raw_sequence_count = len(seq_lengths)
        self.total_valid_frames = sum(seq_lengths)
        self.sampling_summary = {
            "dataset_name": "aist++",
            "raw_sequences": self.raw_sequence_count,
            "valid_frames": self.total_valid_frames,
            "hours": hours,
            "effective_len": len(self.idx2meta),
            "duration_aware_sampling": self.duration_aware_sampling,
        }
        Log.info(
            f"[AIST++] has {hours:.1f} hours motion -> Resampled to {len(self.idx2meta)} samples."
        )

    def __len__(self):
        return len(self.idx2meta)

    def get_music_sampling_records(self) -> list[dict[str, Any]]:
        """Return one AIST++ dance variant per official music token."""
        records: list[dict[str, Any]] = []
        seen: set[str] = set()
        for dataset_index, sequence_id in enumerate(self.idx2meta):
            if sequence_id in seen:
                continue
            seen.add(sequence_id)
            records.append(
                {
                    "dataset_index": dataset_index,
                    "sample_id": sequence_id,
                    "group_id": get_aist_music_token(sequence_id),
                    "num_frames": int(
                        self.motion_files[sequence_id]["bbox_xyxy"].shape[0]
                    ),
                }
            )
        return records

    def _load_data(self, idx):
        sampled_motion = {}
        vid = self.idx2meta[idx]
        motion = self.motion_files[vid]

        music_feat_path = self.root / f"musicfeat_{self.feat_version}/{vid}_musicfeat_fps30.pt"
        music_feat = load_music_feature_tensor(music_feat_path)
        if self.feat_version == "v2":
            validate_musicfeat_v2(music_feat, source=music_feat_path)
        motion_frames = int(motion["bbox_xyxy"].shape[0])
        music_frames = int(music_feat.shape[0])
        seq_length, alignment = resolve_music_motion_alignment(
            sequence_id=vid,
            motion_frames=motion_frames,
            music_frames=music_frames,
            music_feature_path=music_feat_path,
            strict=self.strict_music_alignment,
            max_mismatch=self.max_music_motion_frame_mismatch,
        )
        difference = alignment["difference"]
        self.music_alignment_stats["exact_match_count"] += int(difference == 0)
        self.music_alignment_stats["within_1_count"] += int(difference <= 1)
        self.music_alignment_stats["within_2_count"] += int(difference <= 2)
        self.music_alignment_stats["trimmed_count"] += int(difference > 0)
        self.music_alignment_stats["max_abs_mismatch"] = max(
            self.music_alignment_stats["max_abs_mismatch"], difference
        )
        if self.strict_music_alignment and difference > 0:
            warnings.warn(
                "AIST++ strict alignment trimmed a small frame mismatch: "
                f"sequence_id={vid}, F_motion={motion_frames}, F_music={music_frames}, "
                f"difference={difference}, music_feature_path={music_feat_path}",
                RuntimeWarning,
                stacklevel=2,
            )
        sampled_motion["vid"] = vid
        sampled_motion["music_feature_path"] = str(music_feat_path)
        sampled_motion["alignment"] = alignment
        sampled_motion["contact_supervision_valid"] = bool(
            motion.get("contact_supervision_valid", True)
        )
        sampled_motion["ground_record"] = (
            self.ground_records.get(vid) if self.ground_records is not None else None
        )

        # Random train crop or deterministic fixed-window evaluation crop.
        if self.split in ["train", "minitrain"]:
            target_length = self.motion_frames
            random_crop = True
        elif self.eval_motion_frames is not None:
            target_length = self.eval_motion_frames
            random_crop = False
        else:
            target_length = seq_length
            random_crop = False
        start, length = select_aist_temporal_window(
            sequence_length=seq_length,
            target_length=target_length,
            random_crop=random_crop,
            eval_clip_mode=self.eval_clip_mode,
        )
        if seq_length < target_length:
            Log.info(
                f"[AIST++] ({idx}) sequence shorter than target: {seq_length} < {target_length}"
            )
        end = start + length
        sampled_motion["length"] = length
        sampled_motion["start_end"] = (start, end)

        music_beats = load_music_beats(self.root, vid, music_feat)
        sampled_motion["music_beats"] = torch.as_tensor(music_beats[start:end]).float()

        # Select motion subset
        # body_pose, global_orient, transl, betas
        sampled_motion["smpl_params_glob"] = {
            "body_pose": torch.from_numpy(motion["smpl_pose_global"][start:end][:, 3:66]).float(),
            "betas": torch.zeros((length, 10)).float(),
            "global_orient": torch.from_numpy(motion["smpl_pose_global"][start:end][:, :3]).float(),
            "transl": torch.from_numpy(motion["smpl_trans_global"][start:end]).float(),
        }

        sampled_motion["smpl_params_cam"] = {
            "body_pose": torch.from_numpy(motion["smpl_pose"][start:end][:, 3:66]).float(),
            "betas": torch.zeros((length, 10)).float(),
            "global_orient": torch.from_numpy(motion["smpl_pose"][start:end][:, :3]).float(),
            "transl": torch.from_numpy(motion["smpl_trans"][start:end]).float(),
        }

        # Image as feature
        sampled_motion["f_imgseq"] = torch.zeros((length, 1024)).float()

        bbx_xys = get_bbx_xys_from_xyxy(
            torch.from_numpy(motion["bbox_xyxy"][start:end]), base_enlarge=1.2
        )
        sampled_motion["bbx_xys"] = bbx_xys.float()
        sampled_motion["K_fullimg"] = motion["intrinsics"]
        sampled_motion["kp2d"] = torch.zeros((length, 17, 3)).float()

        # Camera
        sampled_motion["T_w2c"] = motion["T_w2c"]  # (4, 4)

        sampled_motion["music_embed"] = torch.as_tensor(music_feat[start:end]).float()  # (L, 35)

        # load audio
        if not self.load_raw_music_audio or self.split in ["train", "minitrain"]:
            music_fps = 30
            music_array = torch.zeros((length, 1024)).float()
        else:
            from moviepy.editor import AudioFileClip

            music_array = torch.load(os.path.join(self.root, f"audio_array/{vid}.pt"))
            music_array = torch.from_numpy(music_array).float()
            music = AudioFileClip(os.path.join(self.root, f"audio/{vid}.mp3"))
            music_fps = music.fps
            start_audio = int(start * music_fps / 30)
            end_audio = int(end * music_fps / 30)
            music_array = music_array[start_audio:end_audio]

        sampled_motion["music_array"] = music_array
        sampled_motion["music_fps"] = music_fps
        sampled_motion["height"] = motion["height"]
        sampled_motion["width"] = motion["width"]
        return sampled_motion

    def _process_data(self, data, idx):
        length = data["length"]

        # SMPL params in world
        smpl_params_w = data["smpl_params_glob"]
        old_smpl_params_c = data["smpl_params_cam"]
        music_fps = data["music_fps"]

        # SMPL params in cam
        T_w2c = data["T_w2c"]  # (4, 4)
        offset = self.smpl_model.get_skeleton(smpl_params_w["betas"][0])[0]  # (3)
        global_orient_c, transl_c = get_c_rootparam(
            smpl_params_w["global_orient"],
            smpl_params_w["transl"],
            T_w2c,
            offset,
        )
        assert (old_smpl_params_c["global_orient"] - global_orient_c).abs().max().item() < 1e-4, (
            (old_smpl_params_c["global_orient"] - global_orient_c).abs().max().item(),
            data["vid"],
            data["start_end"],
        )
        assert (old_smpl_params_c["transl"] - transl_c).abs().max().item() < 1e-4, (
            (old_smpl_params_c["transl"] - transl_c).abs().max().item(),
            data["vid"],
            data["start_end"],
        )

        smpl_params_c = {
            "body_pose": smpl_params_w["body_pose"].clone(),  # (F, 63)
            "betas": smpl_params_w["betas"].clone(),  # (F, 10)
            "global_orient": global_orient_c,  # (F, 3)
            "transl": transl_c,  # (F, 3)
        }

        # World params
        # Official AIST++ SMPL forward after ``smpl_trans / smpl_scaling`` is
        # Y-up. Keep the old Z-up mode configurable so legacy generalist
        # experiments preserve their exact input contract, while corrected
        # music-only experiments explicitly select Y-up.
        gravity_vec = (
            torch.tensor([0.0, -1.0, 0.0])
            if self.aist_world_up_axis == "y"
            else torch.tensor([0.0, 0.0, -1.0])
        ).float()
        T_w2c = T_w2c.repeat(length, 1, 1)  # (F, 4, 4)
        R_c2gv = get_R_c2gv(T_w2c[..., :3, :3], axis_gravity_in_w=gravity_vec)  # (F, 3, 3)

        # Image
        bbx_xys = data["bbx_xys"]  # (F, 3)
        K_fullimg = data["K_fullimg"].repeat(length, 1, 1)  # (F, 3, 3)
        f_imgseq = data["f_imgseq"]  # (F, 1024)

        normed_T_w2c = normalize_T_w2c(T_w2c)

        cam_angvel = compute_cam_angvel(
            normed_T_w2c[:, :3, :3]
        )  # (F, 6)  slightly different from WHAM
        cam_tvel = compute_cam_tvel(normed_T_w2c[:, :3, 3])  # (F, 3)
        assert cam_tvel.sum() == 0, cam_tvel

        # Returns: do not forget to make it batchable! (last lines)
        max_len = self.motion_frames if self.split in ["train", "minitrain"] else length
        return_data = {
            "meta": {
                "data_name": "aist++",
                "dataset_id": "aist++",
                "idx": idx,
                "vid": data["vid"],
                "height": data["height"],
                "width": data["width"],
                "eval_gen_only": self.eval_gen_only,
                "music_feature_path": data["music_feature_path"],
                "music_motion_alignment": data["alignment"],
                "start_end": data["start_end"],
                "contact_supervision_valid": data["contact_supervision_valid"],
                "aist_world_up_axis": self.aist_world_up_axis,
                "music_group_id": get_aist_music_token(data["vid"]),
            },
            "length": length,
            "smpl_params_c": smpl_params_c,
            "smpl_params_w": smpl_params_w,
            "R_c2gv": R_c2gv,  # (F, 3, 3)
            "gravity_vec": gravity_vec,  # (3)
            "bbx_xys": bbx_xys,  # (F, 3)
            "K_fullimg": K_fullimg,  # (F, 3, 3)
            "f_imgseq": f_imgseq,  # (F, D)
            "kp2d": data["kp2d"],  # (F, 17, 3)
            "cam_angvel": cam_angvel,  # (F, 6)
            "cam_tvel": cam_tvel,  # (F, 3)
            "noisy_cam_tvel": cam_tvel,  # (F, 3)
            "T_w2c": normed_T_w2c,  # (F, 4, 4)
            "music_embed": data["music_embed"],  # (F, C)
            "music_array": data["music_array"],  # (F / 30 * audio_fps, C)
            "music_fps": music_fps,
            "music_beats": data["music_beats"],  # (F,)
            "mask": {
                "valid": get_valid_mask(length, length),
                "humanoid": get_valid_mask(length, 0),
                "has_img_mask": get_valid_mask(length, 0),
                "has_2d_mask": get_valid_mask(
                    length, 0 if self.music_only_conditioning else length
                ),
                "has_cam_mask": get_valid_mask(length, 0),
                "has_audio_mask": get_valid_mask(length, 0),
                "has_music_mask": get_valid_mask(length, length),
                "2d_only": False,
                "vitpose": False,
                "bbx_xys": False,
                "f_imgseq": False,
                "spv_incam_only": False,
                "invalid_contact": not (
                    self.enable_contact_supervision
                    and data["contact_supervision_valid"]
                ),
            },
        }

        ground_record = data.get("ground_record")
        if ground_record is not None:
            ground_valid = bool(ground_record["ground_valid"])
            ground_y = float(ground_record["ground_y"]) if ground_valid else 0.0
            return_data["physics"] = {
                "ground_y": torch.tensor(ground_y, dtype=torch.float32),
                "ground_y_local": torch.tensor(
                    ground_y - float(smpl_params_w["transl"][0, 1]),
                    dtype=torch.float32,
                ),
                "ground_valid": torch.tensor(ground_valid, dtype=torch.bool),
            }

        # Batchable
        if self.split in ["train", "minitrain"]:
            return_data["smpl_params_c"] = repeat_to_max_len_dict(
                return_data["smpl_params_c"], max_len
            )
            return_data["smpl_params_w"] = repeat_to_max_len_dict(
                return_data["smpl_params_w"], max_len
            )
            return_data["R_c2gv"] = repeat_to_max_len(return_data["R_c2gv"], max_len)
            return_data["bbx_xys"] = repeat_to_max_len(return_data["bbx_xys"], max_len)
            return_data["K_fullimg"] = repeat_to_max_len(return_data["K_fullimg"], max_len)
            return_data["f_imgseq"] = repeat_to_max_len(return_data["f_imgseq"], max_len)
            return_data["music_embed"] = repeat_to_max_len(return_data["music_embed"], max_len)
            return_data["music_array"] = repeat_to_max_len(
                return_data["music_array"], int(max_len / 30 * music_fps)
            )
            return_data["music_beats"] = repeat_to_max_len(return_data["music_beats"], max_len)

            return_data["kp2d"] = repeat_to_max_len(return_data["kp2d"], max_len)
            return_data["cam_angvel"] = repeat_to_max_len(return_data["cam_angvel"], max_len)
            return_data["cam_tvel"] = repeat_to_max_len(return_data["cam_tvel"], max_len)
            return_data["noisy_cam_tvel"] = repeat_to_max_len(
                return_data["noisy_cam_tvel"], max_len
            )
            return_data["T_w2c"] = repeat_to_max_len(return_data["T_w2c"], max_len)
        return return_data

    def __getitem__(self, idx):
        data = self._load_data(idx)
        data = self._process_data(data, idx)
        return data
