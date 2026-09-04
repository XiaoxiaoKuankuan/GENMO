"""Independent BUMI music-only diffusion pipeline."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
from hydra.utils import instantiate

from gem.robots.bumi.endecoder import BumiEndecoder
from gem.robots.bumi.feature_codec import BUMI_FEATURE_DIM
from gem.robots.bumi.losses import BUMI_LOSS_CONTRACT_VERSION, BumiRobotLosses


class BumiMusicPipeline(nn.Module):
    """去噪 qpos30 特征、重建 qpos28，并只通过 Torch FK 计算机器人几何。"""

    def __init__(self, args, args_denoiser3d, **_kwargs: Any) -> None:
        super().__init__()
        self.args = args
        self.args_denoiser3d = args_denoiser3d
        train_modes = list(args.get("train_modes", ()))
        if train_modes != ["diffusion"]:
            raise ValueError(
                f"BumiMusicPipeline supports only train_modes=['diffusion'], got {train_modes}"
            )
        if list(args.get("in_attr", ())) != ["encoded_music"]:
            raise ValueError("BumiMusicPipeline accepts only in_attr=['encoded_music']")
        self.denoiser3d = instantiate(args_denoiser3d, _recursive_=False)
        self.endecoder: BumiEndecoder = instantiate(args.endecoder_opt, _recursive_=False)
        if not isinstance(self.endecoder, BumiEndecoder):
            raise TypeError("BumiMusicPipeline endecoder_opt must instantiate BumiEndecoder")
        self.denoiser3d.endecoder = self.endecoder
        self.losses = BumiRobotLosses(
            self.endecoder,
            args.weights,
            fps=30,
            contract_version=args.get("loss_contract", BUMI_LOSS_CONTRACT_VERSION),
            auxiliary_warmup_steps=args.get("auxiliary_warmup_steps", 0),
            ground_semantics=args.get("ground_semantics", None),
            joint_limit_margin_rad=args.get("joint_limit_margin_rad", 0.0),
            joint_limit_topk_fraction=args.get("joint_limit_topk_fraction", 0.01),
            robust_joint_limit_start_step=args.get("robust_joint_limit_start_step", 0),
            robust_joint_limit_warmup_steps=args.get("robust_joint_limit_warmup_steps", 0),
            advanced_physics_start_step=args.get("advanced_physics_start_step", 0),
            advanced_physics_warmup_steps=args.get("advanced_physics_warmup_steps", 0),
            advanced_physics_topk_fraction=args.get("advanced_physics_topk_fraction", 0.05),
            root_tilt_upright_allowance_rad=args.get("root_tilt_upright_allowance_rad", 0.35),
            root_tilt_target_margin_rad=args.get("root_tilt_target_margin_rad", 0.10),
        )

    @staticmethod
    def _prediction(model_output: dict[str, Any]) -> torch.Tensor:
        for key in ("pred_x", "pred_x_start", "pred_xstart"):
            value = model_output.get(key)
            if isinstance(value, torch.Tensor):
                return value
        raise ValueError("BUMI denoiser output has no pred_x/pred_x_start prediction tensor")

    def forward(
        self,
        inputs: dict[str, Any],
        train: bool = False,
        postproc: bool = False,
        static_cam: bool = False,
        global_step: int = 0,
        mode: str | None = None,
        test_mode: str | None = None,
        normalizer_stats: dict | None = None,
    ) -> dict[str, Any]:
        del postproc, static_cam
        if self.endecoder.obs_indices_dict is None:
            self.endecoder.build_obs_indices_dict()
        model_output = self.denoiser3d(
            inputs,
            train=train,
            mode=mode,
            test_mode=test_mode,
            normalizer_stats=normalizer_stats,
        )
        pred_x = self._prediction(model_output)
        if pred_x.shape[-1] != BUMI_FEATURE_DIM:
            raise RuntimeError(
                f"BUMI denoiser must output {BUMI_FEATURE_DIM}D qpos motion, got {pred_x.shape}"
            )
        decode_dict = self.endecoder.decode(pred_x)
        pred_qpos_canonical = self.endecoder.compose_qpos(decode_dict)
        fk = self.endecoder.kinematics.forward_kinematics(pred_qpos_canonical)
        pred_body_link_pos_root_fk = self.endecoder.codec.body_positions_in_root_frame(
            pred_qpos_canonical[..., :3],
            pred_qpos_canonical[..., 3:7],
            fk["body_pos_w"][..., 1:, :],
        )
        outputs: dict[str, Any] = {
            "model_output": model_output,
            "decode_dict": decode_dict,
            "pred_qpos_canonical": pred_qpos_canonical,
            "pred_body_link_pos_root_fk": pred_body_link_pos_root_fk,
            "pred_foot_contact_logits": model_output.get("static_conf_logits"),
        }
        world_anchor = inputs.get("world_anchor")
        if world_anchor is not None:
            pred_qpos_world = self.endecoder.compose_qpos(decode_dict, world_anchor=world_anchor)
            world_fk = self.endecoder.kinematics.forward_kinematics(pred_qpos_world)
            outputs["pred_qpos"] = pred_qpos_world
            outputs["pred_body_link_pos_world_fk"] = world_fk["body_pos_w"][..., 1:, :]
        else:
            outputs["pred_qpos"] = pred_qpos_canonical
        if not train:
            return outputs
        loss_output = self.losses(
            inputs,
            model_output,
            decode_dict,
            pred_qpos_canonical,
            fk,
            global_step=global_step,
        )
        outputs.update(loss_output)
        return outputs


__all__ = ["BumiMusicPipeline"]
