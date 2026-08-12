# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NVIDIA-OneWay-Noncommercial
"""EMDB 验证指标回调。

训练验证阶段会根据 dataset_id 收集 EMDB 输出，计算相机坐标或全局坐标下的 MPJPE、
PVE、RTE、jitter 等指标，并在 epoch 结束时汇总记录。
"""

import numpy as np
import pytorch_lightning as pl
import torch
from einops import einsum

from gem.utils.eval_utils import (
    InvalidMetricInputError,
    apply_invalid_policy,
    as_np_array,
    check_finite_metric_inputs,
    compute_camcoord_metrics,
    compute_global_metrics,
    dump_invalid_eval_sample,
    validate_invalid_policy,
)
from gem.utils.gather import all_gather
from gem.utils.geo_transform import apply_T_on_points
from gem.utils.pylogger import Log
from gem.utils.smplx_utils import make_smplx

_BODY_MODEL_DIR = "inputs/checkpoints/body_models"


class MetricMocap(pl.Callback):
    def __init__(self, emdb_split=1, occ=False, invalid_policy="skip", dump_invalid=True):
        """
        Args:
            emdb_split: 1 to evaluate incam, 2 to evaluate global
            occ: whether evaluating on occluded variant
        """
        super().__init__()
        self.occ = occ
        self.invalid_policy = validate_invalid_policy(invalid_policy)
        self.dump_invalid = bool(dump_invalid)
        self.invalid_sequences: dict[str, str] = {}
        # vid->result
        if emdb_split == 1:
            self.target_dataset_id = "EMDB_1" if not self.occ else "EMDB_1-OCC"
            self.metric_aggregator = {
                "pa_mpjpe": {},
                "mpjpe": {},
                "pve": {},
                "accel": {},
            }
        elif emdb_split == 2:
            self.target_dataset_id = "EMDB_2" if not self.occ else "EMDB_2-OCC"
            self.metric_aggregator = {
                "wa2_mpjpe": {},
                "waa_mpjpe": {},
                "rte": {},
                "jitter": {},
                "fs": {},
            }
        else:
            raise ValueError(f"Unknown emdb_split: {emdb_split}")

        # SMPL-X (for prediction FK) and SMPL (for ground truth FK)
        self.smplx = make_smplx("supermotion")
        self.smpl_model = {
            "male": make_smplx("smpl", gender="male"),
            "female": make_smplx("smpl", gender="female"),
        }

        self.J_regressor = torch.load(f"{_BODY_MODEL_DIR}/smpl_neutral_J_regressor.pt")
        self.smplx2smpl = torch.load(f"{_BODY_MODEL_DIR}/smplx2smpl_sparse.pt")
        self.faces_smpl = self.smpl_model["male"].faces

        # The metrics are calculated similarly for val/test/predict
        self.on_test_batch_end = self.on_validation_batch_end = self.on_predict_batch_end
        self.on_test_epoch_end = self.on_validation_epoch_end = self.on_predict_epoch_end
        self.on_test_epoch_start = self.on_validation_epoch_start = self.on_predict_epoch_start

    # ================== Batch-based Computation  ================== #
    def on_predict_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        """The behaviour is the same for val/test/predict"""
        assert batch["B"] == 1
        dataset_id = batch["meta"][0]["dataset_id"]
        if dataset_id != self.target_dataset_id:
            return

        # Move to cuda if not
        self.smplx = self.smplx.cuda()
        for g in ["male", "female"]:
            self.smpl_model[g] = self.smpl_model[g].cuda()
        self.J_regressor = self.J_regressor.cuda()
        self.smplx2smpl = self.smplx2smpl.cuda()

        vid = batch["meta"][0]["vid"]
        gender = batch["gender"][0]
        T_w2c = batch["gt_T_w2c"][0]
        mask = batch["mask"]["valid"][0]

        try:
            for group_name in ("pred_body_params_incam", "pred_body_params_global"):
                if group_name in outputs:
                    self._check_body_params(outputs[group_name], group_name, vid)

            # Groundtruth (world, cam)
            target_w_params = {k: v[0] for k, v in batch["smpl_params"].items()}
            self._check_body_params(target_w_params, "target_w_params", vid)
            check_finite_metric_inputs({"T_w2c": T_w2c}, sequence_id=vid)
            target_w_output = self.smpl_model[gender](**target_w_params)
            target_w_verts = target_w_output.vertices
            target_w_j3d = torch.matmul(self.J_regressor, target_w_verts)
            target_c_verts = apply_T_on_points(target_w_verts, T_w2c)
            target_c_j3d = apply_T_on_points(target_w_j3d, T_w2c)
            check_finite_metric_inputs(
                {
                    "target_w_verts": target_w_verts,
                    "target_w_j3d": target_w_j3d,
                    "target_c_verts": target_c_verts,
                    "target_c_j3d": target_c_j3d,
                },
                sequence_id=vid,
            )

            if self.target_dataset_id in ["EMDB_1", "EMDB_1-OCC"]:
                if "pred_smpl_vertices_incam" in outputs:
                    pred_c_verts = outputs["pred_smpl_vertices_incam"]
                else:
                    pred_smpl_params_incam = outputs["pred_body_params_incam"]
                    smpl_out = self.smplx(**pred_smpl_params_incam)
                    check_finite_metric_inputs(
                        {"pred_smplx_vertices_incam": smpl_out.vertices}, sequence_id=vid
                    )
                    pred_c_verts = torch.stack(
                        [torch.matmul(self.smplx2smpl, v_) for v_ in smpl_out.vertices]
                    )
                    del smpl_out

                pred_c_j3d = einsum(self.J_regressor, pred_c_verts, "j v, l v i -> l j i")
                check_finite_metric_inputs(
                    {"pred_c_verts": pred_c_verts, "pred_c_j3d": pred_c_j3d},
                    sequence_id=vid,
                )
                sequence_metrics = compute_camcoord_metrics(
                    {
                        "pred_j3d": pred_c_j3d,
                        "target_j3d": target_c_j3d,
                        "pred_verts": pred_c_verts,
                        "target_verts": target_c_verts,
                    },
                    mask=mask,
                    sequence_id=vid,
                )
            else:
                if "pred_smpl_vertices_global" in outputs:
                    pred_ay_verts = outputs["pred_smpl_vertices_global"]
                else:
                    pred_smpl_params_global = outputs["pred_body_params_global"]
                    smpl_out = self.smplx(**pred_smpl_params_global)
                    check_finite_metric_inputs(
                        {"pred_smplx_vertices_global": smpl_out.vertices}, sequence_id=vid
                    )
                    pred_ay_verts = torch.stack(
                        [torch.matmul(self.smplx2smpl, v_) for v_ in smpl_out.vertices]
                    )
                    del smpl_out

                pred_ay_j3d = einsum(self.J_regressor, pred_ay_verts, "j v, l v i -> l j i")
                check_finite_metric_inputs(
                    {"pred_ay_verts": pred_ay_verts, "pred_ay_j3d": pred_ay_j3d},
                    sequence_id=vid,
                )
                sequence_metrics = compute_global_metrics(
                    {
                        "pred_j3d_glob": pred_ay_j3d,
                        "target_j3d_glob": target_w_j3d,
                        "pred_verts_glob": pred_ay_verts,
                        "target_verts_glob": target_w_verts,
                    },
                    mask=mask,
                    sequence_id=vid,
                )

            converted = {key: as_np_array(value) for key, value in sequence_metrics.items()}
            for key, value in converted.items():
                self.metric_aggregator[key][vid] = value
        except InvalidMetricInputError as error:
            self._handle_invalid_sequence(trainer, outputs, batch, vid, error)

    def on_predict_epoch_start(self, trainer, pl_module):
        self._reset_epoch_state()

    @staticmethod
    def _check_body_params(params, prefix, sequence_id):
        names = ("body_pose", "global_orient", "transl", "betas")
        tensors = {
            f"{prefix}.{name}": params[name]
            for name in names
            if name in params and isinstance(params[name], torch.Tensor)
        }
        check_finite_metric_inputs(tensors, sequence_id=sequence_id)

    def _handle_invalid_sequence(self, trainer, outputs, batch, vid, error):
        rank = int(getattr(trainer, "global_rank", 0))
        key = f"rank{rank}:{self.target_dataset_id}:{vid}"
        self.invalid_sequences[key] = str(error)
        action = "raise" if self.invalid_policy == "raise" else "skip"
        Log.warning(
            f"[EvalGuard][{self.target_dataset_id}] {action} invalid sequence\n"
            f"vid={vid}\nstep={getattr(trainer, 'global_step', 0)}\n"
            f"epoch={getattr(trainer, 'current_epoch', 0)}\nrank={rank}\nreason={error}"
        )
        if self.dump_invalid:
            dump_path = dump_invalid_eval_sample(
                trainer,
                self.target_dataset_id,
                vid,
                str(error),
                outputs,
                batch=batch,
                tensor_diagnostics=error.diagnostics,
            )
            if dump_path is not None:
                Log.warning(f"[EvalGuard][{self.target_dataset_id}] dump={dump_path}")
        apply_invalid_policy(self.invalid_policy, error)

    def _reset_epoch_state(self):
        for key in self.metric_aggregator:
            self.metric_aggregator[key] = {}
        self.invalid_sequences = {}

    # ================== Epoch Summary  ================== #
    def on_predict_epoch_end(self, trainer, pl_module):
        global_rank = int(getattr(trainer, "global_rank", 0))
        if "mpjpe" in self.metric_aggregator:
            monitor_metric = "mpjpe"
        else:
            monitor_metric = list(self.metric_aggregator.keys())[0]

        metric_keys = list(self.metric_aggregator)
        with torch.inference_mode(False):
            states = all_gather(
                {"metrics": self.metric_aggregator, "invalid": self.invalid_sequences}
            )
        gathered_metrics = {key: {} for key in metric_keys}
        gathered_invalid = {}
        for state in states:
            for metric_key in metric_keys:
                gathered_metrics[metric_key].update(state["metrics"][metric_key])
            gathered_invalid.update(state["invalid"])
        self.metric_aggregator = gathered_metrics
        self.invalid_sequences = gathered_invalid

        if global_rank == 0 and self.invalid_sequences:
            lines = [
                f"{key}: {reason.splitlines()[0][:300]}"
                for key, reason in list(self.invalid_sequences.items())[:10]
            ]
            Log.warning(
                f"[EvalGuard][{self.target_dataset_id}] invalid sequences="
                f"{len(self.invalid_sequences)}\n" + "\n".join(lines)
            )

        total = len(self.metric_aggregator[monitor_metric])
        if global_rank == 0:
            Log.info(f"{total} sequences evaluated in {self.__class__.__name__}")
        if total == 0:
            if global_rank == 0:
                Log.warning(
                    f"[EvalGuard][{self.target_dataset_id}] no valid sequences; metrics skipped"
                )
            self._reset_epoch_state()
            return

        # print monitored metric per sequence
        mm_per_seq = {k: v.mean() for k, v in self.metric_aggregator[monitor_metric].items()}
        if len(mm_per_seq) > 0:
            sorted_mm_per_seq = sorted(mm_per_seq.items(), key=lambda x: x[1], reverse=True)
            n_worst = 5 if trainer.state.stage == "validate" else len(sorted_mm_per_seq)
            if global_rank == 0:
                Log.info(
                    f"monitored metric {monitor_metric} per sequence\n"
                    + "\n".join([f"{m:5.1f} : {s}" for s, m in sorted_mm_per_seq[:n_worst]])
                    + "\n------"
                )

        # average over all batches
        metrics_avg = {
            k: np.concatenate(list(v.values())).mean() for k, v in self.metric_aggregator.items()
        }
        if global_rank == 0:
            Log.info(
                f"[Metrics] {self.target_dataset_id}:\n"
                + "\n".join(f"{k}: {v:.1f}" for k, v in metrics_avg.items())
                + "\n------"
            )

        # save to logger if available
        if global_rank == 0 and pl_module.logger is not None:
            cur_epoch = pl_module.current_epoch
            for k, v in metrics_avg.items():
                pl_module.logger.log_metrics(
                    {f"val_metric_{self.target_dataset_id}/{k}": v}, step=cur_epoch
                )

        self._reset_epoch_state()
