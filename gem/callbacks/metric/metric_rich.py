# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NVIDIA-OneWay-Noncommercial
"""RICH 验证指标回调。

完整训练配置会用它评估 RICH 数据集，既计算相机坐标下的重建指标，也计算全局坐标
相关的运动指标，帮助判断模型在复杂真实场景中的泛化效果。
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


class MetricRICH(pl.Callback):
    """Metric callback for RICH (and RICH-OCC) dataset.

    Computes both camera-coordinate and global metrics.
    """

    def __init__(self, occ=False, invalid_policy="skip", dump_invalid=True):
        super().__init__()
        self.occ = occ
        self.invalid_policy = validate_invalid_policy(invalid_policy)
        self.dump_invalid = bool(dump_invalid)
        self.invalid_sequences: dict[str, str] = {}
        # vid->result
        self.metric_aggregator = {
            "pa_mpjpe": {},
            "mpjpe": {},
            "pve": {},
            "accel": {},
            "wa2_mpjpe": {},
            "waa_mpjpe": {},
            "rte": {},
            "jitter": {},
            "fs": {},
        }

        # SMPL-X for prediction and GT FK
        self.smplx_model = {
            "male": make_smplx("supermotion"),
            "female": make_smplx("supermotion"),
            "neutral": make_smplx("supermotion"),
        }
        self.J_regressor = torch.load(f"{_BODY_MODEL_DIR}/smpl_neutral_J_regressor.pt")
        self.smplx2smpl = torch.load(f"{_BODY_MODEL_DIR}/smplx2smpl_sparse.pt")

        # The metrics are calculated similarly for val/test/predict
        self.on_test_batch_end = self.on_validation_batch_end = self.on_predict_batch_end
        self.on_test_epoch_end = self.on_validation_epoch_end = self.on_predict_epoch_end
        self.on_test_epoch_start = self.on_validation_epoch_start = self.on_predict_epoch_start

    # ================== Batch-based Computation  ================== #
    def on_predict_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        """The behaviour is the same for val/test/predict"""
        assert batch["B"] == 1
        dataset_id = batch["meta"][0]["dataset_id"]
        if self.occ and dataset_id != "RICH-OCC":
            return
        elif (not self.occ) and dataset_id != "RICH":
            return

        # Move to cuda if not
        for g in ["male", "female", "neutral"]:
            self.smplx_model[g] = self.smplx_model[g].cuda()
        self.J_regressor = self.J_regressor.cuda()
        self.smplx2smpl = self.smplx2smpl.cuda()

        vid = batch["meta"][0]["vid"]
        gender = batch["gender"][0]
        T_w2ay = batch["T_w2ay"][0]
        T_w2c = batch["T_w2c"][0]

        try:
            pred_smpl_params_incam = outputs["pred_body_params_incam"]
            pred_smpl_params_global = outputs["pred_body_params_global"]
            self._check_body_params(pred_smpl_params_incam, "pred_body_params_incam", vid)
            self._check_body_params(pred_smpl_params_global, "pred_body_params_global", vid)

            target_w_params = {k: v[0] for k, v in batch["gt_smpl_params"].items()}
            self._check_body_params(target_w_params, "target_w_params", vid)
            check_finite_metric_inputs({"T_w2c": T_w2c, "T_w2ay": T_w2ay}, sequence_id=vid)
            target_w_output = self.smplx_model[gender](**target_w_params)
            check_finite_metric_inputs(
                {"target_smplx_vertices": target_w_output.vertices}, sequence_id=vid
            )
            target_w_verts = torch.stack(
                [torch.matmul(self.smplx2smpl, vertex) for vertex in target_w_output.vertices]
            )
            target_c_verts = apply_T_on_points(target_w_verts, T_w2c)
            target_c_j3d = torch.matmul(self.J_regressor, target_c_verts)
            target_ay_verts = apply_T_on_points(target_w_verts, T_w2ay)
            target_ay_j3d = torch.matmul(self.J_regressor, target_ay_verts)
            check_finite_metric_inputs(
                {
                    "target_w_verts": target_w_verts,
                    "target_c_verts": target_c_verts,
                    "target_c_j3d": target_c_j3d,
                    "target_ay_verts": target_ay_verts,
                    "target_ay_j3d": target_ay_j3d,
                },
                sequence_id=vid,
            )

            smpl_out = self.smplx_model["neutral"](**pred_smpl_params_incam)
            check_finite_metric_inputs(
                {"pred_smplx_vertices_incam": smpl_out.vertices}, sequence_id=vid
            )
            pred_c_verts = torch.stack(
                [torch.matmul(self.smplx2smpl, vertex) for vertex in smpl_out.vertices]
            )
            pred_c_j3d = einsum(self.J_regressor, pred_c_verts, "j v, l v i -> l j i")

            smpl_out = self.smplx_model["neutral"](**pred_smpl_params_global)
            check_finite_metric_inputs(
                {"pred_smplx_vertices_global": smpl_out.vertices}, sequence_id=vid
            )
            pred_ay_verts = torch.stack(
                [torch.matmul(self.smplx2smpl, vertex) for vertex in smpl_out.vertices]
            )
            pred_ay_j3d = einsum(self.J_regressor, pred_ay_verts, "j v, l v i -> l j i")
            check_finite_metric_inputs(
                {
                    "pred_c_verts": pred_c_verts,
                    "pred_c_j3d": pred_c_j3d,
                    "pred_ay_verts": pred_ay_verts,
                    "pred_ay_j3d": pred_ay_j3d,
                },
                sequence_id=vid,
            )

            camcoord_metrics = compute_camcoord_metrics(
                {
                    "pred_j3d": pred_c_j3d,
                    "target_j3d": target_c_j3d,
                    "pred_verts": pred_c_verts,
                    "target_verts": target_c_verts,
                },
                sequence_id=vid,
            )
            global_metrics = compute_global_metrics(
                {
                    "pred_j3d_glob": pred_ay_j3d,
                    "target_j3d_glob": target_ay_j3d,
                    "pred_verts_glob": pred_ay_verts,
                    "target_verts_glob": target_ay_verts,
                },
                sequence_id=vid,
            )

            # Commit only after both camera and global metrics succeed.
            sequence_metrics = {**camcoord_metrics, **global_metrics}
            converted = {key: as_np_array(value) for key, value in sequence_metrics.items()}
            for key, value in converted.items():
                self.metric_aggregator[key][vid] = value
        except InvalidMetricInputError as error:
            self._handle_invalid_sequence(trainer, outputs, batch, dataset_id, vid, error)

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

    def _handle_invalid_sequence(self, trainer, outputs, batch, dataset_id, vid, error):
        rank = int(getattr(trainer, "global_rank", 0))
        key = f"rank{rank}:{dataset_id}:{vid}"
        self.invalid_sequences[key] = str(error)
        action = "raise" if self.invalid_policy == "raise" else "skip"
        Log.warning(
            f"[EvalGuard][{dataset_id}] {action} invalid sequence\n"
            f"vid={vid}\nstep={getattr(trainer, 'global_step', 0)}\n"
            f"epoch={getattr(trainer, 'current_epoch', 0)}\nrank={rank}\nreason={error}"
        )
        if self.dump_invalid:
            dump_path = dump_invalid_eval_sample(
                trainer,
                dataset_id,
                vid,
                str(error),
                outputs,
                batch=batch,
                tensor_diagnostics=error.diagnostics,
            )
            if dump_path is not None:
                Log.warning(f"[EvalGuard][{dataset_id}] dump={dump_path}")
        apply_invalid_policy(self.invalid_policy, error)

    def _reset_epoch_state(self):
        for key in self.metric_aggregator:
            self.metric_aggregator[key] = {}
        self.invalid_sequences = {}

    # ================== Epoch Summary  ================== #
    def on_predict_epoch_end(self, trainer, pl_module):
        global_rank = int(getattr(trainer, "global_rank", 0))
        monitor_metric = "mpjpe"

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

        dataset_label = "RICH-OCC" if self.occ else "RICH"
        if global_rank == 0 and self.invalid_sequences:
            lines = [
                f"{key}: {reason.splitlines()[0][:300]}"
                for key, reason in list(self.invalid_sequences.items())[:10]
            ]
            Log.warning(
                f"[EvalGuard][{dataset_label}] invalid sequences="
                f"{len(self.invalid_sequences)}\n" + "\n".join(lines)
            )

        total = len(self.metric_aggregator[monitor_metric])
        if global_rank == 0:
            Log.info(f"{total} sequences evaluated in {self.__class__.__name__}")
        if total == 0:
            if global_rank == 0:
                Log.warning(f"[EvalGuard][{dataset_label}] no valid sequences; metrics skipped")
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
                f"[Metrics] RICH{'_OCC' if self.occ else ''}:\n"
                + "\n".join(f"{k}: {v:.1f}" for k, v in metrics_avg.items())
                + "\n------"
            )

        # save to logger if available
        if global_rank == 0 and pl_module.logger is not None:
            cur_epoch = pl_module.current_epoch
            for k, v in metrics_avg.items():
                pl_module.logger.log_metrics(
                    {f"val_metric_RICH{'-OCC' if self.occ else ''}/{k}": v},
                    step=cur_epoch,
                )

        self._reset_epoch_state()
