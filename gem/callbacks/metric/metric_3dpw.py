# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NVIDIA-OneWay-Noncommercial
"""3DPW 验证指标回调。

该 callback 在 validation/test/predict 阶段消费 3DPW 完整序列输出，把预测 SMPL-X
转换到评测用 SMPL 空间，并汇总 PA-MPJPE、MPJPE、PVE、accel 等常用重建指标。
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
    dump_invalid_eval_sample,
    validate_invalid_policy,
)
from gem.utils.gather import all_gather
from gem.utils.geo_transform import apply_T_on_points
from gem.utils.pylogger import Log
from gem.utils.smplx_utils import make_smplx

_BODY_MODEL_DIR = "inputs/checkpoints/body_models"


class Metric3DPW(pl.Callback):
    def __init__(self, invalid_policy="skip", dump_invalid=True):
        super().__init__()
        self.invalid_policy = validate_invalid_policy(invalid_policy)
        self.dump_invalid = bool(dump_invalid)
        self.invalid_sequences: dict[str, str] = {}
        # vid->result
        self.metric_aggregator = {
            "pa_mpjpe": {},
            "mpjpe": {},
            "pve": {},
            "accel": {},
        }

        # SMPLX (flat_hand_mean for 3DPW) and SMPL
        self.smplx = make_smplx("supermotion_EVAL3DPW")
        self.smpl = {
            "male": make_smplx("smpl", gender="male"),
            "female": make_smplx("smpl", gender="female"),
        }
        self.J_regressor = torch.load(
            f"{_BODY_MODEL_DIR}/smpl_3dpw14_J_regressor_sparse.pt"
        ).to_dense()
        self.J_regressor24 = torch.load(f"{_BODY_MODEL_DIR}/smpl_neutral_J_regressor.pt")
        self.smplx2smpl = torch.load(f"{_BODY_MODEL_DIR}/smplx2smpl_sparse.pt")
        self.faces_smpl = self.smpl["male"].faces

        # The metrics are calculated similarly for val/test/predict
        self.on_test_batch_end = self.on_validation_batch_end = self.on_predict_batch_end
        self.on_test_epoch_end = self.on_validation_epoch_end = self.on_predict_epoch_end
        self.on_test_epoch_start = self.on_validation_epoch_start = self.on_predict_epoch_start

    # ================== Batch-based Computation  ================== #
    def on_predict_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        """The behaviour is the same for val/test/predict"""
        assert batch["B"] == 1
        dataset_id = batch["meta"][0]["dataset_id"]
        if dataset_id != "3DPW":
            return

        # Move to cuda if not
        self.smplx = self.smplx.cuda()
        for g in ["male", "female"]:
            self.smpl[g] = self.smpl[g].cuda()
        self.J_regressor = self.J_regressor.cuda()
        self.J_regressor24 = self.J_regressor24.cuda()
        self.smplx2smpl = self.smplx2smpl.cuda()

        vid = batch["meta"][0]["vid"]
        gender = batch["gender"][0]
        T_w2c = batch["gt_T_w2c"][0]
        mask = batch["mask"]["valid"][0]

        try:
            for group_name in ("pred_body_params_incam", "pred_body_params_global"):
                if group_name in outputs:
                    self._check_body_params(outputs[group_name], group_name, vid)
            target_w_params = {k: v[0] for k, v in batch["smpl_params"].items()}
            self._check_body_params(target_w_params, "target_w_params", vid)
            check_finite_metric_inputs({"T_w2c": T_w2c}, sequence_id=vid)

            target_w_output = self.smpl[gender](**target_w_params)
            target_w_verts = target_w_output.vertices
            target_c_verts = apply_T_on_points(target_w_verts, T_w2c)
            target_c_j3d = torch.matmul(self.J_regressor, target_c_verts)
            check_finite_metric_inputs(
                {
                    "target_w_verts": target_w_verts,
                    "target_c_verts": target_c_verts,
                    "target_c_j3d": target_c_j3d,
                },
                sequence_id=vid,
            )

            smpl_out = self.smplx(**outputs["pred_body_params_incam"])
            check_finite_metric_inputs(
                {"pred_smplx_vertices_incam": smpl_out.vertices}, sequence_id=vid
            )
            pred_c_verts = torch.stack(
                [torch.matmul(self.smplx2smpl, vertex) for vertex in smpl_out.vertices]
            )
            pred_c_j3d = einsum(self.J_regressor, pred_c_verts, "j v, l v i -> l j i")
            del smpl_out
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
                pelvis_idxs=[2, 3],
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
        dataset_id = "3DPW"
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
        monitor_metric = "pa_mpjpe"

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
                f"[EvalGuard][3DPW] invalid sequences={len(self.invalid_sequences)}\n"
                + "\n".join(lines)
            )

        total = len(self.metric_aggregator[monitor_metric])
        if global_rank == 0:
            Log.info(f"{total} sequences evaluated in {self.__class__.__name__}")
        if total == 0:
            if global_rank == 0:
                Log.warning("[EvalGuard][3DPW] no valid sequences; metrics skipped")
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
                "[Metrics] 3DPW:\n"
                + "\n".join(f"{k}: {v:.1f}" for k, v in metrics_avg.items())
                + "\n------"
            )

        # save to logger if available
        if global_rank == 0 and pl_module.logger is not None:
            cur_epoch = pl_module.current_epoch
            for k, v in metrics_avg.items():
                pl_module.logger.log_metrics({f"val_metric_3DPW/{k}": v}, step=cur_epoch)

        self._reset_epoch_state()
