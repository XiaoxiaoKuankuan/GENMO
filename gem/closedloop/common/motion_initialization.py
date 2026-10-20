"""把已核验的配对动作转换为训练episode的运动初态和预热参考。

音乐start_frame对应episode第600 tick；reset时从其前一秒的示范姿态/速度开始，
由冻结GMT实际控制50步形成真实历史。前后参考仍使用统一30Hz源、50Hz插值和
600Hz时间轴，后端负责从同一参考派生初始速度，禁止另造不一致的速度数组。
只去除reset帧的水平平移和yaw，保留绝对根高及相对运动；源片段外的预热支持
显式保持边界姿态并记录数量，不延长音乐真实终点或伪造已经执行的本体历史。

该数据只用于显式训练reset的bootstrap。Actor继续只接收其既有音乐、已承诺
参考前缀和实际本体历史；不会额外收到完整配对未来动作。验证/完整MC/部署的
站立启动不调用本模块；显式match_training_reset完整MC允许val配对reset，仅供价值验证。所有数组均复制，不修改数据集或活动度监督对象。
"""
from __future__ import annotations

import copy
import math

import numpy as np

from .time_sampling import MOTION_RESET_CONTRACT


def make_motion_initialization(qpos, source, *, start_frame, plan_frames, phase="train"):
    qpos = np.asarray(qpos, dtype=np.float64)
    if (qpos.ndim != 2 or qpos.shape[1] != 28 or len(qpos) < 2 or not np.isfinite(qpos).all()
            or np.max(np.abs(np.linalg.norm(qpos[:, 3:7], axis=1) - 1)) > 1e-3):
        raise ValueError('Motion initialization requires verified finite qpos28 with unit wxyz quaternions')
    if (type(start_frame) is not int or not 0 <= start_frame <= len(qpos) - 2
            or type(plan_frames) is not int or plan_frames not in (60, 120)):
        raise ValueError('Invalid motion initialization source start or reference horizon')
    allowed_split=(source.get('split') in ('train','val') if phase=='critic_mc' else
        phase=='train' and source.get('split')=='train')
    if not allowed_split or source.get('music_start_frame') != start_frame:
        raise ValueError('Motion initialization must match the selected training music offset')
    names = source.get('source_joint_order')
    if not isinstance(names, list) or len(names) != 21 or len(set(names)) != 21:
        raise ValueError('Motion initialization requires explicit source joint order')
    # -0.2秒是GMT回看；第0秒重置，第1秒交给GENMO，对应音乐start_frame。
    indices = start_frame - 30 + np.arange(-6, plan_frames)
    values = qpos[np.clip(indices, 0, len(qpos) - 1)].copy()
    anchor = values[6].copy()
    w, x, y, z = anchor[3:7] / np.linalg.norm(anchor[3:7])
    yaw = math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    c, s = math.cos(yaw), math.sin(yaw)
    xy = values[:, :2] - anchor[:2]
    values[:, 0] = c * xy[:, 0] + s * xy[:, 1]
    values[:, 1] = -s * xy[:, 0] + c * xy[:, 1]
    quat = values[:, 3:7].copy()
    quat /= np.linalg.norm(quat, axis=1, keepdims=True)
    c, s = math.cos(-yaw / 2), math.sin(-yaw / 2)
    values[:, 3] = c * quat[:, 0] - s * quat[:, 3]
    values[:, 4] = c * quat[:, 1] - s * quat[:, 2]
    values[:, 5] = c * quat[:, 2] + s * quat[:, 1]
    values[:, 6] = c * quat[:, 3] + s * quat[:, 0]
    return dict(contract=MOTION_RESET_CONTRACT, phase=phase, music_start_frame=start_frame,
        music_start_tick=600, source_joint_names=list(names), source_begin_tick=-120,
        source_step_tick=20, bootstrap_qpos=values,
        left_boundary_hold_frames=int((indices < 0).sum()),
        right_boundary_hold_frames=int((indices >= len(qpos)).sum()),
        anchor_xy_yaw=[*anchor[:2].tolist(), yaw], source=copy.deepcopy(source))
