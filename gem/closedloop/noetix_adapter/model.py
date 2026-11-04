"""为 noetix 750/606 Actor 生成十帧独立姿态残差的可微参考 Adapter。

输入是冻结 Actor 归一化后的观测历史750、名义目标606，以及16帧 EDGE35 音乐、
有效标记和相对时间，共1948维；不使用真实世界位置、物理随机化或未来执行真值。
三层512/512/256 SiLU网络零初始化270维输出，逐帧修正未来1..10帧的根位置、
根旋转及21关节。当前与左支持点保持名义姿态；第10帧提供目标第9帧的中心差分。
这是新基座的无状态参考接口，不兼容旧2654维、带跨步提交记忆的Adapter权重。

Actor原生窗口为当前0..9帧，重新编码为traj60×10加当前anchor朝向6；关节速度、
根速度和角速度按姿态修正的中心差分增量更新，不能直接把270维当作动作残差。
名义目标加特征差值保证零残差保留原生编码；Actor权重和RunningNorm不更新，
但保留其输入梯度以便WM上的BPTT。实际执行器延迟、平滑、PD由原训练环境负责。
"""

import torch
from torch import nn

from supermimic.kinematics.rotations import (
    projected_gravity, quat_apply, quat_apply_inverse, quat_from_rotvec,
    quat_inv, quat_mul, rot6d_from_quat, rotvec_from_quat, heading_quat as yaw_quat,
)

GOAL_SIZE = 600
POSE_SIZE = 12 * 28
MUSIC_SIZE = 16 * 35 + 16 + 16
INPUT_SIZE = 750 + 606 + MUSIC_SIZE


def apply_rotation_residual(quat, rotvec):
    """直接按Hamilton定义乘旋转增量，使零增量精确保留输入四元数。

    上游合并乘法公式含大数抵消，乘单位旋转也会引入舍入误差；这里保留
    标量/向量乘积形式，不做零点分支或detach，确保零点梯度连通。
    """
    increment = quat_from_rotvec(rotvec)
    w, xyz = quat[..., :1], quat[..., 1:]
    dw, dxyz = increment[..., :1], increment[..., 1:]
    return torch.cat((w * dw - (xyz * dxyz).sum(-1, keepdim=True),
                      w * dxyz + dw * xyz + torch.cross(xyz, dxyz, dim=-1)), dim=-1)


def correct_goal(goal, poses, residual):
    """将未来1..10帧残差转成原生0..9帧Goal；所有旋转和导数保持可微。"""
    base = goal[..., :600].reshape(-1, 10, 60)
    quat = poses[..., 3:7]
    heading = yaw_quat(quat[:, 1:2])
    d = torch.cat((residual.new_zeros(len(residual), 2, 27), residual), dim=1)
    dp = quat_apply(heading.expand(-1, 12, -1), d[..., :3])
    new_quat = apply_rotation_residual(quat, d[..., 3:6])
    cur, new = quat[:, 1:11], new_quat[:, 1:11]
    dq = d[:, 1:11, 6:]
    dqd = (d[:, 2:, 6:] - d[:, :-2, 6:]) / .04
    dv = (dp[:, 2:] - dp[:, :-2]) / .04
    old_w = rotvec_from_quat(quat_mul(quat[:, 2:], quat_inv(quat[:, :-2]))) / .04
    new_w = rotvec_from_quat(quat_mul(new_quat[:, 2:], quat_inv(new_quat[:, :-2]))) / .04
    w = quat_apply(cur, base[..., 45:48])
    v = quat_apply(yaw_quat(cur), base[..., 48:51])
    # 用同一计算路径相减，零残差时不让旋转往返误差改变原始Goal。
    dw_local = quat_apply_inverse(new, w + (new_w - old_w)) - quat_apply_inverse(cur, w)
    dv_local = quat_apply_inverse(yaw_quat(new), v + dv) - quat_apply_inverse(yaw_quat(cur), v)
    h = heading.expand(-1, 10, -1)
    rotation_delta = rot6d_from_quat(quat_mul(quat_inv(h), new)) - rot6d_from_quat(quat_mul(quat_inv(h), cur))
    delta = torch.cat((dq, dqd, projected_gravity(new) - projected_gravity(cur),
                       dw_local, dv_local, d[:, 1:11, :3], rotation_delta), dim=-1)
    return torch.cat(((base + delta).flatten(1), goal[..., -6:]), dim=-1)


class ReferencePolicy(nn.Module):
    """封装冻结Actor和唯一可训练的参考网络；保持原生Policy调用形式。"""

    def __init__(self, actor, joint_names):
        super().__init__()
        self.actor = actor.eval().requires_grad_(False)
        self.cfg = actor.cfg
        self.net = nn.Sequential(nn.Linear(INPUT_SIZE, 512), nn.SiLU(),
                                 nn.Linear(512, 512), nn.SiLU(),
                                 nn.Linear(512, 256), nn.SiLU(), nn.Linear(256, 270))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)
        bounds = [0., 0., .02, .05, .05, .05]
        bounds += [.05 if "ankle" in name else .1 for name in joint_names]
        self.register_buffer("bounds", torch.tensor(bounds))

    def update_input_norms(self, *args, **kwargs):
        """底层Actor及归一化统计完全冻结，新Adapter不拟合在线归一化。"""

    def references(self, obs, packed):
        nominal = torch.cat((packed[:, :600], packed[:, -6:]), dim=-1)
        poses = packed[:, 600:600 + POSE_SIZE].reshape(-1, 12, 28)
        music = packed[:, 600 + POSE_SIZE:-6]
        x = torch.cat((self.actor.decoder.norm(obs), self.actor.encoders['motion'].norm(nominal), music), dim=-1)
        d = self.net(x).reshape(-1, 10, 27).tanh() * self.bounds
        rot = d[..., 3:6]
        rot = rot / (rot.norm(dim=-1, keepdim=True) / .05).clamp_min(1.)
        d = torch.cat((d[..., :3], rot, d[..., 6:]), dim=-1)
        return correct_goal(nominal, poses, d), d

    def forward(self, obs_hist, goal, interface="motion"):
        corrected, _ = self.references(obs_hist, goal)
        return self.actor(obs_hist, corrected)


class ReferenceLibrary:
    """给原生想象循环提供名义Goal、真实姿态支持和同歌音乐；奖励仍读取名义参考。"""

    def __init__(self, library, music):
        self.library = library
        self.music = music.to(library.device)
        if len(music) != library.num_frames:
            raise ValueError("音乐与动作必须逐帧对齐")

    def __getattr__(self, name):
        return getattr(self.library, name)

    def goal_window(self, frame, k, mode="traj", *, goal_root_height=False):
        nominal = self.library.goal_window(frame, k, mode, goal_root_height=goal_root_height)
        clip = self.library.frame_clip[frame]
        idx = frame[:, None] + torch.arange(-1, 11, device=frame.device)[None]
        idx = torch.maximum(self.library.clip_start[clip, None], idx)
        idx = torch.minimum(self.library.clip_last[clip, None], idx)
        r = self.library.ref(idx.flatten())
        poses = torch.cat((r.root_pos_w, r.root_quat_w, r.joint_pos), dim=-1).reshape(len(frame), -1)
        return torch.cat((nominal, poses, self.music[frame]), dim=-1)
