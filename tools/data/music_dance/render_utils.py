# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NVIDIA-OneWay-Noncommercial
"""保留音乐数据来源核验所需的离线人体网格渲染函数。

AIOZ、CoMPAS3D、FineDance 的数据校验工具仍需渲染上游人体动作。这里从已退役的
SMPL Demo 工具中原样迁入全局网格渲染函数，保持相机、地面和颜色语义；本模块
不加载生成模型、人体估计模型、文本编码器或 Webcam。Open3D 只在实际渲染时导入。
"""

from __future__ import annotations

import numpy as np
import torch
from tqdm import tqdm


def render_global_frames(
    verts_global: torch.Tensor,
    smpl_faces: torch.Tensor,
    W: int,
    H: int,
) -> np.ndarray:
    """Render SMPL mesh in global space with ground plane.

    Args:
        verts_global: (L, V, 3) SMPL vertices in normalized global space
        smpl_faces: (F, 3) face indices
        W, H: frame dimensions

    Returns:
        global_frames: (L, H, W, 3) RGB uint8
    """
    import open3d as o3d

    from gem.utils.cam_utils import create_camera_sensor
    from gem.utils.vis.o3d_render import Settings, create_meshes, get_ground
    from gem.utils.vis.renderer import (
        get_global_cameras_static_v2,
        get_ground_params_from_points,
    )

    L = verts_global.shape[0]

    mat_settings = Settings()
    lit_mat = mat_settings._materials[Settings.LIT]

    # Ground plane parameters
    root_points = verts_global.mean(1)  # (L, 3)
    scale, cx, cz = get_ground_params_from_points(root_points, verts_global)

    # Global camera
    _, _, K_global = create_camera_sensor(W, H, fov_deg=32)
    position, target_center, up_vec = get_global_cameras_static_v2(
        verts_global.clone(),
        beta=6.0,
        cam_height_degree=30,
    )

    renderer = o3d.visualization.rendering.OffscreenRenderer(W, H)
    renderer.scene.set_background([1.0, 1.0, 1.0, 1.0])
    renderer.scene.set_lighting(
        renderer.scene.LightingProfile.NO_SHADOWS, np.array([0.577, -0.577, -0.577])
    )
    renderer.scene.camera.set_projection(
        K_global.cpu().double().numpy(), 0.1, 100.0, float(W), float(H)
    )
    renderer.scene.camera.look_at(
        target_center.cpu().numpy(), position.cpu().numpy(), up_vec.cpu().numpy()
    )

    # Add ground mesh
    ground = get_ground(max(scale, 3) * 1.5, cx, cz)
    gv, gf, gc = ground
    ground_mesh = create_meshes(gv, gf, gc[..., :3])
    ground_mat = o3d.visualization.rendering.MaterialRecord()
    ground_mat.shader = Settings.LIT
    renderer.scene.add_geometry("mesh_ground", ground_mesh, ground_mat)

    color_global = torch.tensor([0.69019608, 0.39215686, 0.95686275])
    global_frames = []
    for i in tqdm(range(L), desc="Global render", leave=False):
        mesh = create_meshes(verts_global[i], smpl_faces, color_global)
        if i > 0:
            renderer.scene.remove_geometry(f"mesh_{i - 1}")
        renderer.scene.add_geometry(f"mesh_{i}", mesh, lit_mat)
        global_frames.append(np.array(renderer.render_to_image()))

    return np.stack(global_frames)
