"""把配对评价的真实执行和实际下发参考导出为可审计的渲染输入。

输入来自同一次50Hz真实物理评价：控制结束状态、该步执行前选定的未来第一帧参考、
原始GENMO名义参考、动作及误差。按真实存活前缀截断，绝不复制失败后的状态。
RobotSpec关节顺序通过名称映射到MJCF原生qpos28，四元数保持wxyz；所有数组的
第一帧对应音乐0.02秒。每首保存NPZ和包含初态、物理参数、音乐与文件SHA的JSON。
本模块只在显式请求轨迹时使用，不参与训练、奖励、参考投影或动力学推进；渲染器
只回放这些实际轨迹，不能将回放称为新一次MuJoCo动力学实验。
"""

import hashlib
import json
from pathlib import Path

import numpy as np
import torch


def export_trajectories(directory, collector, selected, initial, initial_xi, states, live,
                        corrected_frames, action_frames, raw, reasons, *, zero):
    import mujoco
    from supermimic.rotations import mat_to_quat
    from .paired_evaluation import tensor_digest

    output = Path(directory)
    output.mkdir(parents=True, exist_ok=False)
    bank, spec = collector.bank, collector.control.bundle.spec
    model = mujoco.MjModel.from_xml_path(str(spec.mjcf_path))
    names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
             for i in range(model.njnt) if model.jnt_qposadr[i] >= 7]
    order = [spec.joint_names.index(name) for name in names]
    if model.nq != 28 or len(names) != 21:
        raise ValueError('渲染资产必须为自由根和21个具名关节')
    corrected = torch.stack(corrected_frames) if corrected_frames else initial.new_empty(0, len(selected), 28)
    actions = torch.stack(action_frames) if action_frames else initial.new_empty(0, len(selected), 21)
    for column, song in enumerate(selected.tolist()):
        task = bank.tasks[song]
        n = int(live[:, column].sum())
        state = states[:n, column]
        frame = torch.arange(1, n+1, device=bank.device)
        nominal = bank.ref(torch.full_like(frame, song), frame)
        nominal_pose = torch.cat((nominal.root_p, mat_to_quat(nominal.root_R), nominal.q), -1)
        native = lambda value: torch.cat((value[:, :7], value[:, 7:][:, order]), -1).cpu().numpy()
        arrays = dict(qpos=torch.cat((state[:, :7], state[:, 13:34][:, order]), -1).cpu().numpy(),
                      nominal_qpos=native(nominal_pose), corrected_qpos=native(corrected[:n, column]),
                      state=state.cpu().numpy(), action=actions[:n, column].cpu().numpy(),
                      tick=np.arange(1, n+1, dtype=np.int64)*12)
        arrays.update({'raw_'+key: value[:n, column].cpu().numpy()
                       for key, value in raw.items() if value.ndim == 2})
        stem = f'{column:03d}_{task["id"]}'
        path = output/(stem+'.npz')
        np.savez_compressed(path, **arrays)
        metadata = dict(schema='genmo.adapter.paired_physics_trajectory.v1', task=task,
            trajectory_file=path.name, sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            fps=50, quaternion_convention='wxyz', qpos_order='mujoco_native', joint_names=names,
            actual_control_steps=n, planned_steps=int(bank.ends[song]), end_reason=reasons[column],
            zero_adapter=zero, initial_state_sha256=tensor_digest(initial[column]),
            physical_xi_sha256=tensor_digest(initial_xi[column]), mjcf=str(spec.mjcf_path),
            mjcf_sha256=hashlib.sha256(Path(spec.mjcf_path).read_bytes()).hexdigest(),
            physics='IsaacLab3_Newton_MuJoCoWarp_200Hz_50Hz_control',
            reference_semantics='future_first_pose_committed_before_this_physics_step',
            audio_start_seconds=task.get('music_start_tick', 0)/600+.02)
        (output/(stem+'.json')).write_text(json.dumps(metadata, ensure_ascii=False, indent=2))
