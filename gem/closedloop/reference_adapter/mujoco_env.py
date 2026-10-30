"""当前 SuperMimic RobotSpec 的单环境 MuJoCo 动力学与蓝色参考影子。

实际机器人仅在 reset 写入初态，随后由冻结 Actor 的归一化动作经过权威
default_q + action_scales * action、隐式 PD 和力矩上限推进 mj_step。
控制周期 20ms，物理子步 5ms。state55 使用 SM 关节名称顺序及世界角速度，
在 MuJoCo 自由关节的局部角速度边界显式转换；不复制另一套机器人 PD 常量。
参考影子持有独立 MjData，只做 mj_forward 和可视化，不参与碰撞、不扶正实体。
本模块不导入 Isaac、不训练网络；不同求解器的接触数值并不保证与 Newton 相同。
"""

from pathlib import Path
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
import torch


class MujocoPhysicalEnv:
    def __init__(self, spec, device="cpu"):
        self.spec, self.device = spec, torch.device(device)
        path = Path(spec.mjcf_path).resolve()
        root = ET.parse(path).getroot()
        compiler = root.find("compiler")
        if compiler is None:
            compiler = ET.SubElement(root, "compiler")
        # 模型通过字符串加载，网格仍按原 MJCF 的相对路径定位。
        compiler.set("meshdir", str((path.parent / compiler.get("meshdir", ".")).resolve()))
        option = root.find("option")
        if option is None:
            option = ET.SubElement(root, "option")
        option.set("timestep", ".005")
        option.set("integrator", "implicitfast")
        option.set("gravity", "0 0 -9.81")
        world = root.find("worldbody")
        for geom in world.findall(".//body//geom"):
            # 原资产可视网格与足端碰撞均保留，只禁用机器人自碰撞。
            if geom.get("contype", "1") != "0" or geom.get("conaffinity", "1") != "0":
                geom.set("contype", "2")
                geom.set("conaffinity", "1")
                geom.set("friction", "1 0.005 0.0001")
        for geom in world.findall("geom"):
            if geom.get("type") == "plane":
                world.remove(geom)
        ET.SubElement(world, "geom", name="sim2sim_ground", type="plane", size="0 0 .1",
                      contype="1", conaffinity="2", friction="1 .005 .0001",
                      rgba=".65 .70 .76 1")
        ET.SubElement(world, "light", pos="0 0 5", dir="0 0 -1", directional="true")
        old = root.find("actuator")
        if old is not None:
            root.remove(old)
        actuators = ET.SubElement(root, "actuator")
        for i, name in enumerate(spec.joint_names):
            effort = spec.effort_limits[i]
            ET.SubElement(actuators, "general", name="sim2sim_"+name, joint=name,
                          gaintype="fixed", biastype="affine", gainprm=str(spec.kp[i]),
                          biasprm=f"0 {-spec.kp[i]} {-spec.kd[i]}",
                          forcelimited="true", forcerange=f"{-effort} {effort}")
        self.model = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))
        self.data = mujoco.MjData(self.model)
        joints = [self.model.joint(name).id for name in spec.joint_names]
        self.qadr = self.model.jnt_qposadr[joints]
        self.vadr = self.model.jnt_dofadr[joints]
        self.model.dof_armature[self.vadr] = spec.armature
        self.root_id = self.model.body(spec.root_body).id
        self.default_q = np.asarray(spec.default_joint_pos)
        self.alpha = np.asarray(spec.action_scales)
        self.xi = torch.zeros(1, spec.xi_dim, device=self.device)
        self.xi[:, 0] = 1.  # 无随机化，与实测摩擦系数一致。
        self.previous_action = torch.zeros(1, spec.num_joints, device=self.device)
        self.control_steps = 0

    def reset(self, initial):
        state = initial.detach().cpu().numpy().reshape(55)
        if not np.isfinite(state).all() or np.linalg.norm(state[3:7]) < 1e-8:
            raise ValueError("MuJoCo 初态必须是有限的 SM state55")
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[:3] = state[:3]
        self.data.qpos[3:7] = state[3:7]/np.linalg.norm(state[3:7])
        self.data.qpos[self.qadr] = state[13:34]
        R = np.empty(9)
        mujoco.mju_quat2Mat(R, self.data.qpos[3:7])
        self.data.qvel[:3] = state[7:10]
        self.data.qvel[3:6] = R.reshape(3, 3).T @ state[10:13]
        self.data.qvel[self.vadr] = state[34:55]
        self.data.ctrl[:] = state[13:34]
        self.previous_action.zero_()
        self.control_steps = 0
        mujoco.mj_forward(self.model, self.data)
        return self.observe()

    def observe(self):
        R = self.data.xmat[self.root_id].reshape(3, 3)
        state = np.concatenate((self.data.qpos[:7], self.data.qvel[:3], R@self.data.qvel[3:6],
                                self.data.qpos[self.qadr], self.data.qvel[self.vadr]))
        return dict(state=torch.as_tensor(state, dtype=torch.float32, device=self.device)[None],
                    xi=self.xi, previous_action=self.previous_action)

    def step_physics(self, action):
        executed = action.detach().cpu().numpy().reshape(self.spec.num_joints)
        if not np.isfinite(executed).all():
            raise FloatingPointError("拒绝非有限控制动作")
        bad = (mujoco.mjtWarning.mjWARN_BADQPOS, mujoco.mjtWarning.mjWARN_BADQVEL,
               mujoco.mjtWarning.mjWARN_BADQACC)
        warning_counts = [self.data.warning[k].number for k in bad]
        self.data.ctrl[:] = self.default_q + self.alpha*executed
        for _ in range(4):
            mujoco.mj_step(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)
        if any(self.data.warning[k].number > n for k, n in zip(bad, warning_counts)):
            raise FloatingPointError("MuJoCo 报告状态/加速度数值异常，不接受求解器自动复位后的状态")
        if not np.isfinite(self.data.qpos).all() or not np.isfinite(self.data.qvel).all():
            raise FloatingPointError("MuJoCo 物理状态非有限")
        self.previous_action.copy_(action)
        self.control_steps += 1
        return self.observe()["state"]


class NominalGhost:
    """原始 GENMO 名义参考的独立可视数据；不修改实体 qpos 或物理模型。"""
    def __init__(self, env):
        self.env = env
        self.data = mujoco.MjData(env.model)
        self.scene = mujoco.MjvScene(env.model, maxgeom=1500)
        self.option = mujoco.MjvOption()

    def update(self, reference):
        p, R, q = (value[0].detach().cpu().numpy() for value in
                   (reference.root_p, reference.root_R, reference.q))
        self.data.qpos[:3] = p
        mujoco.mju_mat2Quat(self.data.qpos[3:7], np.asarray(R, dtype=np.float64).ravel())
        self.data.qpos[self.env.qadr] = q
        mujoco.mj_forward(self.env.model, self.data)

    def append(self, scene, camera):
        mujoco.mjv_updateScene(self.env.model, self.data, self.option, None, camera,
                              mujoco.mjtCatBit.mjCAT_DYNAMIC, self.scene)
        for source in self.scene.geoms[:self.scene.ngeom]:
            if source.objtype != mujoco.mjtObj.mjOBJ_GEOM or source.rgba[3] <= 0:
                continue
            if scene.ngeom == scene.maxgeom:
                break
            dest = scene.geoms[scene.ngeom]
            mujoco.mjv_initGeom(dest, source.type, source.size.astype(np.float64),
                               source.pos.astype(np.float64), source.mat.astype(np.float64).ravel(),
                               np.array([.15, .65, .95, .25], dtype=np.float32))
            for field in ("dataid", "objtype", "objid", "texcoord", "modelrbound"):
                setattr(dest, field, getattr(source, field))
            dest.matid = -1
            dest.category = mujoco.mjtCatBit.mjCAT_DECOR
            dest.transparent = 1
            scene.ngeom += 1
