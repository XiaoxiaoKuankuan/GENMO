"""真实 Newton 多环境采集的薄适配器与设备端接触统计。

物理创建、独立槽位 reset、50Hz 执行器和四个 200Hz 子步完全复用冻结 SM bridge 的
StreamingPhysicalEnv，不创建另一套动力学、关节映射、FK 或旋转算法。各环境的音乐、
参考和历史由采集器管理；这里提供批量真实状态与动作接口，不自动复位失败轨迹。

MuJoCo-Warp 的 contact.worldid 是世界编号，geom 是每世界共享模型中的几何体编号；
两者不能互换。GPU kernel 仅归约 nacon 指定的有效槽位，按旧 contact_events 的
dist < includemargin 和合法 geom 条件生成每世界 [接触数, 左脚接触, 右脚接触]。
几何体到脚部的名称映射只在初始化时读取一次 CPU 模型，不逐控制步复制所有接触。

求解器诊断通过原 Newton manager 的图内回调累计每个物理子步的容量/状态信息。
CPU 只在采集块边界读取小型统计；容量溢出、非法接触索引不允许作为有效 sidecar 保存。
求解迭代上限与内存容量溢出分别报告，不改变原求解器、执行器或终止条件。
"""

from __future__ import annotations

import torch
import warp as wp


CONTACT_COLUMNS = ("active_count", "left_foot", "right_foot")
HEALTH_COLUMNS = ("overflow_bits", "max_nefc", "max_nacon", "max_ncollision", "hard_failure")


@wp.kernel(enable_backward=False)
def _reduce_contacts(
    nacon: wp.array(dtype=wp.int32),
    geom: wp.array(dtype=wp.vec2i),
    worldid: wp.array(dtype=wp.int32),
    distance: wp.array(dtype=wp.float32),
    margin: wp.array(dtype=wp.float32),
    foot_by_geom: wp.array2d(dtype=wp.int32),
    output: wp.array2d(dtype=wp.int32),
    invalid: wp.array(dtype=wp.int32),
):
    i = wp.tid()
    if i < nacon[0]:
        pair = geom[i]
        world = worldid[i]
        # 与 bridge 一致：负 geom（例如 flex）不是本机器人有效刚性接触。
        if distance[i] < margin[i] and pair[0] >= 0 and pair[1] >= 0:
            if (world < 0 or world >= output.shape[0] or
                    pair[0] >= foot_by_geom.shape[0] or pair[1] >= foot_by_geom.shape[0]):
                wp.atomic_add(invalid, 0, 1)
            else:
                wp.atomic_add(output, world, 0, 1)
                left = wp.max(foot_by_geom[pair[0], 0], foot_by_geom[pair[1], 0])
                right = wp.max(foot_by_geom[pair[0], 1], foot_by_geom[pair[1], 1])
                wp.atomic_max(output, world, 1, left)
                wp.atomic_max(output, world, 2, right)


@wp.kernel(enable_backward=False)
def _accumulate_solver_health(
    overflow: wp.array(dtype=wp.int32),
    nefc: wp.array(dtype=wp.int32),
    nacon: wp.array(dtype=wp.int32),
    ncollision: wp.array(dtype=wp.int32),
    output: wp.array2d(dtype=wp.int32),
):
    world = wp.tid()
    output[world, 0] = output[world, 0] | overflow[world]
    output[world, 1] = wp.max(output[world, 1], nefc[world])
    output[world, 2] = wp.max(output[world, 2], nacon[0])
    output[world, 3] = wp.max(output[world, 3], ncollision[0])


def reduce_contacts_tensor(nacon, geom, worldid, distance, margin, foot_by_geom, num_worlds):
    """可在 CPU 单测的同一个 Warp kernel；生产环境复用预分配缓冲，避免逐步分配。"""
    wp.init()
    device = str(geom.device)
    output = torch.zeros((num_worlds, 3), dtype=torch.int32, device=device)
    invalid = torch.zeros(1, dtype=torch.int32, device=device)
    wp.launch(_reduce_contacts, dim=len(geom), inputs=[
        wp.from_torch(nacon, dtype=wp.int32), wp.from_torch(geom, dtype=wp.vec2i),
        wp.from_torch(worldid, dtype=wp.int32), wp.from_torch(distance), wp.from_torch(margin),
        wp.from_torch(foot_by_geom, dtype=wp.int32), wp.from_torch(output, dtype=wp.int32),
        wp.from_torch(invalid, dtype=wp.int32)], device=device)
    return output, invalid


def health_failure_mask(health, invalid_contacts, *, contact_capacity, constraint_capacity, capacity_mask):
    """容量等号仍合法；只把超过容量、真实容量位和坏索引作为硬失败。"""
    return ((health[:, 0].bitwise_and(capacity_mask) != 0) |
            (health[:, 1] > constraint_capacity) | (health[:, 2] > contact_capacity) |
            (health[:, 3] > contact_capacity) | (invalid_contacts.reshape(()) != 0))


class BatchedStreamingPhysicalEnv:
    """每 GPU 单个批量 Newton 进程使用的组合适配器，不修改共享 SM 源码。

    reset(ids,state)、observe()、step_physics(action) 与原 bridge 一致。接触张量和
    health 张量都位于设备上；返回接触是独立快照，调用者可缓冲若干控制步后统一搬回 CPU。
    任何失败由采集器先保存轨迹再复用槽位，适配器从不隐式 reset。
    """

    def __init__(self, sim_cfg, num_envs, bundle):
        import mujoco
        from mujoco_warp import OverflowType
        from supermimic.bridge.streaming_env import StreamingPhysicalEnv

        if abs(float(sim_cfg.dt) - .02) > 1e-12 or int(sim_cfg.physics.num_substeps) != 4:
            raise ValueError("真实采集必须保持控制50Hz、Newton物理200Hz")
        self._env = StreamingPhysicalEnv(sim_cfg, num_envs, bundle)
        self._manager = self._env.sim.backend_manager
        self._solver = self._manager._solver
        if not hasattr(self._solver, "mjw_data"):
            raise RuntimeError("批量接触仅支持真实MuJoCo-Warp求解器")
        self._data = data = self._solver.mjw_data
        if int(data.nworld) != num_envs:
            raise ValueError("MuJoCo-Warp世界数量与环境槽位不一致")
        if self._manager._graph is not None:
            raise RuntimeError("求解器已捕获CUDA图，不能漏记子步诊断；请在首次step前创建适配器")
        for field in ("overflow", "nefc", "nacon", "ncollision"):
            if not hasattr(data, field):
                raise RuntimeError(f"当前MuJoCo-Warp缺少容量诊断字段 {field}")
        self.contact_capacity = int(data.naconmax)
        self.constraint_capacity = int(data.njmax)
        self.capacity_mask = int(OverflowType.ALL & ~(OverflowType.ITERATIONS | OverflowType.LS_ITERATIONS))
        self.iteration_mask = int(OverflowType.ITERATIONS | OverflowType.LS_ITERATIONS)
        model = self._solver.mj_model
        foot_by_geom = []
        for geom in range(model.ngeom):
            body = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, int(model.geom_bodyid[geom])) or "world"
            foot_by_geom.append((int("l_ankle_roll" in body), int("r_ankle_roll" in body)))
        self._foot_by_geom = wp.array(foot_by_geom, dtype=wp.int32, device=self.device)
        self._contacts = wp.zeros((num_envs, 3), dtype=wp.int32, device=self.device)
        self._invalid_contacts = wp.zeros(1, dtype=wp.int32, device=self.device)
        self._health = wp.zeros((num_envs, 4), dtype=wp.int32, device=self.device)
        # 首次物理图捕获前编译本模块。回调只启动固定形状Warp kernel，不做CPU同步。
        wp.load_module(module=__name__, device=self.device)
        self._manager.register_state_force_callback(self._before_substep)
        self._manager.register_post_step_callback(self._collect_health)

    def __getattr__(self, name):
        return getattr(self._env, name)

    def _collect_health(self):
        data = self._data
        wp.launch(_accumulate_solver_health, dim=self.N, inputs=[
            data.overflow, data.nefc, data.nacon, data.ncollision, self._health], device=self.device)

    def _before_substep(self, _state):
        # 第2至4子步前读取前一子步结果，post_step再读第4子步，避免只检查最后一个。
        self._collect_health()

    def reset(self, env_ids, initial_state):
        self._env.reset(env_ids, initial_state)
        # 全进程锁存峰值/溢出，不能因块内复用槽位而擦掉上一首的失败证据。
        wp.to_torch(self._data.overflow)[env_ids] = 0

    def observe(self):
        return self._env.observe()

    def step_physics(self, normalized_action):
        return self._env.step_physics(normalized_action)

    def contact_events_tensor(self):
        """返回[N,3] int32真实活动接触；全容量遍历不依赖CPU nacon.item()。"""
        self._contacts.zero_()
        # 坏world索引无法归属可靠槽位，累计到本进程结束，不能被下一步覆盖消失。
        contact = self._data.contact
        wp.launch(_reduce_contacts, dim=self.contact_capacity, inputs=[
            self._data.nacon, contact.geom, contact.worldid, contact.dist, contact.includemargin,
            self._foot_by_geom, self._contacts, self._invalid_contacts], device=self.device)
        return wp.to_torch(self._contacts).clone()

    def solver_health_tensor(self):
        """返回[N,5]，前四列为进程内峰值/位或，第五列表示不可发布的硬失败。"""
        values = wp.to_torch(self._health).clone()
        bad = health_failure_mask(values, wp.to_torch(self._invalid_contacts),
            contact_capacity=self.contact_capacity, constraint_capacity=self.constraint_capacity,
            capacity_mask=self.capacity_mask)
        return torch.cat((values, bad[:, None].to(torch.int32)), dim=1)

    def assert_solver_healthy(self):
        """块边界的小型CPU读回；容量溢出不得仅记录警告后继续发布正常缓存。"""
        health = self.solver_health_tensor().cpu()
        bad = health[:, -1].nonzero().flatten().tolist()
        if bad:
            raise RuntimeError(f"Newton求解器容量/接触索引失败，槽位={bad[:32]}，"
                f"naconmax={self.contact_capacity},njmax={self.constraint_capacity}")
        return {"contact_capacity_all_worlds": self.contact_capacity,
            "constraint_capacity_per_world": self.constraint_capacity,
            "maximum_contacts_all_worlds": int(health[:, 2].max()),
            "maximum_constraints_per_world": int(health[:, 1].max()),
            "maximum_broadphase_all_worlds": int(health[:, 3].max()),
            "iteration_limit_worlds": int((health[:, 0].bitwise_and(self.iteration_mask) != 0).sum()),
            "capacity_overflow_worlds": 0}

    def contact_contract(self):
        return {"source": "mujoco_warp_solver_active_contacts", "encoding": "world_reduced_int32_v1",
            "columns": list(CONTACT_COLUMNS), "world_identity": "contact.worldid",
            "predicate": "dist<includemargin and both geom>=0", "geometry_to_foot": "geom_bodyid/body_name",
            "foot_by_geom": wp.to_torch(self._foot_by_geom).cpu().tolist(),
            "health_columns": list(HEALTH_COLUMNS), "health_all_200hz_substeps": True}
