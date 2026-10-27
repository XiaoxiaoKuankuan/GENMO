"""音乐动作保持的物理空间稳健损失，避免直接对 qpos30 混合编码做距离。

执行损失按根部位置/姿态/速度、根部坐标系身体位置、身体姿态和速度、关节角/速度
分别以物理尺度归一化，使用 pseudo-Huber 保持大误差下的非饱和导数。
局部教师先与学生共同解码为世界 qpos28，再通过同一 SM FK 比较；只对可修改帧
计入位置和姿态损失，时间差分保留跨已知前缀边界的导数，已知坐标绝不成为优化变量。
固定音乐教师和同请求局部教师承担不同职责，不能用后者替代固定音乐时间线。
"""

from dataclasses import dataclass
import torch
from supermimic.mdp import unflatten_state
from supermimic.rotations import quat_to_mat, so3_log, yaw_mat, mat_to_quat


DEFAULT_SCALES = {"root_position": .3, "root_orientation": .4, "root_velocity": 1.,
    "body_position": .15, "body_orientation": .3, "body_velocity": 1.,
    "joint_position": .35, "joint_velocity": 2., "joint_acceleration": 20.}


def robust(value, scale):
    return torch.sqrt(1+(value/scale).square())-1


@dataclass(frozen=True)
class AnchorAlignment:
    """执行用真实状态、参考保持用已记录参考的anchor；整段H共享固定变换。"""
    rotation: torch.Tensor
    translation: torch.Tensor

    @classmethod
    def at_anchor(cls, state, target):
        p, R, *_ = unflatten_state(state.detach())
        return cls.from_pose(p,R,target)

    @classmethod
    def at_reference_anchor(cls, reference, target):
        # 使用候选更新前已经记录的参考，不能使用候选自身可变根部重新拟合变换。
        return cls.from_pose(reference.root_p.detach(),reference.root_R.detach(),target)

    @classmethod
    def from_pose(cls,p,R,target):
        rotation = (yaw_mat(R) @ yaw_mat(target.root_R.detach()).transpose(-1, -2)).detach()
        translated = p - (rotation @ target.root_p.detach()[..., None]).squeeze(-1)
        translation = torch.cat((translated[..., :2], torch.zeros_like(translated[..., 2:])), -1).detach()
        return cls(rotation, translation)

    def is_identity(self):
        """严格逐元素恒等检测；不使用容差，不近似非零初始坐标变换。"""
        cached=getattr(self,"_identity_exact",None)
        if cached is None:
            identity=torch.eye(3,device=self.rotation.device,dtype=self.rotation.dtype).expand_as(self.rotation)
            cached=torch.equal(self.rotation,identity) and torch.equal(self.translation,torch.zeros_like(self.translation))
            object.__setattr__(self,"_identity_exact",cached)
        return cached

    def reference(self, ref):
        r, t = self.rotation, self.translation
        return ref._replace(root_p=(r@ref.root_p.detach()[..., None]).squeeze(-1)+t,
            root_R=r@ref.root_R.detach(), root_v=(r@ref.root_v.detach()[..., None]).squeeze(-1),
            root_w=(r@ref.root_w.detach()[..., None]).squeeze(-1), q=ref.q.detach(), qd=ref.qd.detach())

    def poses(self, qpos):
        qpos = qpos.detach()
        r, t = self.rotation[:, None], self.translation[:, None]
        return torch.cat(((r@qpos[..., :3, None]).squeeze(-1)+t,
                          mat_to_quat(r@quat_to_mat(qpos[..., 3:7])), qpos[..., 7:]), -1)


def execution_loss(state, reference, kin, scales=None):
    scales = dict(DEFAULT_SCALES, **(scales or {}))
    p, R, v, w, q, qd = unflatten_state(state)
    ref = reference
    bp, bR, bv, _ = kin.forward(p, R, v, w, q, qd, all_bodies=True)
    tp, tR, tv, _ = kin.forward(ref.root_p, ref.root_R, ref.root_v, ref.root_w,
                                ref.q, ref.qd, all_bodies=True)
    local = (R.transpose(-1, -2)[:, None]@(bp-p[:, None])[..., None]).squeeze(-1)
    target_local = (ref.root_R.transpose(-1, -2)[:, None]@(tp-ref.root_p[:, None])[..., None]).squeeze(-1)
    errors = {"root_position": p-ref.root_p, "root_orientation": so3_log(ref.root_R.transpose(-1, -2)@R),
        "root_velocity": v-ref.root_v, "body_position": local-target_local,
        "body_orientation": so3_log(tR.transpose(-1, -2)@bR), "body_velocity": bv-tv,
        "joint_position": q-ref.q, "joint_velocity": qd-ref.qd}
    groups = {k: robust(x, scales[k]).flatten(1).mean(-1) for k,x in errors.items()}
    return torch.stack(list(groups.values())).mean(0), groups


def reference_prior(candidate, teacher, adapter, *, scales=None, global_reference=None, mutable_mask=None):
    scales = dict(DEFAULT_SCALES, **(scales or {}))
    a, b = candidate.world_qpos, teacher.world_qpos.detach()
    prefix = candidate.ticket.prefix_frames
    if a.shape != b.shape or prefix < 1 or prefix >= a.shape[1]:
        raise ValueError("动作保持需要相同60帧物理轨迹和合法保护前缀")
    free = torch.arange(a.shape[1], device=a.device) >= prefix
    if mutable_mask is not None:
        free = free & mutable_mask
    if not bool(free.any()):
        raise ValueError("参考保持没有可修改且有真实目标支持的源点")
    a = torch.cat((a[:, :prefix].detach(), a[:, prefix:]), 1)
    if mutable_mask is not None:
        a = torch.where(free[None,:,None],a,a.detach())
    def state_and_bodies(x):
        R = quat_to_mat(x[..., 3:7]); q = x[..., 7:][..., adapter.to_sm]
        zeros = torch.zeros_like(x[..., :3])
        bodies, orientations, _, _ = adapter.runtime.kin.forward(x[..., :3], R, zeros, zeros, q,
                                             torch.zeros_like(q), all_bodies=True)
        local = (R.transpose(-1,-2)[:, :, None]@(bodies-x[..., :3][:, :, None])[..., None]).squeeze(-1)
        return R, q, local, R.transpose(-1,-2)[:, :, None]@orientations
    Ra, qa, ba, bRa = state_and_bodies(a)
    Rb, qb, bb, bRb = state_and_bodies(b)
    errors = {"root_position": a[..., :3]-b[..., :3],
        "root_orientation": so3_log(Rb.transpose(-1,-2)@Ra), "joint_position": qa-qb,
        "body_position": ba-bb, "body_orientation": so3_log(bRb.transpose(-1,-2)@bRa)}
    groups = {k: robust(v[:, free], scales[k]).mean() for k,v in errors.items()}
    velocity_mask=free[1:] if mutable_mask is None else free[1:] & free[:-1]
    acceleration_mask=free[2:] if mutable_mask is None else free[2:] & free[1:-1] & free[:-2]
    def temporal(value,mask,scale):
        return robust(value[:,mask],scale).mean() if bool(mask.any()) else value.sum()*0.
    groups["root_velocity"] = temporal(torch.diff(a[..., :3]-b[..., :3], dim=1)*30,velocity_mask,scales["root_velocity"])
    groups["joint_velocity"] = temporal(torch.diff(qa-qb, dim=1)*30,velocity_mask,scales["joint_velocity"])
    groups["body_velocity"] = temporal(torch.diff(ba-bb, dim=1)*30,velocity_mask,scales["body_velocity"])
    groups["joint_acceleration"] = temporal(torch.diff(qa-qb, n=2, dim=1)*900,acceleration_mask,scales["joint_acceleration"])
    # 独立时间线保留整曲原动作；这里只报告偏差，执行主损失直接使用该目标。
    if global_reference is not None:
        ticks = candidate.ticket.request_tick+torch.arange(60, device=a.device)*20
        fixed = global_reference.source_at(ticks).detach()
        groups["fixed_timeline_joint_deviation_rms"] = (a[:, free, 7:]-fixed[:, free, 7:]).square().mean().sqrt()
    loss = torch.stack([v for k,v in groups.items() if not k.endswith("deviation_rms")]).mean()
    return loss, groups


def consumed_music_support(sample,horizon):
    """本请求真正成为当前消费参考的支持点；后来其它请求覆盖的未来舞蹈不归本计划。"""
    points=[];version=sample.generation.reference_version
    for step in sample.transitions[:horizon]:
        cache=step.start.reference
        support=cache.dependency_support(step.end.tick,future_steps=0)
        points.append(support[cache.source_versions[(support-cache.lower)//20]==version])
    return torch.unique(torch.cat(points))


def consumed_music_ticks(sample,horizon):
    """只选择本计划影响实际当前参考的50Hz时刻，不把尚未接管的未来目标当当前动作。"""
    ticks=[];version=sample.generation.reference_version
    for step in sample.transitions[:horizon]:
        cache=step.start.reference;support=cache.dependency_support(step.end.tick,future_steps=0)
        if bool((cache.source_versions[(support-cache.lower)//20]==version).any()):ticks.append(step.end.tick)
    return torch.tensor(ticks,device=sample.state.state.device,dtype=torch.long)


def online_reference_objective(candidate,fixed,adapter,anchor_state,anchor_tick,control_ticks,support_ticks,scales,
                               reference_records,reference_version):
    """候选与固定目标都在原50Hz控制时刻比较，速度沿用SM相同参考构造。

    整个60帧候选只允许相关可修改源点传导梯度。教师直接使用保存的原始消费速度，
    不将50Hz参考反插值到30Hz来制造虚假的速度/加速度保持项。
    """
    from .imagination import replace_plan_version
    from supermimic.rotations import mat_to_quat
    from supermimic.motion import ReferenceBatch
    clock=candidate.ticket.request_tick+torch.arange(60,device=candidate.world_qpos.device)*20
    free=(torch.arange(60,device=clock.device)>=candidate.ticket.prefix_frames)&torch.isin(clock,support_ticks)
    ticks=control_ticks[(control_ticks>=fixed.lower)&(control_ticks<=fixed.upper)]
    if not bool(free.any()) or not len(ticks):raise ValueError("本anchor没有可修改的当前音乐消费支持")
    if reference_records is None or reference_version is None:
        raise ValueError("在线音乐保持必须保留每个实际控制时刻的混合参考版本")
    allowed=set(ticks.tolist());generated=[]
    for step in reference_records:
        if step.end.tick not in allowed:continue
        # 后来的参考覆盖、旧保护源点必须固定；detach当前候选值不能代替这些真实常量。
        cache=replace_plan_version(step.start.reference,candidate,reference_version)
        generated.append(cache.read_current_future(step.end.tick,0)[0])
    if len(generated)!=len(ticks):raise ValueError("固定音乐目标与实际消费记录时刻不一一对应")
    target=fixed.reference_at_ticks(ticks)
    generated_flat=ReferenceBatch(*(torch.cat(values) for values in zip(*generated)))
    target_flat=ReferenceBatch(*(v.detach().flatten(0,1) for v in target))
    recorded_anchor=reference_records[0].start.reference.read_current_future(anchor_tick,0)[0]
    alignment=AnchorAlignment.at_reference_anchor(recorded_anchor,fixed.read_current_future(anchor_tick,0)[0])
    target_flat=alignment.reference(target_flat)
    state=torch.cat((generated_flat.root_p,mat_to_quat(generated_flat.root_R),generated_flat.root_v,
        generated_flat.root_w,generated_flat.q,generated_flat.qd),-1)
    _,parts=execution_loss(state,target_flat,adapter.runtime.kin,scales)
    parts={k:v.mean() for k,v in parts.items()}
    for group,indices in (("upper",[i for i,n in enumerate(adapter.bundle.spec.joint_names) if "arm" in n or "elbow" in n]),
                          ("lower",[i for i,n in enumerate(adapter.bundle.spec.joint_names) if any(x in n for x in ("leg","knee","ankle"))])):
        x,y=generated_flat.q[:,indices],target_flat.q[:,indices]
        parts[group+"_amplitude"]=robust((x.amax(0)-x.amin(0))-(y.amax(0)-y.amin(0)),.35).mean()
        parts[group+"_velocity"]=robust(generated_flat.qd[:,indices]-target_flat.qd[:,indices],2.).mean()
    return torch.stack(list(parts.values())).mean(),parts


def reference_objectives(candidate, local, fixed, adapter, *, anchor_state, anchor_tick, scales=None,global_support_ticks=None,global_control_ticks=None,
                         global_reference_records=None,global_reference_version=None):
    """局部与固定音乐约束分离；固定目标只在真实anchor处对齐一次，已知前缀detach。"""
    from dataclasses import replace
    local_loss, local_parts = reference_prior(candidate, local, adapter, scales=scales)
    if hasattr(fixed,"reference_at_ticks"):
        if global_control_ticks is None or global_support_ticks is None:raise ValueError("在线音乐保持需要实际消费时刻及源支持")
        global_loss,global_parts=online_reference_objective(candidate,fixed,adapter,anchor_state,anchor_tick,
            global_control_ticks,global_support_ticks,scales,global_reference_records,global_reference_version)
        return local_loss,global_loss,{"local":local_parts,"global":global_parts}
    clock = candidate.ticket.request_tick + torch.arange(60, device=candidate.world_qpos.device)*20
    valid = (clock >= fixed.lower) & (clock <= fixed.upper) & (clock >= anchor_tick)
    if global_support_ticks is not None:
        valid=valid & torch.isin(clock,global_support_ticks)
    # 越过音乐真实目标末端的点只用于占位，valid遮罩阻止其进入任何损失。
    target = fixed.source_at(clock.clamp(fixed.lower, fixed.upper)).detach()
    align = AnchorAlignment.at_anchor(anchor_state, fixed.read_current_future(anchor_tick, future_steps=0)[0])
    target = align.poses(target)
    global_plan = replace(candidate, world_qpos=target, contact=candidate.contact.detach())
    global_loss, global_parts = reference_prior(candidate, global_plan, adapter, scales=scales, mutable_mask=valid)
    free = valid & (torch.arange(60, device=clock.device) >= candidate.ticket.prefix_frames)
    a, b = candidate.world_qpos[:, free, 7:], target[:, free, 7:]
    # GENMO原顺序按SM映射索引分组，不能假设前12维恰为腿。
    a, b = a[..., adapter.to_sm], b[..., adapter.to_sm]
    for group, selected in (("upper", [i for i,n in enumerate(adapter.bundle.spec.joint_names) if "arm" in n or "elbow" in n]),
                            ("lower", [i for i,n in enumerate(adapter.bundle.spec.joint_names) if "leg" in n or "knee" in n or "ankle" in n])):
        if not selected or a.shape[1] < 2: continue
        x, y = a[..., selected], b[..., selected]
        amp_a, amp_b = x.amax(1)-x.amin(1), y.amax(1)-y.amin(1)
        global_parts[group+"_amplitude"] = robust(amp_a-amp_b, .35).mean()
        spacing=torch.diff(clock[free]).to(x)/600.
        da, db = torch.diff(x,dim=1)/spacing[None,:,None], torch.diff(y,dim=1)/spacing[None,:,None]
        global_parts[group+"_velocity"] = robust(da-db, 2.).mean()
    global_loss = torch.stack(list(global_parts.values())).mean()
    return local_loss, global_loss, {"local":local_parts, "global":global_parts}
