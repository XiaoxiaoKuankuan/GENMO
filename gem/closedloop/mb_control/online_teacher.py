"""冻结Stage1真实在线参考时间线的持久化、精确读取与独立构建。

输入是原Stage1、同一冻结SM、真实Newton逐控制步轨迹及异步请求记录。构建阶段保存
当时实际消费的ReferenceBatch基础字段、真实执行state55、参考版本、噪声与接管边界；
训练仅加载已冻结文件，不使用学生轨迹改写。50Hz参考速度直接来自当时控制参考，
不能把30Hz二次插值差分冒充原速度。任意源时刻仅用显式插值供参考保持，目标末端
没有支持时必须遮罩或拒绝；不补造未来。缓存身份绑定数据、模型、物理和事件，源码
提交仅记录。该模块不进入部署GENMO生成线程。
"""

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import torch
from supermimic.rotations import quat_to_mat, mat_to_quat
from supermimic.bridge.reference import slerp, current_future
from supermimic.motion import ReferenceBatch
from .contracts import ReferenceExhausted


def identity_digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(",", ":")).encode()).hexdigest()


def recorded_delivery_events(payload):
    """实际完成与控制线程收到结果分别保存；末尾未接管请求属于删失事件，不猜延迟。"""
    events={str(e["request_tick"]):{"actual_ready_tick":e["actual_ready_tick"],"commit_tick":e["commit_tick"],
        "delivery_tick":e["commit_tick"] if e["commit_tick"] is not None else e["rejection_tick"],
        "delivered":True} for e in payload["events"]}
    if payload.get("controls",{}).get("data"):
        pending=payload["controls"]["data"][-1]["data"]["start"]["data"].get("pending_request")
        if pending is not None:
            tick=pending["data"]["ticket"]["data"]["request_tick"]
            if str(tick) not in events:
                events[str(tick)]={"delivered":False,"censored_at_tick":int(payload["ticks"][-1])}
    return events


def same_frozen_timeline_identity(a,b):
    """派生回放字段不改变目标；核对原始数据身份而非新增日志字段或源码。"""
    keys=("base","delay_events_sha256","noise_identities","consumption_versions_sha256","timing_mode")
    return all(a.get(k)==b.get(k) for k in keys)


@dataclass(frozen=True)
class OnlineTeacherTimeline:
    ticks: torch.Tensor
    reference_state: torch.Tensor  # [1,N,55]，SM关节序，含原始参考速度。
    source_versions: torch.Tensor
    identity: dict
    kin: object
    to_sm: tuple

    def __post_init__(self):
        if self.reference_state.shape != (1,len(self.ticks),55) or not bool(((self.ticks[1:]-self.ticks[:-1])==12).all()):
            raise ValueError("在线教师必须保存连续50Hz原始参考state55")
        if self.reference_state.requires_grad or not torch.isfinite(self.reference_state).all():
            raise ValueError("在线教师必须有限且完全冻结")
        object.__setattr__(self,"_lower",int(self.ticks[0]));object.__setattr__(self,"_upper",int(self.ticks[-1]))

    @property
    def lower(self): return self._lower

    @property
    def upper(self): return self._upper

    def source_at(self, ticks):
        if bool(((ticks<self.lower)|(ticks>self.upper)).any()):
            raise ReferenceExhausted("超出冻结在线教师真实目标支持")
        offset=ticks-self.lower;lo=(offset//12).long();hi=(lo+1).clamp_max(len(self.ticks)-1)
        a=(offset%12).to(self.reference_state)/12
        left,right=self.reference_state[:,lo],self.reference_state[:,hi]
        inv=torch.argsort(torch.tensor(self.to_sm,device=ticks.device))
        return torch.cat(((1-a[None,:,None])*left[...,:3]+a[None,:,None]*right[...,:3],
            slerp(left[...,3:7],right[...,3:7],a[None]),
            ((1-a[None,:,None])*left[...,13:34]+a[None,:,None]*right[...,13:34])[...,inv]),-1).detach()

    def read_current_future(self, control_tick, future_steps=10):
        end=control_tick+12*future_steps
        if control_tick%12 or control_tick<self.lower or end>self.upper:
            raise ReferenceExhausted("在线教师50Hz目标支持不足")
        return current_future(self.reference_at_ticks(torch.arange(control_tick,end+1,12,device=self.ticks.device)))

    def reference_at_ticks(self,ticks):
        """批量读取原始50Hz参考，保留原速度，避免50→30→50重采样制造保持误差。"""
        if bool(((ticks%12!=0)|(ticks<self.lower)|(ticks>self.upper)).any()):
            raise ReferenceExhausted("在线教师只对已有50Hz实际消费时间戳提供原始参考")
        s=self.reference_state[:,((ticks-self.lower)//12).long()]
        p,R,v,w,q,qd=s[...,:3],quat_to_mat(s[...,3:7]),s[...,7:10],s[...,10:13],s[...,13:34],s[...,34:55]
        bp,bR,bv,bw=self.kin.forward(p,R,v,w,q,qd)
        Rt=R.transpose(-1,-2)
        return ReferenceBatch(p,R,v,w,q,qd,bp,bR,bv,bw,
            (Rt@v[...,None]).squeeze(-1),(Rt@w[...,None]).squeeze(-1),R[...,2,2])

    def detached(self): return self


def online_base_identity(teacher, music, initial):
    from .teacher import tensor_digest
    return {"schema":"stage1_online_consumed_target.v1", "teacher":teacher.identity,
        "music_initial_sha256":tensor_digest(music,initial),
        "request_noise_seed":teacher.config.get("online_seed",42),
        "request_noise_rule":"seed_plus_request_tick_grid_times_1009.v1",
        "robot":teacher.adapter.bundle.metadata["robot"],
        "sm_sha256":teacher.adapter.bundle.metadata["sha256"],
        "clock":dict(teacher.scheduler.__dict__),
        "physics_backend":"isaaclab3_newton_mujoco_warp",
        "physics_determinism":teacher.physics_determinism,
        "target_definition":"actually_consumed_reference_with_original_velocity.v1"}


def load_online_timeline(teacher,music,initial,*,allow_partial_for_real_mc=False):
    from supermimic.bridge.bundle import file_digest
    base=online_base_identity(teacher,music,initial);key=identity_digest(base)
    index_path=Path(teacher.config["online_manifest"])
    index=json.loads(index_path.read_text())
    item=index["timelines"].get(key)
    if item is None:raise ValueError("当前音乐/初态/资产/噪声/调度没有已核对的在线教师时间线")
    path=Path(item["path"])
    if file_digest(path)!=item["sha256"]:raise ValueError("冻结在线教师缓存内容改变")
    payload=torch.load(path,map_location="cpu",weights_only=False)
    if payload["identity"]["base"]!=base or identity_digest(payload["identity"])!=item["identity_sha256"]:
        raise ValueError("在线教师事件/噪声/物理身份不匹配")
    if not payload["complete"] and not allow_partial_for_real_mc:
        raise ValueError("提前失败的在线教师不能伪装完整音乐目标")
    identity=dict(payload["identity"])
    identity.setdefault("delay_events",{str(e["request_tick"]):e["actual_ready_tick"]-e["request_tick"] for e in payload["events"]})
    identity["delivery_events"]=recorded_delivery_events(payload)
    return OnlineTeacherTimeline(payload["ticks"].to(initial.device),payload["reference_state"].to(initial),
        payload["source_versions"].to(initial.device),identity,teacher.adapter.runtime.kin,teacher.adapter.to_sm)


def build_online_timelines(collector,teacher,tasks,cfg,output,device):
    """独立准备全曲真实在线标签；原独立运动学时间线单独保存作为A/B研究对照。"""
    from .application import task_inputs,seed_all
    from .checkpoint import pack_value
    from .teacher import tensor_digest
    from supermimic.bridge.bundle import atomic_save,file_digest
    output=Path(output);index={"schema":"stage1_online_teacher_index.v1","timelines":{}}
    collector.record_controls=True;collector.record_contacts=True
    collector.noise_seed=teacher.config.get("online_seed",42)
    previous_teacher=getattr(collector.rewards,"teacher",None);previous_mode=collector.rewards.mode
    if any(not torch.equal(v,teacher.stage1.state_dict()[n]) for n,v in collector.actor.stage1.state_dict().items()):
        raise ValueError("在线教师构建只允许原Stage1参数")
    if any(bool(v.detach().count_nonzero()) for v in collector.actor.future_encoder.output.parameters()):
        raise ValueError("原Stage1在线教师构建要求未来条件输出严格为零，不能混入已训练FutureEncoder")
    collector.rewards.teacher=None;collector.rewards.mode="upper_eight"
    try:
        for task in tasks:
            seed_all(collector.noise_seed)
            music,initial=task_inputs(task,collector.adapter,device)
            collector.reset(music,initial)
            controls=[]
            while not collector.terminal:
                last=collector.collect(1);controls.extend(collector.take_controls())
            if not controls:raise RuntimeError("在线教师没有真实控制步")
            from .returns import close_real_mc_episode
            close_real_mc_episode(controls,last[-1])
            # 使用已部署的拼批CFG/编译加速路径，但逐请求检查原冻结Stage1的同噪声输出。
            # 此检查在episode结束后运行，不污染该episode的真实生成延迟。
            parity=[]
            for g in list(collector.generations.values())[::max(1,len(collector.generations)//3)]:
                original=teacher.generate(g.request.snapshot,g.request.ticket,g.request.noise)
                error=float((g.plan.world_qpos-original.world_qpos).abs().max());parity.append(error)
                if error>1e-5:raise ValueError("在线构建Actor输出不是原Stage1同噪声输出")
            refs=[];versions=[];consumed_support=[]
            pairs=[(controls[0].start.tick,controls[0].start.reference)]
            pairs.extend((x.end.tick,x.start.reference) for x in controls)
            for tick,cache in pairs:
                ref=cache.read_current_future(tick,future_steps=0)[0]
                refs.append(torch.cat((ref.root_p,mat_to_quat(ref.root_R),ref.root_v,ref.root_w,ref.q,ref.qd),-1).cpu())
                versions.append(cache.version)
                support=cache.dependency_support(tick,future_steps=0)
                consumed_support.append({"tick":tick,"source_ticks":support.tolist(),
                    "source_versions":cache.source_versions[(support-cache.lower)//20].tolist()})
            base=online_base_identity(teacher,music,initial)
            events=[{k:v for k,v in e.items() if k!="timings"} for e in collector.events]
            noises=[{"tick":g.request.ticket.request_tick,"sha256":tensor_digest(g.request.noise)} for g in collector.generations.values()]
            identity={"base":base,"delay_events_sha256":identity_digest(events),"noise_identities":noises,
                      "delay_events":{str(e["request_tick"]):e["actual_ready_tick"]-e["request_tick"] for e in collector.events},
                      "consumption_versions_sha256":identity_digest(versions),"timing_mode":collector.timing_mode}
            identity["consumed_support_versions_sha256"]=identity_digest(consumed_support)
            key=identity_digest(base);path=output/(key+".pt")
            research=teacher.independent_timeline(music,initial)
            ticks=torch.tensor([x[0] for x in pairs],dtype=torch.long)
            atomic_save({"identity":identity,"complete":last[-1].reason=="music_end","reason":last[-1].reason,
                "ticks":ticks,"reference_state":torch.stack(refs,1),"source_versions":torch.tensor(versions),
                "source_versions_meaning":"current_cache_version_not_all_contributing_support_versions",
                "consumed_reference_support":consumed_support,
                "actual_state":torch.stack([controls[0].start.state.cpu()]+[x.end.state.cpu() for x in controls],1),
                "events":collector.events,"controls":pack_value(tuple(controls)),
                "cancelled_requests":collector.cancelled_requests,
                "terminal_event":pack_value(last[-1]),
                "generations":pack_value(tuple(collector.generations.values())),"initial":initial.cpu(),"task":task,
                "research_independent_world_qpos":research.source_at(ticks.to(initial.device)).cpu(),
                "research_seed":teacher.config.get("seed"),"online_noise_seed":collector.noise_seed,
                "research_noise_identity_matches_online":False,"original_stage1_parity_max_abs":max(parity,default=0.)},path)
            index["timelines"][key]={"path":str(path),"sha256":file_digest(path),"identity_sha256":identity_digest(identity),
                "task":task["id"],"complete":last[-1].reason=="music_end"}
            (output/"online_manifest.json").write_text(json.dumps(index,ensure_ascii=False,indent=2))
            print(json.dumps({"event":"online_teacher_completed","task":task["id"],"steps":len(controls),"complete":last[-1].reason=="music_end"}),flush=True)
    finally:
        collector.rewards.teacher=previous_teacher;collector.rewards.mode=previous_mode
    return index
