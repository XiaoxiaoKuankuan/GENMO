"""从完整真实在线逐控制轨迹重算新音乐目标的合法MC监督。

原始轨迹必须有每个真实控制步的状态、动作、当时参考及实际终止；稀疏旧价值快照
不能输入本模块。每个价值样本在自身真实时刻固定一次SE(2)对齐，随后直到真实终止
使用同一目标变换，和短时Actor的目标上下文一致。该变换显式进入Critic状态，绝不
送入GENMO历史。MC目标只包含执行奖励，BC与参考正则不冒充物理奖励。pi/wm和Actor
均不训练，旧Critic不导入；重拟合与独立验证由现有拟合入口执行。
"""
from dataclasses import replace
import json
from pathlib import Path
import torch
from supermimic.mdp import unflatten_state
from supermimic.motion import ReferenceBatch
from .dance_losses import execution_loss,AnchorAlignment
from .checkpoint import pack_state
from .music_repair_audit import online_episode


def critic_reward_contract(cfg):
    from .short_audit import teacher_target_contract
    target=teacher_target_contract(cfg)
    return {"teacher":{k:target[k] for k in ("mode","online_seed","manifest_sha256")},
        "rewards":cfg["rewards"],"alignment_context":"fixed_at_value_state_until_real_termination.v1"}


def reference_from_states(states,kin):
    p,R,v,w,q,qd=unflatten_state(states)
    bp,bR,bv,bw=kin.forward(p,R,v,w,q,qd)
    Rt=R.transpose(-1,-2)
    return ReferenceBatch(p,R,v,w,q,qd,bp,bR,bv,bw,(Rt@v[...,None]).squeeze(-1),
        (Rt@w[...,None]).squeeze(-1),R[...,2,2])


@torch.no_grad()
def aligned_real_mc(controls,adapter,reward_cfg,gamma,*,phase=0,consumed_reference_states=None):
    """完整真实episode的固定目标上下文MC；原Stage1或学生轨迹均可逐步重标。

    当前参考从每个真实控制步当时的缓存读取，固定教师从独立冻结时间线读取，不能
    在学生数据上把两者混为一条参考。返回价值状态带固定SE(2)上下文的稀疏样本，
    所有中间真实奖励完整累计，不将行政截断补成终点。
    """
    if not controls or not controls[-1].terminal:raise ValueError("MC需要完整真实终止")
    states=torch.cat([x.end.state for x in controls]);kin=adapter.runtime.kin;device=states.device
    ticks=torch.tensor([x.end.tick for x in controls],device=device,dtype=torch.long)
    teacher=controls[0].start.teacher_reference
    if teacher is None or not hasattr(teacher,"reference_at_ticks"):raise ValueError("MC缺少固定真实在线音乐目标")
    target=ReferenceBatch(*(v.flatten(0,1) for v in teacher.reference_at_ticks(ticks)))
    if consumed_reference_states is None:
        refs=[x.start.reference.read_current_future(x.end.tick,0)[0] for x in controls]
        refs=ReferenceBatch(*(torch.cat(values) for values in zip(*refs)))
    else:
        refs=reference_from_states(consumed_reference_states.to(states),kin)
    track,_=execution_loss(states,refs,kin,reward_cfg.get("dance_scales"))
    q=states[:,13:34];actions=torch.cat([x.end.previous_action for x in controls]);previous=torch.cat([x.start.previous_action for x in controls])
    rate=(actions-previous).square().sum(-1)
    joint=(torch.relu(q-adapter.runtime.mdp.soft_hi)+torch.relu(adapter.runtime.mdp.soft_lo-q)).sum(-1)
    physical=reward_cfg.get("action_rate_weight",0.)*rate+reward_cfg.get("joint_limit_weight",2.)*joint
    pairs=[]
    for i in range(phase,len(controls),5):
        z=controls[i].start;alignment=AnchorAlignment.at_anchor(z.state,teacher.read_current_future(z.tick,0)[0])
        selected=ReferenceBatch(*(x[i:] for x in target))
        dance,_=execution_loss(states[i:],alignment.reference(selected),kin,reward_cfg.get("dance_scales"))
        rewards=-(dance+reward_cfg.get("track_weight",.1)*track[i:]+physical[i:])*.02
        discount=gamma**torch.arange(len(rewards),device=device,dtype=rewards.dtype)
        z=replace(z,teacher_alignment_rotation=alignment.rotation,teacher_alignment_translation=alignment.translation)
        pairs.append((z.detached(),(discount*rewards).sum().reshape(1).detach()))
    return pairs


@torch.no_grad()
def relabel_online_mc(cfg,actor,engine,tasks,output,device):
    output=Path(output);(output/"mc_episodes").mkdir(parents=True,exist_ok=False)
    episodes=[];steps=0;split=cfg["mc_collection"]["split"]
    for episode,task in enumerate(tasks):
        payload,initial,controls,_,target=online_episode(cfg,engine,task,device,allow_partial_for_real_mc=True)
        if not controls[-1].terminal:raise ValueError("逐步轨迹未真实终止，不能伪造完整MC")
        pairs=aligned_real_mc(controls,engine.adapter,cfg["rewards"],cfg["critic"]["gamma_per_control"],
            phase=episode%5,consumed_reference_states=payload["reference_state"][0,1:])
        info={"task":task["id"],"seed":cfg["teacher"].get("online_seed",42),"reason":payload["reason"],
            "music_completed":payload["complete"],"target_support_end_tick":target.upper,
            "failed_episode_is_real_terminal_not_completed_music":not payload["complete"],
            "steps":len(controls),"mc_return":float(pairs[0][1]),"commits":sum(g["accepted"] for g in payload["events"]),
            "mc_return_start_tick":pairs[0][0].tick,
            "pending_value_samples":sum(z.pending for z,_ in pairs),"non_request_grid_samples":sum(z.tick%60!=0 for z,_ in pairs),
            "source_online_identity":payload["identity"],"relabel_scope":"all_real_control_steps_no_sparse_reward_interpolation"}
        memo={};torch.save({"episode":info,"pairs":[(pack_state(z,memo),y.cpu()) for z,y in pairs],
            "evidence":"real_complete_episode_mc","reward_contract":critic_reward_contract(cfg)},
            output/"mc_episodes"/f"{split}_{episode:03d}.pt")
        episodes.append(info);steps+=len(controls)
        print(json.dumps({"event":"relabel_real_mc","task":task["id"],"steps":len(controls),"pairs":len(pairs)}),flush=True)
    torch.save({"actor":actor.state_dict(),"episodes":episodes,"physical_control_steps":steps,"split":split,
        "contract":{k:cfg[k] for k in ("model","clock","rewards","critic")},"teacher_identity":engine.teacher.identity,
        "critic_reward_contract":critic_reward_contract(cfg),"frozen_sha256":engine.adapter.bundle.metadata["sha256"],
        "physics_determinism":cfg["execution"].get("physics_determinism")},output/"mc_complete.pt")
    (output/"manifest.json").write_text(json.dumps({"config":cfg,"source":"complete_original_stage1_real_online_trajectories"},ensure_ascii=False,indent=2))
