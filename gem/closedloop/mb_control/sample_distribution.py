"""真实短轨迹的训练覆盖统计与分层抽样，不用WM预测替代实际运动标签。

速度来自真实state55的关节速度，幅度来自同一anchor真实H步的关节范围，接触来自
Newton求解器的已发生接触事件。每个候选都记录来源、动作类别、合法/超域状态。
分层抽样只改变同Actor版本已有合法候选的选择，不扩大WM支持域、不伪造困难动作。
"""

from collections import Counter
import torch


def describe_request(z,adapter,collector,episode):
    """A组在请求发出时立即统计，不从后来成功的anchor倒推全部请求。"""
    qd=z.state[:,34:55].detach();speed=float(qd.square().mean().sqrt())
    leg=[i for i,n in enumerate(adapter.bundle.spec.joint_names) if any(k in n for k in ("leg","knee","ankle"))]
    arm=[i for i,n in enumerate(adapter.bundle.spec.joint_names) if "arm" in n or "elbow" in n]
    leg_speed=float(qd[:,leg].square().mean().sqrt());arm_speed=float(qd[:,arm].square().mean().sqrt())
    category="low_motion" if speed<.2 else "active_upper_body" if arm_speed>.7 and arm_speed>1.5*leg_speed else "whole_body_motion"
    if float(z.state[:,12].abs().mean())>.2:category="turning"
    history=z.genmo_history[:,z.genmo_valid[0],6:27]
    amplitude=float((history.amax(1)-history.amin(1)).mean()) if history.shape[1]>1 else 0.
    return {"stage":"A_all_real_requests","task":z.task_id,"source":z.task_source,
        "request_tick":z.tick,"collector":collector,"episode":episode,
        "speed_rms_rad_s":speed,"speed_bin":"low" if speed<.2 else "medium" if speed<.7 else "high",
        "root_speed_m_s":float(z.state[:,7:10].norm(dim=-1).mean()),
        "leg_speed_rad_s":leg_speed,
        "joint_range_rad":amplitude,"amplitude_bin":"low" if amplitude<.1 else "medium" if amplitude<.4 else "high",
        "motion_class":category,"declared_motion_class":z.motion_class,"history_valid_steps":int(z.genmo_valid.sum()),
        "contact":"not_recorded_at_request","amplitude_scope":"request_history_not_future_H"}


def describe_sample(sample, adapter=None):
    states = torch.cat([x.end.state.detach() for x in sample.transitions])
    speed = float(states[:, 34:55].square().mean().sqrt())
    amplitude = float((states[:,13:34].amax(0)-states[:,13:34].amin(0)).mean())
    contact = sample.transitions[0].event.get("contacts")
    phase = ("unknown" if contact is None else "double_support" if contact["left_foot"] and contact["right_foot"]
             else "single_support" if contact["left_foot"] != contact["right_foot"] else "airborne")
    contacts=[x.event.get("contacts") for x in sample.transitions]
    changes=sum(a is not None and b is not None and (a["left_foot"],a["right_foot"])!=(b["left_foot"],b["right_foot"])
                for a,b in zip(contacts,contacts[1:]))
    category = "low_motion" if speed < .2 else "whole_body_motion"
    if adapter is not None:
        names = adapter.bundle.spec.joint_names
        arm = [i for i,n in enumerate(names) if "arm" in n or "elbow" in n]
        leg = [i for i,n in enumerate(names) if "leg" in n or "knee" in n or "ankle" in n]
        ar = float(states[:,34:55][:,arm].square().mean().sqrt())
        lr = float(states[:,34:55][:,leg].square().mean().sqrt())
        if ar > .7 and ar > 1.5*lr: category = "active_upper_body"
        elif phase == "single_support" and lr > .4: category = "stepping"
        elif float(states[:,12].abs().mean()) > .2: category = "turning"
    return {"source": sample.state.task_source, "task": sample.state.task_id,
        "declared_motion_class": sample.state.motion_class, "motion_class": category,
        "speed_rms_rad_s": speed, "joint_range_rad": amplitude, "contact": phase,
        "root_speed_m_s":float(states[:,7:10].norm(dim=-1).mean()),"contact_changes":changes,
        "leg_speed_rad_s":lr if adapter is not None else None,
        "speed_bin": "low" if speed < .2 else "medium" if speed < .7 else "high",
        "amplitude_bin": "low" if amplitude < .1 else "medium" if amplitude < .4 else "high",
        "request_tick": sample.generation.request.ticket.request_tick,
        "actor_version":sample.generation.request.actor_version,"reference_version":sample.generation.reference_version,
        "collector": sample.collector_id, "episode": sample.episode_index}


def stratified_sample(eligible, count, rng,labels=None):
    labels = labels if labels is not None else [describe_sample(x) for x in eligible]
    def stratum(x):
        return (x["source"],x["speed_bin"],x["amplitude_bin"],x["contact"],bool(x["contact_changes"]),x["motion_class"])
    counts = Counter(stratum(x) for x in labels)
    remaining = list(range(len(eligible)))
    selected = [];used=set()
    for _ in range(count):
        weights = [(1. if stratum(labels[i]) not in used else .1)/counts[stratum(labels[i])] for i in remaining]
        index = rng.choices(remaining, weights=weights, k=1)[0]
        selected.append(eligible[index]);remaining.remove(index)
        used.add(stratum(labels[index]))
    return selected


def coverage_summary(rows,minimum_high=1,minimum_contact_change=1):
    high=sum(r.get("speed_bin")=="high" for r in rows)
    observed=[r for r in rows if r.get("contact_changes") is not None]
    transitions=sum(r["contact_changes"]>0 for r in observed) if observed else None
    reasons=[]
    if high<minimum_high:reasons.append("insufficient_effective_high_speed")
    if transitions is None:reasons.append("contact_transition_not_observed")
    elif transitions<minimum_contact_change:reasons.append("insufficient_effective_contact_transition")
    def statistics(key):
        values=sorted(r[key] for r in rows if r.get(key) is not None)
        return {"samples":len(values),"minimum":values[0],"mean":sum(values)/len(values),
            "median":values[len(values)//2],"maximum":values[-1]} if values else {"samples":0}
    return {"samples":len(rows),"high_speed":high,"contact_transition":transitions,
        "contact_transition_observed_samples":len(observed),
        "coverage_insufficient":bool(reasons),"reasons":reasons,
        "motion_statistics":{k:statistics(k) for k in ("speed_rms_rad_s","root_speed_m_s","leg_speed_rad_s","joint_range_rad")},
        "counts":[{"source":k[0],"speed":k[1],"contact":k[2],"motion_class":k[3],"amplitude":k[4],"count":v}
            for k,v in Counter((r.get("source"),r.get("speed_bin"),r.get("contact"),r.get("motion_class"),r.get("amplitude_bin")) for r in rows).items()]}
