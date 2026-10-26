"""真实短轨迹的训练覆盖统计与分层抽样，不用WM预测替代实际运动标签。

速度来自真实state55的关节速度，幅度来自同一anchor真实H步的关节范围，接触来自
Newton求解器的已发生接触事件。每个候选都记录来源、动作类别、合法/超域状态。
分层抽样只改变同Actor版本已有合法候选的选择，不扩大WM支持域、不伪造困难动作。
"""

from collections import Counter
import torch


def describe_sample(sample, adapter=None):
    states = torch.cat([x.end.state.detach() for x in sample.transitions])
    speed = float(states[:, 34:55].square().mean().sqrt())
    amplitude = float((states[:,13:34].amax(0)-states[:,13:34].amin(0)).mean())
    contact = sample.transitions[0].event.get("contacts")
    phase = ("unknown" if contact is None else "double_support" if contact["left_foot"] and contact["right_foot"]
             else "single_support" if contact["left_foot"] != contact["right_foot"] else "airborne")
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
        "speed_bin": "low" if speed < .2 else "medium" if speed < .7 else "high",
        "amplitude_bin": "low" if amplitude < .1 else "medium" if amplitude < .4 else "high",
        "request_tick": sample.generation.request.ticket.request_tick,
        "collector": sample.collector_id, "episode": sample.episode_index}


def stratified_sample(eligible, count, rng):
    labels = [describe_sample(x) for x in eligible]
    counts = Counter((x["source"], x["speed_bin"], x["contact"]) for x in labels)
    remaining = list(range(len(eligible)))
    selected = []
    for _ in range(count):
        weights = [1/counts[(labels[i]["source"], labels[i]["speed_bin"], labels[i]["contact"])] for i in remaining]
        index = rng.choices(remaining, weights=weights, k=1)[0]
        selected.append(eligible[index]);remaining.remove(index)
    return selected
