"""真实离线请求的静态筛选、运动分层和独立可恢复抽样。

只读取v2连续物理数据和请求metadata，在GENMO/FK装配之前排除零前缀、缺失真实起点及
H窗口/速度halo不足。真实运动和接触类别仅决定训练抽样与认证，绝不传入Actor条件。
来源与时间使用独立PCG游标，优先轮换音乐、在源内均匀抽不同相位；监督入口不受影响。
分类阈值是预声明物理单位，不随学生Loss或验证成功率调整。认证信息与音乐白名单分离。
"""

from collections import Counter, defaultdict
import copy
import json

import numpy as np
import torch

from gem.closedloop.stage1_dataset import collate_stage1_training_samples
from gem.closedloop.training import batch_to_device
from .contracts import source_key, PhysicsRejected

CLASSIFICATION = {"standing_speed_m_s": .08, "standing_joint_speed_rad_s": .35,
                  "arm_speed_rad_s": .5, "leg_speed_rad_s": .5,
                  "turn_speed_rad_s": .25, "high_excursion_rad": .5, "fast_joint_rad_s": 2.}


def motion_tags(states, joint_names, contacts=None):
    from supermimic.mdp import unflatten_state
    _, r, v, w, q, qd = unflatten_state(states.detach().float())
    arms = [i for i,n in enumerate(joint_names) if any(k in n.lower() for k in ("arm", "shoulder", "elbow"))]
    legs = [i for i,n in enumerate(joint_names) if any(k in n.lower() for k in ("leg", "hip", "knee", "ankle"))]
    speed = float(v[..., :2].norm(dim=-1).mean())
    angular = (r.transpose(-1,-2) @ w[...,None]).squeeze(-1)
    tags = []
    if speed < CLASSIFICATION["standing_speed_m_s"] and float(qd.abs().mean()) < CLASSIFICATION["standing_joint_speed_rad_s"]:
        tags.append("standing")
    if arms and float(qd[...,arms].abs().mean()) >= CLASSIFICATION["arm_speed_rad_s"]:
        tags.append("arm_swing")
    if legs and (float(qd[...,legs].abs().mean()) >= CLASSIFICATION["leg_speed_rad_s"] or speed >= .08):
        tags.append("stepping")
    if float(angular[...,2].abs().mean()) >= CLASSIFICATION["turn_speed_rad_s"]:
        tags.append("turning")
    if float((q.amax(0)-q.amin(0)).amax()) >= CLASSIFICATION["high_excursion_rad"]:
        tags.append("high_amplitude")
    if float(qd.abs().amax()) >= CLASSIFICATION["fast_joint_rad_s"]:
        tags.append("fast_joints")
    if not tags:
        tags.append("low_motion")
    if contacts is not None:
        modes = [(bool(c.get("left_foot",False)), bool(c.get("right_foot",False))) for c in contacts]
        if any(a != b for a,b in modes):
            tags.append("single_support")
        if any(a and b for a,b in modes):
            tags.append("double_support")
        if any(not a and not b for a,b in modes):
            tags.append("airborne")
        if any((not a and c) or (not b and d) for (a,b),(c,d) in zip(modes,modes[1:])):
            tags.append("landing")
    return sorted(tags)


def static_request(episode, index, horizon):
    request = episode["offline_requests"]
    tick = int(request["frames"][index])*20
    p = int(request["prefix_frames"][index])
    if not bool(request["valid"][index]):
        return None, request["reasons"][index]
    if p < 1:
        return None, "insufficient_known_prefix"
    first = int(request["forecast_ticks"][index])
    if first < tick or first % 12 or tick % 60:
        return None, "sidecar_time_mismatch"
    if first+horizon*12 > (len(episode["state"])-1)*12:
        return None, "incomplete_real_window"
    # read_current_future在末步后还需要当前速度右端点；预览最远到末步+132tick。
    if first+(horizon-1)*12+132 > tick+59*20:
        return None, "candidate_right_halo_insufficient"
    if tick+59*20 >= episode["source_frames"]*20:
        return None, "incomplete_data_window"
    return {"frame": tick//20, "request_tick": tick, "start_tick": first, "prefix_frames": p}, None


class PhysicsRequestStream:
    def __init__(self, catalog, store, joint_names, horizon, seed, rank=0, certification=None, *, historical_mode="strict", noise_levels=None):
        self.catalog, self.store, self.horizon = catalog, store, horizon
        self.source_rng = np.random.default_rng(seed+3000+rank)
        self.time_rng = np.random.default_rng(seed+4000+rank)
        self.draws, self.order, self.cursor = 0, [], 0
        self.sources, self.requests, self.inventory = {}, {}, []
        initial_guards = {}
        for dataset,row in catalog.sources():
            key = json.dumps([dataset.dataset_name, dataset.split, dataset.rows[row]["sample_id"]], separators=(",",":"),ensure_ascii=False)
            if key not in store.index["sources"]:
                self.inventory.append({"source": key, "reason": "missing_real_sidecar", "valid": 0})
                continue
            meta = dataset.get_window(row, 0)["meta"]
            episode = store.episode(meta)
            accepted, reasons, categories = [], Counter(), Counter()
            for i in range(len(episode["offline_requests"]["frames"])):
                item, reason = static_request(episode, i, horizon)
                if reason:
                    reasons[reason] += 1
                    continue
                start = item["start_tick"]//12
                tags = motion_tags(episode["state"][start:start+horizon+1], joint_names,
                                   episode["contacts"][start:start+horizon])
                item.update(tags=tags, source=key, dataset=dataset.dataset_name, music_identity=episode["music_sha256"])
                categories.update(tags)
                item["historically_certified"] = certification.certified(tags) if certification is not None else False
                item["timesteps"] = list(noise_levels or [50, 100, 250])
                if not torch.isfinite(episode["state"][start:start+horizon+1]).all():
                    reasons["nonfinite_real_window"] += 1
                    continue
                if historical_mode == "strict" and certification is not None and not certification.certified(tags):
                    reasons["uncertified_motion_type"] += 1
                    continue
                if historical_mode == "strict" and certification is not None:
                    profile = '|'.join(tags)
                    if profile not in initial_guards:
                        initial_guards[profile] = certification.for_tags(tags,device='cpu')
                    try:
                        initial_guards[profile].check_state(episode['state'][start:start+1])
                    except PhysicsRejected:
                        reasons['static_real_start_out_of_domain'] += 1
                        continue
                    try:
                        initial_guards[profile].check_state(episode['state'][start:start+horizon+1])
                    except PhysicsRejected:
                        reasons['static_real_window_state_out_of_domain'] += 1
                        continue
                    try:
                        initial_guards[profile].check_action(episode['action'][start:start+horizon])
                    except PhysicsRejected:
                        reasons['static_real_window_action_out_of_domain'] += 1
                        continue
                    item["timesteps"] = certification.timesteps_for(tags)
                    if not item["timesteps"]:
                        reasons["uncertified_noise_level"] += 1
                        continue
                accepted.append(item)
            self.inventory.append({"source": key, "valid": len(accepted), "rejections": dict(reasons), "categories": dict(categories)})
            if accepted:
                self.sources[key], self.requests[key] = (dataset,row), accepted
        self.keys = sorted(self.sources)
        self.groups = {}
        for key, options in self.requests.items():
            grouped = defaultdict(list)
            for request in options:
                grouped["|".join(request["tags"])].append(request)
            self.groups[key] = [grouped[name] for name in sorted(grouped)]
        self.identity = {"sidecar": store.identity, "horizon": horizon, "keys": self.keys,
                         "static_selection":"time_prefix_halo_profile_real_window_state_action_v2", "historical_mode": historical_mode,
                         "classification": CLASSIFICATION, "catalog": catalog.identity}

    def next(self, count, device):
        if not self.keys:
            raise PhysicsRejected("no_certified_offline_requests")
        samples, records = [], []
        for _ in range(count):
            if self.cursor == len(self.order):
                self.order = self.source_rng.permutation(len(self.keys)).tolist()
                self.cursor = 0
            key = self.keys[self.order[self.cursor]]
            self.cursor += 1
            groups = self.groups[key]
            group = groups[int(self.time_rng.integers(len(groups)))]
            request = group[int(self.time_rng.integers(len(group)))]
            dataset,row = self.sources[key]
            sample = dataset.get_window(row, request["frame"])
            if source_key(sample["meta"]) != key or int(sample["prefix_frames"]) != request["prefix_frames"]:
                raise ValueError("静态索引与原数据窗口身份不一致")
            samples.append(sample)
            records.append(dict(request, draw=self.draws))
            self.draws += 1
        return batch_to_device(collate_stage1_training_samples(samples),device), records

    def state_dict(self):
        return {"source_rng": copy.deepcopy(self.source_rng.bit_generator.state),
                "time_rng": copy.deepcopy(self.time_rng.bit_generator.state), "draws": self.draws,
                "order": self.order.copy(), "cursor": self.cursor, "identity": self.identity}

    def load_state_dict(self, state):
        if state["identity"] != self.identity:
            raise ValueError("独立物理采样的来源/时间支持合同不一致")
        self.source_rng.bit_generator.state = state["source_rng"]
        self.time_rng.bit_generator.state = state["time_rng"]
        self.draws, self.order, self.cursor = state["draws"], state["order"], state["cursor"]
