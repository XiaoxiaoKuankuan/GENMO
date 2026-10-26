"""固定当前GENMO的真实完整episode MC采集与独立Upper Critic预热。

训练/验证音乐ID和随机种子分离；从真实终止向前累计带dt和变长折扣的MC目标，
行政截断不能伪装完整回报。预先配置MSE、解释方差与训练均值常数基线改进门槛，
达到门槛才允许使用该Critic；lambdaV=0的短时Actor依据独立物理验证，不由Critic
失败自动否决。质量不合格留存原始目标和曲线并退出独立拟合；不修改WM，
不将想象目标当成独立验证集。保存紧凑快照用于审计，不保存物理求解器隐藏状态。
"""

import json
import random
from pathlib import Path

import torch

from .application import task_inputs, seed_all
from .checkpoint import pack_state,pack_value
from .music_mc import critic_reward_contract
from .returns import close_real_mc_episode


def choose_warmup_tasks(tasks, config):
    groups = {}
    for task in tasks:
        groups.setdefault(task["source"], []).append(task)
    rng = random.Random(config["seed"])
    for values in groups.values():
        rng.shuffle(values)
    chosen = []
    while len(chosen) < config["train_episodes"] and any(groups.values()):
        for source in sorted(groups):
            if groups[source] and len(chosen) < config["train_episodes"]:
                chosen.append(groups[source].pop())
    return chosen


def save_warmup(path, trainer, cfg):
    """真实MC质量通过的独立价值状态，可在同模型/奖励/调度合同下导入正式训练。"""
    from supermimic.bridge.bundle import atomic_save
    if not trainer.critic_ready:
        raise ValueError("未通过独立质量的Critic不能导出为ready")
    schema=("genmo.real_mc_critic.online_teacher_alignment.v3" if cfg.get("teacher",{}).get("target_mode")=="online_consumed_reference_v1" else
            "genmo.real_mc_critic.teacher_music.v2" if cfg.get("teacher",{}).get("enabled") else "genmo.real_mc_critic.control50.v1")
    atomic_save({"schema": schema, "actor": trainer.actor.state_dict(),
        "critic": trainer.critic.state_dict(), "optimizer": trainer.critic_optimizer.state_dict(),
        "quality": trainer.critic_quality, "critic_updates": trainer.critic_updates,
        "contract": {k: cfg[k] for k in ("model", "clock", "rewards", "critic")},
        "teacher_identity": trainer.imagination.teacher.identity if getattr(trainer.imagination,"teacher",None) else None,
        "teacher_seed":cfg.get("teacher",{}).get("seed"),
        "critic_reward_contract":critic_reward_contract(cfg) if cfg.get("teacher",{}).get("target_mode")=="online_consumed_reference_v1" else None,
        "physics_determinism": cfg.get("execution", {}).get("physics_determinism", "not_guaranteed"),
        "frozen_sha256": trainer.imagination.adapter.bundle.metadata["sha256"]}, path)


def load_warmup(path, trainer, cfg):
    ck = torch.load(path, weights_only=False, map_location="cpu")
    expected_schema = ("genmo.real_mc_critic.online_teacher_alignment.v3" if cfg.get("teacher",{}).get("target_mode")=="online_consumed_reference_v1" else
        "genmo.real_mc_critic.teacher_music.v2" if cfg.get("teacher",{}).get("enabled") else "genmo.real_mc_critic.control50.v1")
    if ck.get("schema") != expected_schema or ck["quality"].get("passed") is not True:
        raise ValueError("需要任意50Hz真实MC质量通过的Upper Critic")
    if cfg.get("teacher",{}).get("enabled") and ck.get("teacher_identity") != trainer.imagination.teacher.identity:
        raise ValueError("新Critic必须来自当前冻结音乐教师与新执行奖励，不能导入旧价值")
    if cfg.get("teacher",{}).get("enabled") and ck.get("teacher_seed")!=cfg["teacher"]["seed"]:
        raise ValueError("固定音乐时间线的教师噪声种子不匹配，Critic目标不相同")
    if cfg.get("teacher",{}).get("target_mode")=="online_consumed_reference_v1":
        if ck.get("critic_reward_contract")!=critic_reward_contract(cfg):
            raise ValueError("旧MC奖励或在线教师目标合同不兼容，不允许导入")
    if ck["frozen_sha256"] != trainer.imagination.adapter.bundle.metadata["sha256"] or ck["contract"] != {
            k: cfg[k] for k in ("model", "clock", "rewards", "critic")}:
        raise ValueError("Critic实际资产、模型条件、奖励或调度合同改变")
    if ck.get("physics_determinism", "not_guaranteed") != cfg.get("execution", {}).get("physics_determinism", "not_guaranteed"):
        raise ValueError("价值预热与当前Newton确定性设置不一致")
    if any(not torch.equal(v.detach().cpu(), ck["actor"][n]) for n,v in trainer.actor.state_dict().items()):
        raise ValueError("预热回报不属于当前GENMO权重")
    trainer.critic.load_state_dict(ck["critic"], strict=True)
    trainer.critic_optimizer.load_state_dict(ck["optimizer"])
    trainer.critic_updates = ck["critic_updates"]
    trainer.critic_quality, trainer.critic_ready = ck["quality"], True
    return ck["quality"]


def collect_mc(collector, tasks, *, seed, device, max_control_steps, output=None, split="train", episode_offset=0):
    collector.record_controls = True
    pairs, episodes, steps = [], [], 0
    for i, task in enumerate(tasks):
        i += episode_offset
        seed_all(seed + i)
        music, initial = task_inputs(task, collector.adapter, device)
        collector.reset(music, initial)
        intervals = []
        while not collector.terminal:
            decisions = collector.collect(1)
            block = collector.take_controls()
            intervals.extend(block)
            steps += sum(x.steps for x in block)
            if steps > max_control_steps:
                raise RuntimeError("Critic真实MC采集达到预设控制步预算，禁止将截断伪装MC")
        close_real_mc_episode(intervals, decisions[-1])
        value = torch.zeros_like(intervals[-1].reward)
        episode_pairs = []
        for x in reversed(intervals):
            value = x.reward + x.continuation * value
            # 以五种50Hz相位覆盖非请求网格和pending；H端点相位也自然落在分布中。
            # 保存每五控制步，完整MC仍逐控制步递推，未采样时刻的奖励不会丢失。
            if (x.start.tick//12) % 5 == i % 5:
                episode_pairs.append((x.start.detached(), value.detach().clone()))
        pairs.extend(reversed(episode_pairs))
        if getattr(getattr(collector.rewards,"teacher",None),"config",{}).get("target_mode")=="online_consumed_reference_v1":
            from .music_mc import aligned_real_mc
            # 替换上面的旧世界坐标回报，当前音乐目标使用每个价值状态固定一次的对齐。
            if episode_pairs:del pairs[-len(episode_pairs):]
            episode_pairs=list(reversed(aligned_real_mc(intervals,collector.adapter,collector.rewards.config,
                collector.gamma,phase=i%5)))
            pairs.extend(reversed(episode_pairs));value=episode_pairs[-1][1]
        episode = {"task": task["id"], "seed": seed+i, "reason": intervals[-1].reason,
                         "mc_return_start_tick":episode_pairs[-1][0].tick if getattr(getattr(collector.rewards,"teacher",None),"config",{}).get("target_mode")=="online_consumed_reference_v1" else intervals[0].start.tick,
                         "steps": sum(x.steps for x in intervals), "mc_return": float(value.mean()),
                         "commits": sum(bool(x.get("accepted")) for x in collector.events),
                         "pending_value_samples": sum(z.pending for z, _ in episode_pairs),
                         "non_request_grid_samples": sum(z.tick % 60 != 0 for z, _ in episode_pairs)}
        episodes.append(episode)
        if output is not None:
            from supermimic.bridge.bundle import atomic_save
            directory = Path(output)/"mc_episodes"
            directory.mkdir(exist_ok=True)
            memo = {}
            atomic_save({"episode": episode, "pairs": [(pack_state(z, memo), y.cpu()) for z,y in reversed(episode_pairs)],
                         "controls":pack_value(tuple(intervals),memo),
                         "evidence": "real_complete_episode_mc"}, directory/f"{split}_{i:03d}.pt")
            with (Path(output)/"mc_progress.jsonl").open("a") as stream:
                stream.write(json.dumps({"split": split, "episode": i, **episode}, ensure_ascii=False)+"\n")
        print(json.dumps({"event": "critic_real_mc_episode", "split": split, "episode": i, **episode},
                         ensure_ascii=False), flush=True)
    return pairs, episodes, steps


@torch.no_grad()
def value_quality(critic, pairs, training_mean):
    critic.eval()
    targets = torch.cat([y.flatten() for _, y in pairs])
    predictions = torch.cat([critic(z).flatten() for z, _ in pairs])
    mse = float((predictions-targets).square().mean())
    baseline = float((targets-training_mean).square().mean())
    variance = float(targets.var(unbiased=False))
    return {"mse": mse, "constant_baseline_mse": baseline,
            "pending_samples": sum(z.pending for z, _ in pairs),
            "non_request_grid_samples": sum(z.tick % 60 != 0 for z, _ in pairs),
            "relative_mse_improvement": 1-mse/baseline if baseline > 1e-12 else None,
            "explained_variance": 1-float((targets-predictions).var(unbiased=False))/variance if variance > 1e-12 else None,
            "target_variance": variance, "samples": len(targets)}


def warmup_critic(trainer, collector, train_tasks, validation_tasks, config, output, *, device):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    if {t["id"] for t in train_tasks} & {t["id"] for t in validation_tasks}:
        raise ValueError("Critic预热与验证音乐必须独立")
    actor_before = {n: p.detach().clone() for n, p in trainer.actor.named_parameters()}
    train, train_episodes, ntrain = collect_mc(collector, train_tasks, seed=config["seed"], device=device,
        max_control_steps=config["max_collection_steps"], output=output, split="train")
    validation, val_episodes, nval = collect_mc(collector, validation_tasks, seed=config["seed"]+100000,
        device=device, max_control_steps=config["max_collection_steps"]-ntrain, output=output, split="validation")
    memo = {}
    payload = {"train": [(pack_state(z, memo), y.cpu()) for z,y in train],
               "validation": [(pack_state(z, memo), y.cpu()) for z,y in validation],
               "train_episodes": train_episodes, "validation_episodes": val_episodes, "criteria": config}
    torch.save(payload, output / "real_mc.pt")
    return fit_mc_dataset(trainer, train, validation, config, output, ntrain+nval, actor_before)


def fit_mc_dataset(trainer, train, validation, config, output, physical_steps, actor_before):
    """使用完整真实MC训练新Critic；允许多个物理进程提前收集，监督定义不变。"""
    output = Path(output)
    if min(len(train), len(validation)) < config["minimum_samples_per_split"] or not all(
            any(z.pending for z, _ in pairs) and any(z.tick % 60 for z, _ in pairs) for pairs in (train, validation)):
        raise RuntimeError("独立真实MC样本不足，Critic不能获准正式使用")
    mean = torch.cat([y for _, y in train]).mean()
    rng = random.Random(config["seed"])
    metrics = value_quality(trainer.critic, validation, mean)
    for update in range(1, config["max_updates"]+1):
        batch = [train[rng.randrange(len(train))] for _ in range(config["batch_size"])]
        loss, norm = trainer.fit_critic(batch)
        if update % config["validate_every"] != 0:
            continue
        metrics = value_quality(trainer.critic, validation, mean)
        passed = (update >= config["minimum_updates"] and metrics["mse"] <= config["maximum_mse"] and
                  metrics["explained_variance"] is not None and metrics["explained_variance"] >= config["minimum_explained_variance"] and
                  metrics["relative_mse_improvement"] is not None and metrics["relative_mse_improvement"] >= config["minimum_relative_improvement"])
        row = {"critic_update": update, "train_mse": loss, "gradient_norm": norm, **metrics, "passed": passed}
        with (output/"quality.jsonl").open("a") as stream:
            stream.write(json.dumps(row, allow_nan=False)+"\n")
        print(json.dumps({"event":"critic_warmup", **row}), flush=True)
        if passed:
            trainer.critic_ready = True
            break
    trainer.imagination.adapter.bundle.assert_unchanged(trainer.frozen_identity)
    if any(not torch.equal(p, actor_before[n]) for n, p in trainer.actor.named_parameters()):
        raise RuntimeError("Critic预热期间GENMO发生变化")
    trainer.critic_quality = {**metrics, "passed": trainer.critic_ready, "physical_control_steps": physical_steps,
                              "criteria": config, "targets": "real_complete_episode_mc", "genmo_unchanged": True}
    trainer.critic_quality["reward_mode"] = trainer.imagination.rewards.mode
    trainer.critic_quality["teacher_identity"] = (trainer.imagination.teacher.identity
        if getattr(trainer.imagination,"teacher",None) else None)
    (output/"quality.json").write_text(json.dumps(trainer.critic_quality, ensure_ascii=False, indent=2))
    if not trainer.critic_ready:
        raise RuntimeError("Critic独立验证未达标，不得用于Actor终点价值；lambdaV=0的短时Actor仍单独按物理证据验收")
    return trainer.critic_quality
