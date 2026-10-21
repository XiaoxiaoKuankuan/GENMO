"""限定预算的同Replay单因素对照，不改已完成运行、不追加采集或重新拟合Critic。

输入为已完成四组实验的原始工作树、Replay快照、冻结优势和固定训练音乐评价计划。
从原s595000重新初始化，复用原ActorTrainer，仅将Min-SNR切回uniform；保持真实
优势权重、来源因子、batch顺序、噪声、95步对应的Epoch预算、学习率和无BC状态。
所有训练与评价写入新的独立目录；原state、Replay、标签、模型均只读。原始源码
提交与工具SHA分别记录，评价复用原物理身份/时钟和任务随机流并新建PhysX世界。
group_only变体保持原时间目标，仅将非首次有效候选的优势设为零(权重为1)，直接检验
普通后续局部信用的加权是否有益；候选/普通监督份额、噪声、步数和学习率完全不变。
natural_unweighted与natural_awr只取消首次候选的固定监督份额，恢复普通来源平衡，
分别使用全1和原真实优势权重；用于同Replay隔离过度强调首次状态的影响。候选身份、
样本顺序、扩散噪声、训练步数及奖励不变，不以新的采集分布冒充单因素对照。
nstep变体仅将普通后续样本的参考信用优势改为已冻结的普通N-step优势；首次组优势、
候选监督份额及Critic保持原样，用真实物理对照判断参考暴露近似是否有额外价值。
只允许八卡、一个64次以内的已完成有限运行；启动和评价前检查剩余墙钟，STOP或
时间边界保存完整优化步，物理阶段停止发新任务并排空。它不是原运行的隐式续训。
"""
from __future__ import annotations

import argparse
import copy
from datetime import datetime, timedelta
import hashlib
import json
import os
from pathlib import Path
import sys
import time


def uniform_config(actor_config):
    """只改变时间目标；禁止顺带开启BC、教师、修改学习率或训练预算。"""
    result = copy.deepcopy(actor_config)
    if result.get("bc_enabled", True) or result.get("reference_weight", 0.) != 0.:
        raise ValueError("This bounded control requires the original explicit no-BC/no-teacher run")
    if result.get("timestep_weighting", {}).get("mode") != "min_snr":
        raise ValueError("Source must be the Min-SNR experiment")
    result["timestep_weighting"] = dict(mode="uniform")
    return result


def group_only_weights(labels, sample_ids, *, ordinary="zero"):
    """首次保留组优势，其他样本置零或切换冻结N-step；整轮归一化，不修改原标签。"""
    import torch
    from gem.closedloop.awr.advantages import label_advantages
    if labels.group_valid is None or not labels.group_valid.any():
        raise ValueError("A group-only ablation requires actual verified candidate labels")
    lookup = {s:i for i,s in enumerate(labels.sample_ids)}
    ids = torch.tensor([lookup[s] for s in sample_ids])
    if set(sample_ids) != set(lookup):
        raise ValueError("Group-only weights must cover the identical Replay")
    normalized = torch.zeros_like(labels.weights)
    if ordinary == "nstep":
        scale = labels.identity["component_scales"]["nstep"]
        if not scale["available"] or labels.nstep_advantage is None:
            raise ValueError("N-step control requires frozen cross-fitted N-step labels")
        normalized = labels.nstep_advantage/max(scale["std"], 1e-8)
    elif ordinary != "zero":
        raise ValueError("Ordinary signal must be zero or nstep")
    sigma = labels.identity["component_scales"]["group"]["std"]
    normalized[labels.group_valid] = labels.group_advantage[labels.group_valid]/max(sigma, 1e-8)
    c = labels.identity["weight_config"]
    counterfactual = label_advantages(labels.sample_ids, labels.returns, labels.values,
        sources=labels.source, identity=dict(original_label_id=labels.identity["label_id"],
            ablation="verified_first_group_ordinary_"+ordinary,
            source_signal_config=labels.identity.get("signal_config"),
            component_scales=labels.identity["component_scales"]),
        beta=c["beta"], weight_cap=c["weight_cap"], epsilon=c["epsilon"],
        weight_mean_normalize=c["weight_mean_normalize"], final_normalized=normalized, allow_degenerate=True)
    for key in ("mc_advantage", "nstep_advantage", "credit_advantage", "group_advantage",
                "credit_valid", "credit_exposure", "group_valid"):
        value = getattr(labels, key)
        setattr(counterfactual, key, value.clone() if value is not None else None)
    counterfactual.source = list(labels.source)
    counterfactual.label_mode = "counterfactual_"+ordinary+":grouped"
    return counterfactual.weights[ids].clone(), counterfactual


def natural_candidate_config(actor_config):
    """取消显式候选份额，保留来源平衡和原更新预算；不修改输入配置。"""
    result = copy.deepcopy(actor_config)
    if (result.get("bc_enabled", True) or result.get("reference_weight", 0.) != 0.
            or result.get("source_balance") != "uniform_sources"
            or result.get("candidate_objective_fraction") is None):
        raise ValueError("Require the explicit no-BC/no-teacher candidate allocation experiment")
    result["candidate_objective_fraction"] = None
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-run", type=Path, required=True)
    parser.add_argument("--runtime-repo", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--finish-before-utc", required=True)
    parser.add_argument("--variant", choices=("uniform_time", "group_only", "natural_unweighted", "natural_awr", "nstep"),
                        default="uniform_time")
    args = parser.parse_args()
    source, repo, output = [p.resolve() for p in (args.source_run, args.runtime_repo, args.output_dir)]
    deadline = datetime.fromisoformat(args.finish_before_utc.replace("Z", "+00:00")).timestamp()
    if output.exists() or output.parent != source.parent or deadline-time.time() < 20*60:
        parser.error("Require a new sibling output and at least twenty minutes before the hard task deadline")
    before = (source/"state.json").read_bytes()
    state = json.loads(before)
    request = json.loads((source/"request.json").read_text())
    if (state["status"] != "complete" or state["macro"] != 1 or state["config"]["awr"]["initial_episodes"] > 64
            or Path(request["paths"]["genmo_repo"]).resolve() != repo):
        parser.error("Require the completed bounded experiment and its original source")
    sys.path.insert(0, str(repo))
    import torch
    import torch.distributed as dist
    from gem.closedloop.awr.config import preflight, physical_config
    from gem.closedloop.awr.distributed import Distributed
    from gem.closedloop.awr.io import atomic_json, read_tree, digest, cpu_copy, file_sha256
    from gem.closedloop.awr.records import AWRCompleteEpisode, TRAINING_METADATA_KEYS
    from gem.closedloop.awr.dataset import ReplayDataset
    from gem.closedloop.awr.advantages import AdvantageLabelSet
    from gem.closedloop.awr.trainer import ActorTrainer
    from gem.closedloop.awr.checkpoint import Checkpoints, capture_rng
    from gem.closedloop.awr.evaluation import evaluate, paired_comparison
    from gem.closedloop.awr.metrics import parameter_changes, stage1_validation
    from gem.closedloop.awr.sampler import AWRSampler
    from gem.closedloop.awr.resources import prepare_file_handles, GPUResourceMonitor
    from gem.closedloop.common.supervision import load_actor, Stage1Supervision
    from gem.closedloop.common.full_dataset import FullMusicCatalog
    from gem.closedloop.online_conditions import OnlineConditionBuilder
    rank = int(os.environ["RANK"])
    if (int(os.environ["WORLD_SIZE"]) != 8 or int(os.environ["LOCAL_RANK"]) != rank
            or os.environ.get("CUDA_VISIBLE_DEVICES") != "0,1,2,3,4,5,6,7"):
        parser.error("Use exactly eight local ranks")
    dist.init_process_group("gloo", timeout=timedelta(minutes=30))
    group = monitor = None
    started = time.time()
    try:
        d = Distributed(f"cuda:{rank}")
        pre = d.root(lambda: preflight(request))
        if state["source_commits"] != {k:v["commit"] for k,v in pre["repositories"].items()}:
            raise ValueError("Original model/physics source identity changed")
        if pre["configuration_assets"] != state["configuration_assets"]:
            raise ValueError("Frozen physical configuration changed")
        d.root(lambda: output.mkdir(parents=True, exist_ok=False))
        torch.cuda.set_device(rank); torch.set_num_threads(4)
        torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
        torch.backends.cuda.enable_flash_sdp(False); torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_math_sdp(True)
        group = dist.new_group(backend="nccl", timeout=timedelta(minutes=30)); d.tensor_group = group
        prepare_file_handles(state["config"]["awr"]["replay_capacity"])
        config = copy.deepcopy(state["config"])
        c = (uniform_config(config["awr"]["actor"]) if args.variant == "uniform_time"
             else copy.deepcopy(config["awr"]["actor"]))
        if args.variant in ("natural_unweighted", "natural_awr"):
            c = natural_candidate_config(c)
        if c.get("bc_enabled", True) or c.get("reference_weight", 0.) != 0.:
            raise ValueError("This diagnostic requires BC and teacher disabled")
        snapshot_id = state["artifacts"]["macro000_replay"]
        identity = dict(source_run=str(source), variant=args.variant, original_sources=state["source_commits"],
            source_request=state["request_sha256"], replay_snapshot=snapshot_id, actor_config=c,
            label_manifest_sha256=file_sha256(Path(state["artifacts"]["macro000_labels"])/"manifest.json"),
            tool_sha256=file_sha256(__file__), source_state_sha256=hashlib.sha256(before).hexdigest())
        progress = dict(status="running", phase="load", identity=identity, started_unix=started)
        def publish(**values):
            progress.update(values); d.root(lambda: atomic_json(output/"state.json", progress))
        publish()
        # 保留原运行名/namespace，评价seed与任务计划不因输出目录改变。
        runtime = physical_config(config, run_id=source.name, rank=rank, device=d.device)
        os.environ["ISAACLAB_PATH"] = config["runtime"]["isaaclab_root"]
        os.environ["LD_LIBRARY_PATH"] = config["runtime"]["sysroot_library"]+os.pathsep+os.environ.get("LD_LIBRARY_PATH", "")
        actor, train_config, loaded = d.local(lambda: load_actor(runtime)); d.sync_model(actor)
        if rank == 0: monitor = GPUResourceMonitor(output).start()
        baseline = json.loads(Path(state["artifacts"]["stage1_fixed"]).read_text())
        if d.assert_models_equal(actor) != baseline["actor_fingerprint"]:
            raise ValueError("Ablation initialization differs from frozen Stage1")
        initial = cpu_copy(actor.state_dict())
        snapshot = json.loads((source/"replay/snapshots"/(snapshot_id+".json")).read_text())
        def load_dataset():
            episodes, entries = [], []
            for name in snapshot["episode_ids"]:
                path = source/"replay/episodes"/name
                episode = AWRCompleteEpisode.from_dict(read_tree(path, episode_metadata_keys=TRAINING_METADATA_KEYS))
                episode.training_view = True
                if episode.episode_id != name or digest(episode.contract) != snapshot["contract_sha256"]:
                    raise ValueError("Replay Episode or physical contract changed")
                episodes.append(episode); entries.append([name, json.loads((path/"manifest.json").read_text())["sha256"]])
            if digest(dict(contract=snapshot["contract_sha256"], episodes=entries)) != snapshot_id:
                raise ValueError("Immutable Replay snapshot changed")
            if snapshot["contract_sha256"] != baseline["physics_identity"]:
                raise ValueError("Baseline and Replay physics differ")
            labels = AdvantageLabelSet.load(state["artifacts"]["macro000_labels"])
            dataset = ReplayDataset(episodes, labels, expected_signal=config["awr"]["advantage"],
                expected_candidates=config["awr"]["candidate_sampling"], expected_snapshot=snapshot_id)
            if args.variant in ("group_only", "nstep"):
                dataset.weights, counterfactual = group_only_weights(labels, dataset.sample_ids,
                    ordinary="nstep" if args.variant == "nstep" else "zero")
                if rank == 0:
                    counterfactual.save(output/"counterfactual_labels")
            elif args.variant == "natural_unweighted":
                dataset.weights = torch.ones_like(dataset.weights)
            return dataset
        data = d.local(load_dataset)
        torch.manual_seed(config["awr"]["seed"]+rank*100003)
        trainer = ActorTrainer(actor, data, None, c, d, seed=config["awr"]["seed"], bc_weight=0.)
        cp = Checkpoints(output/"checkpoints", identity)
        def save(name):
            ranks = d.gather(dict(bc=None, rng=capture_rng(d.device)))
            payload = trainer.state_dict() if rank == 0 else None
            if payload is not None: payload["rank_states"] = ranks
            return d.root(lambda: cp.save(name, payload, progress=dict(step=trainer.steps, phase="actor_"+args.variant)))
        def step(current, entry):
            if rank == 0:
                with (output/"steps.jsonl").open("a") as f: f.write(json.dumps(entry)+"\n")
        stop = lambda: (output/"STOP").exists() or time.time() >= deadline-10*60
        publish(phase="actor_"+args.variant)
        result = trainer.run(on_step=step, should_stop=stop)
        final = save("final" if result["complete"] else "stopped")
        publish(checkpoint=final, training=d.gather(result))
        if not result["complete"] or deadline-time.time() < 15*60:
            publish(status="stopped", phase="before_evaluation"); return
        changes = d.root(lambda: parameter_changes(actor, initial))
        del initial, trainer, data
        bc = Stage1Supervision(runtime, actor, train_config, source_step=loaded["source_global_step"], split="train")
        losses = d.gather(stage1_validation(actor, bc, batches=4, batch_size=config["awr"]["actor"]["bc_microbatch"],
            device=d.device, seed=config["awr"]["seed"]+700001+rank))
        del bc
        catalog = d.local(lambda: FullMusicCatalog(config["paths"]["data_root"]))
        audit = json.loads(Path(state["artifacts"]["data_audit"]).read_text())
        d.local(lambda: catalog.apply_audit(audit))
        plan = json.loads(Path(state["artifacts"]["evaluation_fixed"]).read_text())
        policy = AWRSampler(actor)
        builder = OnlineConditionBuilder(actor.endecoder.codec, history_steps=50, motion_frames=60)
        publish(phase="evaluation")
        evaluation = evaluate(runtime, policy, builder, catalog, plan, output/"evaluation", d,
            fingerprint=d.assert_models_equal(actor), physics_identity=baseline["physics_identity"],
            progress_path=output/"evaluation_progress", should_stop=stop)
        comparisons = {}
        comparison_keys = (("stage1", "stage1_fixed"), ("self_distillation", "macro000_self_distillation_fixed"),
                           ("source_awr", "macro000_awr_fixed"))
        if args.variant == "uniform_time":
            comparison_keys = (("stage1", "stage1_fixed"), ("uniform_self_distillation", "macro000_self_distillation_uniform_fixed"),
                               ("minsnr_self_distillation", "macro000_self_distillation_fixed"), ("minsnr_awr", "macro000_awr_fixed"))
        for name, key in comparison_keys:
            previous = json.loads(Path(state["artifacts"][key]).read_text())
            comparisons[name] = paired_comparison(previous, evaluation, selection_policy="reward_only")
        if (source/"state.json").read_bytes() != before:
            raise ValueError("Source experiment was mutated during independent control")
        schema = "genmo.awr.uniform_time_ablation.v1" if args.variant == "uniform_time" else "genmo.awr.replay_objective_ablation.v1"
        report = dict(schema=schema, variant=args.variant, identity=identity, checkpoint=final,
            training=progress["training"], evaluation=evaluation, comparisons=comparisons,
            parameter_changes=changes, stage1_train_diagnostic_loss=sum(map(sum,losses))/sum(map(len,losses)),
            source_state_unchanged=True, new_collection_episodes=0, new_critic_updates=0,
            seconds=time.time()-started, completed_unix=time.time())
        d.root(lambda: atomic_json(output/"report.json", report))
        publish(status="complete", phase="complete", finished_unix=time.time())
    finally:
        if monitor is not None: monitor.close()
        if group is not None: dist.destroy_process_group(group)
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
