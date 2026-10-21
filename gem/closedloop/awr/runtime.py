"""AWR五阶段宏迭代状态机及Stage1/自蒸馏/AWR对照。

主rank发布元数据、完整Episode索引和checkpoint；物理rank只写不可变数组分片。
阶段产物提交后才推进状态，恢复读取原任务/Replay/标签，不改变预算。STOP只停止发
新任务，已开始Episode排空；Actor在完整优化步保存。所有物理阶段新建世界。
"""
from __future__ import annotations

import copy
import json
from pathlib import Path
import time
import uuid
from types import SimpleNamespace
import torch

from gem.closedloop.common.full_dataset import FullMusicCatalog
from gem.closedloop.common.supervision import load_actor, Stage1Supervision
from gem.closedloop.common.evaluation import _model_fingerprint
from gem.closedloop.online_conditions import OnlineConditionBuilder
from .config import physical_config
from .checkpoint import Checkpoints, capture_rng, stop_requested, GracefulStop
from .collection import CompleteCollector
from .sampler import AWRSampler
from .tasks import folds_for_catalog, training_tasks, evaluation_plan
from .world import calibrate, open_world, close_isolated_world, budget_for_tasks
from .replay import EpisodeReplay
from .dataset import ReplayDataset
from .fvi import fit_crossfit
from .advantages import AdvantageLabelSet
from .trainer import ActorTrainer, choose_bc_coefficient
from .evaluation import evaluate, paired_comparison, checkpoint_selection
from .io import atomic_json, read_tree, write_tree, cpu_copy, digest
from .metrics import exploration, parameter_changes, fixed_outputs, stage1_validation


class AWRRun:
    def __init__(self, config, path, distributed, preflight, *, resume=False):
        self.root, self.d = Path(path).resolve(), distributed
        self.request, self.preflight = copy.deepcopy(config), preflight
        def initialize():
            if resume:
                state = json.loads((self.root/"state.json").read_text())
                if state["request_sha256"] != digest(config):
                    raise ValueError("Resume cannot change the saved run configuration/budget")
                if state["source_commits"] != {k: v["commit"] for k, v in preflight["repositories"].items()}:
                    raise ValueError("Resume source commits differ from the immutable run")
                if state["configuration_assets"] != preflight["configuration_assets"]:
                    raise ValueError("Resume GMT control/Isaac configuration identity changed")
                (self.root/"STOP").unlink(missing_ok=True)
                return state
            self.root.mkdir(parents=True, exist_ok=False)
            value = dict(schema="genmo.awr.run.v1", request_sha256=digest(config),
                         source_commits={k: v["commit"] for k, v in preflight["repositories"].items()},
                         configuration_assets=preflight["configuration_assets"],
                         config=config, artifacts={}, macro=0, phase="initialize", status="running",
                         created_unix=time.time())
            atomic_json(self.root/"request.json", config)
            atomic_json(self.root/"preflight.json", preflight)
            atomic_json(self.root/"state.json", value)
            return value
        self.state = self.d.root(initialize)
        self.config = copy.deepcopy(self.state["config"])
        self.config["awr"]["source_commits"] = self.state["source_commits"]
        self.config["awr"]["configuration_assets"] = self.state["configuration_assets"]
        self.runtime_config = physical_config(self.config, run_id=self.root.name,
                                              rank=self.d.rank, device=self.d.device)
        self.checkpoints = Checkpoints(self.root/"checkpoints",
            dict(request=self.state["request_sha256"], sources=self.state["source_commits"]))
        self.actor, self.train_config, self.load_report = self.d.local(lambda: load_actor(self.runtime_config))
        self.d.sync_model(self.actor)
        self.initial = cpu_copy(self.actor.state_dict())
        self.baseline_fingerprint = self.d.assert_models_equal(self.actor)
        self.policy = AWRSampler(self.actor)
        self.builder = OnlineConditionBuilder(self.actor.endecoder.codec, history_steps=50, motion_frames=60)
        self.catalog = self.d.local(lambda: FullMusicCatalog(self.config["paths"]["data_root"]))
        self.replay = None

    def publish(self, **values):
        self.state.update(values)
        self.state["config"] = self.config
        self.d.root(lambda: atomic_json(self.root/"state.json", self.state))

    def artifact(self, name, value):
        self.state["artifacts"][name] = value
        self.publish()

    def json_artifact(self, name, factory):
        if name not in self.state["artifacts"]:
            value = self.d.root(factory)
            path = self.root/"artifacts"/(name+".json")
            self.d.root(lambda: atomic_json(path, value))
            self.artifact(name, str(path))
        return self.d.root(lambda: json.loads(Path(self.state["artifacts"][name]).read_text()))

    def checkpoint(self, name, payload, progress):
        reference = self.d.root(lambda: self.checkpoints.save(
            name+"_"+uuid.uuid4().hex[:8], payload, progress=progress))
        self.artifact(name, reference)
        return reference

    def prepare(self):
        from .resources import prepare_file_handles
        limits = self.d.local(lambda: prepare_file_handles(self.config["awr"]["replay_capacity"]))
        limits_by_rank = self.d.gather(limits)
        self.d.root(lambda: atomic_json(self.root/"resources"/("file_handles_"+uuid.uuid4().hex[:12]+".json"),
                                       dict(ranks=limits_by_rank, unix=time.time())))
        self.publish(phase="data_audit", status="running")
        if "data_audit" not in self.state["artifacts"]:
            def audit():
                count = 0
                def progress(_):
                    nonlocal count
                    count += 1
                    if count % 250 == 0:
                        print(f"[AWR] 完整清单核验 {count} 条", flush=True)
                return self.catalog.audit_files(progress)
            self.json_artifact("data_audit", audit)
        audit = self.d.root(lambda: json.loads(Path(self.state["artifacts"]["data_audit"]).read_text()))
        self.d.local(lambda: self.catalog.apply_audit(audit))
        self.folds = self.json_artifact("folds", lambda: folds_for_catalog(self.catalog, self.config["awr"]["seed"]))
        self.fixed_plan = self.json_artifact("evaluation_fixed", lambda: evaluation_plan(self.catalog, self.config))
        self.full_plan = self.json_artifact("evaluation_all", lambda: evaluation_plan(self.catalog, self.config, all_unique=True))
        if self.config["timing"]["deployment_profile"] is None:
            self.publish(phase="deployment_calibration")
            attempt = self.d.root(lambda: uuid.uuid4().hex[:8])
            profile = calibrate(self.runtime_config, self.policy, self.builder, self.catalog,
                                self.fixed_plan["tasks"], self.root/"calibration"/attempt, self.d, self.baseline_fingerprint)
            self.config["timing"]["deployment_profile"] = profile
            self.publish()
        self.runtime_config = physical_config(self.config, run_id=self.root.name, rank=self.d.rank, device=self.d.device)
        self.contract = dict(schema="genmo.awr.replay_physics.v1", assets=self.config["assets"],
            configuration_assets=self.state["configuration_assets"],
            reward=self.config["reward"], environment=self.config["environment"], termination=self.config["termination"],
            timing=self.config["timing"], sampler=self.policy.kernel_config,
            horizon_frames=60, source_commits=self.state["source_commits"])
        heldout = [s for split in ("val", "test") for pool in self.catalog.samples[split].values() for s in pool]
        options = dict(capacity=self.config["awr"]["replay_capacity"], contract=self.contract,
            heldout_groups=[s["group_id"] for s in heldout],
            heldout_audio=[s["row"]["source_audio_sha256"] for s in heldout],
            min_free_bytes=self.config["awr"]["storage"]["min_free_bytes"],
            max_bytes=self.config["awr"]["storage"]["max_replay_bytes"])
        self.d.root(lambda: EpisodeReplay(self.root/"replay", **options).close())
        self.replay = self.d.local(lambda: EpisodeReplay(self.root/"replay", read_only_index=self.d.rank != 0, **options))
        self.eval_actor("stage1_fixed", self.fixed_plan)
        if self.config["awr"]["final_all_unique"]:
            self.eval_actor("stage1_all", self.full_plan)
        self.bc_validation("stage1")

    def eval_actor(self, name, plan):
        if name in self.state["artifacts"]:
            return self.d.root(lambda: json.loads(Path(self.state["artifacts"][name]).read_text()))
        if self.d.root(lambda: stop_requested(self.root)):
            raise GracefulStop("Stopped before launching new evaluation tasks")
        self.publish(phase="evaluation_"+name)
        attempt = self.d.root(lambda: uuid.uuid4().hex[:8])
        report = evaluate(self.runtime_config, self.policy, self.builder, self.catalog, plan,
            self.root/"evaluation"/(name+"_"+attempt),
            self.d, fingerprint=self.d.assert_models_equal(self.actor), physics_identity=digest(self.contract),
            progress_path=self.root/"evaluation_progress"/name,
            should_stop=lambda: stop_requested(self.root))
        # 每个阶段只提交完全有效的报告；失败目录保留，重试使用独立尝试目录。
        path = self.root/"artifacts"/(name+".json")
        self.d.root(lambda: atomic_json(path, report))
        self.artifact(name, str(path))
        return report

    def bc_validation(self, name):
        key = name+"_bc_validation"
        if key in self.state["artifacts"]:
            return
        split = self.config["evaluation"].get("stage1_diagnostic_split", self.config["evaluation"]["split"])
        bc = Stage1Supervision(self.runtime_config, self.actor, self.train_config,
                               source_step=self.load_report["source_global_step"], split=split)
        seed = self.config["awr"]["seed"]+700001+self.d.rank
        losses = stage1_validation(self.actor, bc, batches=4,
            batch_size=self.config["awr"]["actor"]["bc_microbatch"], device=self.d.device, seed=seed)
        values = self.d.gather(losses)
        report = dict(stage1_full_loss=sum(sum(v) for v in values)/sum(map(len, values)),
                      batches=sum(map(len, values)), split=split, source_step=bc.source_step,
                      seeds=[self.config["awr"]["seed"]+700001+i for i in range(self.d.world_size)],
                      fixed_validation_randomness=True, training_rng_restored=True)
        self.json_artifact(key, lambda: report)
        del bc

    def collect(self, macro):
        key = f"macro{macro:03d}_replay"
        if key in self.state["artifacts"]:
            return self.replay.load_snapshot(self.state["artifacts"][key])
        self.publish(phase="collect", macro=macro)
        plan = self.json_artifact(f"macro{macro:03d}_tasks",
                                 lambda: training_tasks(self.catalog, self.folds, self.config, self.d.world_size, macro))
        completion = self.config["awr"].get("collection_completion")
        if completion and self.d.root(lambda: (self.root/f"macro{macro:03d}"/"supplement"/"plan.json").exists()):
            return self.complete_collection_with_budget()
        completed = self.d.root(lambda: self._recover_publications(macro))
        warmup_finished = []
        finite_attempt_plan = self.config["version"] == "genmo.stage2.awr.signals.v2"
        if completion or finite_attempt_plan:
            from .collection_budget import finished_warmup_tasks
            warmup_finished = self.d.root(lambda: finished_warmup_tasks(
                self.root, macro, {t["task_id"] for t in plan["tasks"]}))
        finished = set(completed) | set(warmup_finished)
        from .candidates import collection_rounds
        rounds = collection_rounds(plan["tasks"], self.config["awr"].get("candidate_sampling", {}),
                                   rank=self.d.rank, finished=finished)
        attempt = self.d.root(lambda: uuid.uuid4().hex[:8])
        path = self.root/f"macro{macro:03d}"/"collection"/attempt/f"rank{self.d.rank:02d}"
        handle, driver = None, None
        started = time.perf_counter()
        publications = []
        def run_local():
            nonlocal handle, driver
            if not tasks:
                return dict(tasks=[], seconds=0.)
            slots = min(self.config["runtime"]["num_envs"], len(tasks))
            handle = open_world(self.runtime_config, path/"world", self.d, num_envs=slots)
            driver = CompleteCollector(self.runtime_config, self.policy, self.builder, handle.world,
                budget_for_tasks(tasks, slots), path/"execution", num_envs=slots,
                actor_fingerprint=self.d_fingerprint)
            def publish_episode(episode):
                publication = self.replay.publish(episode)
                publication.update(task_id=episode.task["task_id"], macro=macro, published_unix_ns=time.time_ns())
                # 数组已经原子落盘后发布完成回执；恢复先登记这些回执，避免重复物理奖励。
                atomic_json(self.root/"replay"/"publications"/(episode.episode_id+".json"), publication)
                publications.append(publication)
                return publication
            return driver.collect(tasks, self.catalog, self.contract, on_episode=publish_episode,
                                  should_stop=lambda: stop_requested(self.root))
        self.d_fingerprint = self.d.assert_models_equal(self.actor)
        round_reports = []
        base_path = path
        for number, tasks in rounds:
            fresh_round = self.config["awr"].get("candidate_sampling", {}).get("world_lifecycle") == "fresh_candidate_round"
            path = base_path/f"round{number:03d}" if fresh_round else base_path
            handle = driver = None
            try:
                local_report = self.d.local(run_local)
            finally:
                if driver: driver.close()
                if handle: close_isolated_world(handle)
            round_reports.append(dict(round=number, report=local_report))
            if self.d.root(lambda: stop_requested(self.root)):
                break
        from collections import Counter
        timing, used = Counter(), Counter()
        for item in round_reports:
            report = item["report"]
            timing.update({k: v for k, v in report.get("timing", {}).items() if isinstance(v, (int, float))})
            used.update(report.get("actual_budget", {}).get("used", {}))
        local_report = dict(rounds=round_reports,
            tasks=[t for item in round_reports for t in item["report"]["tasks"]],
            seconds=sum(item["report"]["seconds"] for item in round_reports),
            timing=dict(timing), actual_budget=dict(used=dict(used)),
            generation_batches=[b for item in round_reports for b in item["report"].get("generation_batches", [])])
        reports = self.d.gather(local_report)
        self.json_artifact(f"macro{macro:03d}_collection_{attempt}", lambda: dict(ranks=reports, seconds=time.perf_counter()-started))
        completed = self.d.root(lambda: self._recover_publications(macro))
        if self.d.root(lambda: stop_requested(self.root)):
            self.publish(status="stopped", phase="collect")
            return None
        if len(completed) != len(plan["tasks"]):
            if self.d.root(lambda: stop_requested(self.root)):
                self.publish(status="stopped", phase="collect")
                return None
            if completion:
                return self.complete_collection_with_budget()
            if finite_attempt_plan:
                from .collection_budget import finished_warmup_tasks
                excluded = self.d.root(lambda: finished_warmup_tasks(
                    self.root, macro, {t["task_id"] for t in plan["tasks"]}))
                if set(completed) | set(excluded) != {t["task_id"] for t in plan["tasks"]}:
                    raise RuntimeError("Candidate plan has unfinished nonterminal tasks; resume them before labeling")
                self.json_artifact(f"macro{macro:03d}_candidate_completion", lambda: dict(
                    planned_attempts=len(plan["tasks"]), complete_episodes=len(completed),
                    warmup_failed_task_ids=excluded, supplemental_attempts=0,
                    incomplete_candidate_groups_use_standard_advantage=True))
            else:
                raise RuntimeError(f"Complete Episode acceptance failed: {len(completed)}/{len(plan['tasks'])}; inspect excluded failures/truncations")
        snapshot = self.d.root(lambda: self.replay.snapshot().to_dict())
        self.artifact(key, snapshot["snapshot_id"])
        return self.replay.load_snapshot(snapshot["snapshot_id"])

    def complete_collection_with_budget(self):
        """沿用已验收的补采事务，全运行共享用户授权次数，不重新计算失败为成功。"""
        from .collection_budget import allocate_collection_budget, update_collection_budget
        from .io import file_sha256
        from tools import complete_stage2_awr_collection as completion
        settings = self.config["awr"]["collection_completion"]
        options = dict(total_limit=settings["max_extra_attempts_total"],
                       authorization=settings["authorization"])
        limit = self.d.root(lambda: allocate_collection_budget(self.root, self.state, **options))
        args = SimpleNamespace(max_extra_attempts=limit, authorization=options["authorization"])
        tool_path = Path(completion.__file__).resolve()
        identity = dict(git_commit=self.state["source_commits"]["genmo"], file=str(tool_path),
                        sha256=file_sha256(tool_path))
        try:
            accepted = completion.complete_collection(self, args, identity)
        finally:
            self.d.root(lambda: update_collection_budget(self.root, self.state, **options))
        if not accepted:
            return None
        key = f"macro{self.state['macro']:03d}_replay"
        return self.replay.load_snapshot(self.state["artifacts"][key])

    def _recover_publications(self, macro):
        tasks = set()
        # 回执前断电：完整原子Episode分片可重建发布回执，不重复采集该task。
        for directory in (self.root/"replay"/"episodes").glob("*"):
            receipt = self.root/"replay"/"publications"/(directory.name+".json")
            if directory.is_dir() and not directory.name.startswith(".") and not receipt.exists():
                tree = read_tree(directory)
                value = dict(episode_id=directory.name,
                    sha256=json.loads((directory/"manifest.json").read_text())["sha256"],
                    task_id=tree["task"]["task_id"], macro=tree["task"]["macro_iteration"],
                    published_unix_ns=directory.stat().st_mtime_ns)
                atomic_json(receipt, value)
        values = [json.loads(p.read_text()) for p in (self.root/"replay"/"publications").glob("*.json")]
        for value in sorted(values, key=lambda v: (v["macro"], v["published_unix_ns"], v["episode_id"])):
            self.replay.register(value)
            if value["macro"] == macro:
                if value["task_id"] in tasks:
                    raise ValueError("Two physical Episodes claim the same completed task")
                tasks.add(value["task_id"])
        return sorted(tasks)

    def run_macro(self, macro):
        self.config["awr"]["actor_version"] = macro
        self.runtime_config["awr"]["actor_version"] = macro
        behavior = copy.deepcopy(self.initial)
        if macro > 0:
            previous = self.checkpoints.load(self.state["artifacts"][f"macro{macro-1:03d}_awr_final"])
            self.actor.load_state_dict(previous["payload"]["actor"])
            behavior = cpu_copy(self.actor.state_dict())
        snapshot = self.collect(macro)
        if snapshot is None:
            return False
        if self.d.root(lambda: stop_requested(self.root)):
            self.publish(status="stopped", phase="collect_complete")
            return False
        episodes = self.d.local(lambda: [self.replay.read(key, training=True) for key in snapshot.episode_ids])
        prefix = f"macro{macro:03d}"
        exploration_report = self.json_artifact(prefix+"_exploration", lambda: exploration(episodes))
        label_key = prefix+"_labels"
        if label_key not in self.state["artifacts"]:
            self.publish(phase="critic_fvi")
            self.actor.load_state_dict(self.initial)
            progress_key = prefix+"_critic_progress"
            resume = (self.checkpoints.load(self.state["artifacts"][progress_key])["payload"]
                      if progress_key in self.state["artifacts"] else None)
            def save_critic(phase, payload):
                self.checkpoint(progress_key, payload, dict(macro=macro, phase=phase))
                if self.d.root(lambda: stop_requested(self.root)):
                    raise GracefulStop("Critic stopped at committed MC/FVI fit boundary")
            labels, critic_report, critics = fit_crossfit(self.actor, episodes, self.folds,
                self.config["awr"]["critic"], self.d, seed=self.config["awr"]["seed"]+macro*1009,
                resume=resume, checkpoint=save_critic, label_config=self.config["awr"].get("advantage"),
                candidate_config=self.config["awr"].get("candidate_sampling"), replay_snapshot=snapshot.snapshot_id,
                candidate_plan=self.json_artifact(prefix+"_tasks", lambda: None)["tasks"])
            self.checkpoint(prefix+"_critics", critics, dict(macro=macro, phase="advantage_labeling"))
            self.json_artifact(prefix+"_critic_report", lambda: critic_report)
            path = self.root/prefix/"labels"
            self.d.root(lambda: labels.save(path))
            self.artifact(label_key, str(path))
            self.json_artifact(prefix+"_advantage_distribution", lambda: labels.statistics)
        labels = AdvantageLabelSet.load(self.state["artifacts"][label_key])
        self.actor.load_state_dict(behavior)
        dataset_options = dict(expected_signal=self.config["awr"].get("advantage"),
            expected_candidates=self.config["awr"].get("candidate_sampling"), expected_snapshot=snapshot.snapshot_id)
        dataset = ReplayDataset(episodes, labels, **dataset_options)
        before_outputs = fixed_outputs(self.policy, dataset.rows)
        self.d.root(lambda: write_tree(self.root/prefix/"fixed_outputs_before", before_outputs))
        bc = self.training_bc()
        trainer = ActorTrainer(self.actor, dataset, bc, self.config["awr"]["actor"], self.d,
                               seed=self.config["awr"]["seed"]+macro)
        coefficient_key = prefix+"_bc_calibration"
        if coefficient_key not in self.state["artifacts"]:
            self.publish(phase="gradient_calibration")
            calibration = trainer.calibrate(self.config["awr"]["actor"]["calibration_batches"],
                                            self.config["awr"]["actor"]["bc_gradient_ratio"])
            if self.config["awr"]["actor"].get("bc_enabled", True):
                calibration = choose_bc_coefficient(calibration, self.config["awr"]["actor"].get("bc_fixed_weight"))
            self.json_artifact(coefficient_key, lambda: calibration)
        coefficient = self.json_artifact(coefficient_key, lambda: None)["coefficient"]
        del trainer, bc
        methods = ["self_distillation", "awr"] if self.config["awr"]["run_ablation"] else ["awr"]
        if self.config["awr"].get("uniform_ablation", False):
            methods.insert(0, "self_distillation_uniform")
        for method in methods:
            method_key = prefix+"_"+method
            final_key = method_key+"_final"
            if final_key not in self.state["artifacts"]:
                self.publish(phase="actor_"+method)
                self.actor.load_state_dict(behavior)
                torch.manual_seed(self.config["awr"]["seed"]+macro+self.d.rank*100003)
                dataset = ReplayDataset(episodes, labels, unit_weights=method.startswith("self_distillation"), **dataset_options)
                bc = self.training_bc()
                training_config = copy.deepcopy(self.config["awr"]["actor"])
                if method == "self_distillation_uniform":
                    training_config["timestep_weighting"] = dict(mode="uniform")
                trainer = ActorTrainer(self.actor, dataset, bc, training_config, self.d,
                    seed=self.config["awr"]["seed"]+macro, bc_weight=coefficient)
                latest_key = method_key+"_progress"
                if latest_key in self.state["artifacts"]:
                    state = self.checkpoints.load(self.state["artifacts"][latest_key])["payload"]
                    trainer.load_state_dict(state, rank_state=state["rank_states"][self.d.rank])
                def save(current, entry):
                    if entry["step"] % self.config["awr"]["actor"]["checkpoint_every_steps"] == 0:
                        self.save_actor(latest_key, current, macro, method)
                    if self.d.rank == 0:
                        with (self.root/prefix/(method+"_steps.jsonl")).open("a") as stream:
                            stream.write(json.dumps(entry)+"\n")
                result = trainer.run(on_step=save, should_stop=lambda: stop_requested(self.root))
                if not result["complete"]:
                    self.save_actor(latest_key, trainer, macro, method)
                    self.publish(status="stopped")
                    return False
                self.save_actor(final_key, trainer, macro, method)
                rank_results = self.d.gather(result)
                self.json_artifact(method_key+"_training", lambda: dict(ranks=rank_results))
                changes = self.d.root(lambda: parameter_changes(self.actor, behavior))
                self.json_artifact(method_key+"_parameter_changes", lambda: changes)
                outputs = fixed_outputs(self.policy, dataset.rows)
                self.d.root(lambda: write_tree(self.root/prefix/(method+"_fixed_outputs_after"), outputs))
                del trainer, bc
            else:
                state = self.checkpoints.load(self.state["artifacts"][final_key])
                self.actor.load_state_dict(state["payload"]["actor"])
            self.bc_validation(method_key)
            current = self.eval_actor(method_key+"_fixed", self.fixed_plan)
            baseline = self.eval_actor("stage1_fixed", self.fixed_plan)
            comparison = paired_comparison(baseline, current, selection_policy=self.config["evaluation"].get("selection_policy", "reward_only"))
            self.json_artifact(method_key+"_comparison", lambda: comparison)
            if self.config["awr"]["final_all_unique"] and macro == self.config["awr"]["max_macro_iterations"]-1:
                self.eval_actor(method_key+"_all", self.full_plan)
        awr = self.json_artifact(prefix+"_awr_comparison", lambda: None)
        if self.config["awr"]["run_ablation"]:
            b = self.eval_actor(prefix+"_self_distillation_fixed", self.fixed_plan)
            c = self.eval_actor(prefix+"_awr_fixed", self.fixed_plan)
            self.json_artifact(prefix+"_awr_vs_self_distillation", lambda: paired_comparison(b, c))
            if self.config["awr"].get("uniform_ablation", False):
                uniform = self.eval_actor(prefix+"_self_distillation_uniform_fixed", self.fixed_plan)
                self.json_artifact(prefix+"_minsnr_vs_uniform", lambda: paired_comparison(uniform, b))
        if self.config["awr"]["final_all_unique"] and macro == self.config["awr"]["max_macro_iterations"]-1:
            baseline = self.eval_actor("stage1_all", self.full_plan)
            current = self.eval_actor(prefix+"_awr_all", self.full_plan)
            self.json_artifact(prefix+"_all_awr_vs_stage1", lambda: paired_comparison(baseline, current))
            if self.config["awr"]["run_ablation"]:
                other = self.eval_actor(prefix+"_self_distillation_all", self.full_plan)
                self.json_artifact(prefix+"_all_awr_vs_self_distillation", lambda: paired_comparison(other, current))
        reference = self.state["artifacts"][prefix+"_awr_final"]
        old_key = self.state.get("best_evaluation_key", "stage1_fixed")
        previous = self.eval_actor(old_key, self.fixed_plan)
        current = self.eval_actor(prefix+"_awr_fixed", self.fixed_plan)
        versus_saved = paired_comparison(previous, current, selection_policy=self.config["evaluation"].get("selection_policy", "reward_only"))
        self.json_artifact(prefix+"_versus_best_saved", lambda: versus_saved)
        self.publish(macro=macro+1, phase="macro_complete",
                     **checkpoint_selection(self.state, reference, awr, versus_saved, prefix+"_awr_fixed"))
        self.json_artifact(prefix+"_replay_cost", self.replay.report)
        return True

    def training_bc(self):
        """关闭BC时完全不创建训练采样器；独立无梯度能力诊断仍可读取真实Stage1标签。"""
        if not self.config["awr"]["actor"].get("bc_enabled", True):
            return None
        return Stage1Supervision(self.runtime_config, self.actor, self.train_config,
                                 source_step=self.load_report["source_global_step"])

    def save_actor(self, name, trainer, macro, method):
        local = dict(bc=trainer.bc.state_dict() if trainer.bc_enabled else None, rng=capture_rng(self.d.device))
        ranks = self.d.gather(local)
        payload = trainer.state_dict() if self.d.rank == 0 else None
        if payload is not None:
            payload["rank_states"] = ranks
        return self.checkpoint(name, payload, dict(macro=macro, phase="actor", method=method,
                                                 epoch=trainer.epoch, cursor=trainer.cursor, step=trainer.steps))

    def manual_evaluate(self, reference=None, *, all_unique=False):
        """单独评价不重开已完成训练；同模型/计划重入时恢复原评价账本。"""
        parent = self.state.get("manual_evaluation_parent") or {
            key: self.state[key] for key in ("status", "phase")}
        self.publish(manual_evaluation_parent=parent, manual_evaluation_status="running")
        try:
            self.prepare()
            reference = self.state.get("latest") if reference is None else reference
            if reference:
                self.actor.load_state_dict(self.checkpoints.load(reference)["payload"]["actor"])
            plan = self.full_plan if all_unique else self.fixed_plan
            name = "manual_"+digest(dict(actor=self.d.assert_models_equal(self.actor),
                                         plan=plan["plan_sha256"]))
            self.publish(manual_evaluation_key=name)
            report = self.eval_actor(name, plan)
            self.publish(manual_evaluation_status="complete")
            return report
        except GracefulStop:
            self.publish(manual_evaluation_status="stopped")
            return None
        except Exception:
            self.publish(manual_evaluation_status="failed")
            raise
        finally:
            self.publish(**parent, manual_evaluation_parent=None)

    def run(self):
        try:
            if self.state["status"] == "complete":
                return 0
            self.prepare()
            for macro in range(self.state["macro"], self.config["awr"]["max_macro_iterations"]):
                if not self.run_macro(macro):
                    return 0
            self.publish(status="complete", phase="complete", finished_unix=time.time())
            return 0
        except GracefulStop:
            self.publish(status="stopped")
            return 0
        finally:
            if self.replay is not None:
                self.replay.close()
