"""服务器2正式启动器：实际验收与资源通过后创建独立tmux训练。

读取已标定配置和明确的训练/验证/固定评价集合，记录两个源码提交但不作相等门禁。
真实WM、梯度、全曲基线未通过时写preflight_blocked.json并退出，绝不把空会话算启动。
成功则固定配置、命令、GPU UUID与PID；--resume-run使用最近完整checkpoint并从新
物理episode采集。允许显式恢复旧版本因质量评价停止的完整检查点；不自动重启，
不占用已有计算进程的GPU。质量指标仅记录，数值故障仍保留原诊断。
"""

import argparse
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import shlex
import shutil
import socket
import subprocess
import sys

from omegaconf import OmegaConf


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024*1024), b""):
            value.update(block)
    return value.hexdigest()


def run(command):
    return subprocess.check_output(command, text=True).strip()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--train-tasks", required=True)
    parser.add_argument("--validation-tasks", required=True)
    parser.add_argument("--evaluation-tasks", required=True)
    parser.add_argument("--gradient-report", required=True)
    parser.add_argument("--baseline-report", required=True)
    parser.add_argument("--training-gpu", type=int, default=0)
    parser.add_argument("--evaluation-gpu", type=int, default=1)
    parser.add_argument("--max-actor-updates", type=int, default=5000)
    parser.add_argument("--resume-run")
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[1]
    os.chdir(repo)
    cfg = OmegaConf.to_container(OmegaConf.load(args.config), resolve=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    output = Path(args.resume_run).resolve() if args.resume_run else Path(args.output_root).resolve()/f"formal_{timestamp}"
    output.mkdir(parents=True, exist_ok=bool(args.resume_run))
    if args.resume_run:
        cfg = OmegaConf.to_container(OmegaConf.load(output/"config.yaml"), resolve=True)
    else:
        cfg["tasks"] = str(Path(args.train_tasks).resolve())
        cfg["formal"] = dict(cfg.get("formal", {}), critic_validation_tasks=str(Path(args.validation_tasks).resolve()),
                             evaluation_tasks=str(Path(args.evaluation_tasks).resolve()), evaluation_gpu=args.evaluation_gpu,
                             preflight={"gradient_report":str(Path(args.gradient_report).resolve()),
                                        "baseline_report":str(Path(args.baseline_report).resolve())})
        cfg["training"]["require_critic_quality"] = True
        OmegaConf.save(OmegaConf.create(cfg), output/"config.yaml")
    failures, checks = [], {}
    def check(name, predicate):
        try:
            result = predicate()
            checks[name] = result
            if result is not True:
                failures.append(name)
        except Exception as error:
            checks[name] = repr(error)
            failures.append(name)
    check("genmo_branch", lambda: run(["git","branch","--show-current"]) == "feature/stage2-supermimic-worldmodel")
    check("supermimic_branch", lambda: run(["git","-C",cfg["supermimic"]["repo"],"branch","--show-current"]) == "feature/genmo-worldmodel-bridge")
    check("clean_code", lambda: not run(["git","status","--porcelain"]))
    check("tmux", lambda: shutil.which("tmux") is not None)
    check("disk_at_least_200GiB", lambda: shutil.disk_usage(output).free >= 200*1024**3)
    evaluation_gpus = cfg["formal"].get("evaluation_gpus", [args.evaluation_gpu])
    worker_gpus = cfg.get("execution", {}).get("physical_worker_gpus", [])
    sys.path.insert(0, str(repo))
    from gem.closedloop.mb_control.parallel_layout import physical_gpu_layout
    layout = {}
    def layout_check():
        layout.update(physical_gpu_layout(args.training_gpu, worker_gpus, evaluation_gpus,
            share_evaluation=cfg["formal"].get("reuse_collection_gpus_for_evaluation", False)))
        return args.evaluation_gpu == evaluation_gpus[0]
    check("physical_gpu_layout", layout_check)
    check("isaaclab_path", lambda: Path(os.environ.get("ISAACLAB_PATH", "/nonexistent")).is_dir())
    gpu_info = []
    try:
        lines = run(["nvidia-smi","--query-gpu=index,uuid,memory.used,memory.total","--format=csv,noheader,nounits"]).splitlines()
        gpu_info = [list(map(str.strip,line.split(","))) for line in lines]
        used = run(["nvidia-smi","--query-compute-apps=gpu_uuid,pid","--format=csv,noheader"]).splitlines()
        occupied = {line.split(",")[0].strip() for line in used if "," in line}
        for index in sorted(set([args.training_gpu,*evaluation_gpus,*worker_gpus])):
            check(f"gpu_{index}_free", lambda index=index: any(int(g[0]) == index and g[1] not in occupied and int(g[2]) < 1024 for g in gpu_info))
    except Exception as error:
        failures.append("gpu_inventory")
        checks["gpu_inventory"] = repr(error)
    for group, keys in (("stage1",("checkpoint","stats","kinematics","data_config")),("supermimic",("checkpoint",))):
        for key in keys:
            check(f"asset_{group}_{key}", lambda group=group,key=key: Path(cfg[group][key]).is_file())
    check("task_manifests", lambda: all(Path(p).is_file() for p in (args.train_tasks,args.validation_tasks,args.evaluation_tasks)))
    check("formal_settings", lambda: all(k in cfg["formal"] for k in ("critic_warmup", "max_control_steps", "max_wall_hours",
        "maximum_invalid_streak", "checkpoint_every", "evaluate_every", "degradation")))
    def gradients():
        report = json.loads(Path(args.gradient_report).read_text())
        if cfg.get("teacher", {}).get("enabled"):
            return (report.get("passed") is True and report.get("reward_contract") == cfg["rewards"] and
                    report.get("horizon") == cfg["imagination"]["horizon_steps"] and
                    report.get("teacher_storage_independent") is True and
                    report.get("teacher_parameters_statistics_unchanged") is True and
                    report.get("teacher_preserves_student_rng") is True and
                    report.get("fixed_timeline_unchanged_after_actor_update") is True and
                    report.get("known_prefix_prior_gradient_max_abs") == 0 and
                    report.get("zero_forecast_stage1_max_abs", 1) <= 1e-5 and
                    report.get("frozen_parameters_and_buffers_unchanged") is True and
                    report["actual_update"]["accepted_execution_feedback_update"] is True)
        return (report["execution_paths"]["passed"] is True and report["zero_forecast_stage1_max_abs"] <= 1e-5 and
                report["checkpoint_gradient_max_abs"] <= 1e-5 and report["frozen_parameters_and_buffers_unchanged"] is True)
    check("execution_gradients", gradients)
    def baseline():
        report = json.loads(Path(args.baseline_report).read_text())
        return (report["mode"] == "stage1" and report["episodes"] >= 8 and report["completed"] == report["episodes"] and
                all(r["commits"] >= 2 for r in report["physical_results"]))
    check("complete_physical_baseline", baseline)
    def world_model():
        report = json.loads(Path(cfg["imagination"]["physical_validation_report"]).read_text())
        if cfg["imagination"].get("mode") == "physical_anchor_short_bptt":
            sys.path.insert(0, str(repo))
            sys.path.insert(0, cfg["supermimic"]["repo"])
            from gem.closedloop.mb_control.validation import select_short_horizon
            select_short_horizon(cfg, report)
            return report["frozen_sha256"] == digest(cfg["supermimic"]["checkpoint"]) and bool(report.get("validated_state_envelope"))
        needed = cfg["imagination"]["validated_model_steps"]
        return (all(report.get(k) is True for k in ("prediction_passed","physical_direction_passed","known_reference_passed","reset_replay_repeatability_passed")) and
                isinstance(needed,int) and 0 < needed <= report["supported_control_steps"] and
                report["frozen_sha256"] == digest(cfg["supermimic"]["checkpoint"]) and bool(report.get("validated_state_envelope")))
    check("world_model_physics_and_direction", world_model)
    def parallel_collection():
        report = json.loads(Path(cfg["formal"]["parallel_collection_report"]).read_text())
        return (report["passed"] is True and report["physical_collectors"] == 1+len(worker_gpus) and
                report["frozen_sha256"] == digest(cfg["supermimic"]["checkpoint"]) and
                report["contract"] == {k: cfg[k] for k in ("model", "clock", "rewards")} and
                report["physics_determinism"] == cfg["execution"].get("physics_determinism") and
                report["horizon_steps"] == cfg["imagination"]["horizon_steps"])
    if worker_gpus:
        check("real_parallel_collection", parallel_collection)
    def latency():
        profile = json.loads(Path(cfg["latency_profile"]).read_text())
        return profile.get("forecast_enabled") is True and len(profile.get("latency_ticks", [])) >= 20 and profile["frozen_sha256"] == digest(cfg["supermimic"]["checkpoint"])
    check("measured_full_path_latency", latency)
    if args.resume_run:
        status_path = output/"status.json"
        if status_path.exists():
            status = json.loads(status_path.read_text())
            check("previous_failure_allows_explicit_resume", lambda: status.get("status") != "failed" or
                  "独立真实物理连续评价退化" in status.get("error", ""))
        check("complete_latest_exists", lambda: (output/"checkpoints/latest.pt").is_file())
        stale = output/"training.lock"
        if stale.exists():
            pid = int(stale.read_text())
            check("previous_process_gone", lambda: not Path(f"/proc/{pid}").exists())
            if checks.get("previous_process_gone") is True:
                stale.unlink()
    evidence = {"host":socket.gethostname(),"checks":checks,"failures":failures,"gpu_inventory":gpu_info,
                "actual_actor_updates":0 if not args.resume_run else None,"source_hash_gate":False,
                "physical_gpu_layout": layout,
                "genmo_commit":run(["git","rev-parse","HEAD"]),"supermimic_commit":run(["git","-C",cfg["supermimic"]["repo"],"rev-parse","HEAD"])}
    (output/(f"preflight_{timestamp}.json" if args.resume_run else "preflight.json")).write_text(json.dumps(evidence,ensure_ascii=False,indent=2))
    if failures:
        (output/f"preflight_blocked_{timestamp}.json").write_text(json.dumps(evidence,ensure_ascii=False,indent=2))
        print(json.dumps({"status":"blocked_before_training","output":str(output),"failed_checks":failures},ensure_ascii=False))
        return 2
    OmegaConf.save(OmegaConf.create(cfg), output/"config.yaml")
    command = [sys.executable,"-u","tools/train_stage2_supermimic.py","--config",str(output/"config.yaml"),
               "--run-mode","formal","--mode","full","--max-actor-updates",str(args.max_actor_updates),"--output",str(output)]
    if args.resume_run:
        command.append("--resume-latest")
    session = f"genmo-sm-{timestamp}"
    (output/"logs").mkdir(exist_ok=True)
    # 已运行的tmux服务器不会继承新客户端的环境，必须把已验收的物理依赖和
    # 编译缓存路径显式交给训练进程；只传白名单，不把凭据等整个环境写入日志。
    runtime_environment = {key: os.environ[key] for key in (
        "ISAACLAB_PATH", "TORCHINDUCTOR_CACHE_DIR", "TRITON_CACHE_DIR",
        "TORCHINDUCTOR_COMPILE_THREADS") if key in os.environ}
    invocation = ["env", f"CUDA_VISIBLE_DEVICES={args.training_gpu}", "PYTHONDONTWRITEBYTECODE=1",
                  "OMP_NUM_THREADS=4", "MKL_NUM_THREADS=4",
                  *(f"{key}={value}" for key, value in runtime_environment.items()), *command]
    shell_command = "exec " + shlex.join(invocation) + " >> " + shlex.quote(str(output/"logs/train.log")) + " 2>&1"
    subprocess.run(["tmux","new-session","-d","-s",session,"-c",str(repo),shell_command],check=True)
    pid = int(run(["tmux","display-message","-p","-t",session,"#{pane_pid}"]))
    (output/f"launch_{timestamp}.json").write_text(json.dumps({**evidence,"command":command,"session":session,"pid":pid,
        "status":"launched_pending_effective_update_verification","training_gpu":args.training_gpu,"evaluation_gpu":args.evaluation_gpu,
        "runtime_environment":runtime_environment},ensure_ascii=False,indent=2))
    print(json.dumps({"output":str(output),"session":session,"pid":pid,"status":"launched_not_yet_verified"},ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
