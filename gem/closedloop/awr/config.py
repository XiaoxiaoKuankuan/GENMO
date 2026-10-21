"""AWR专用配置验证、资产身份及公共物理接口适配。

用户编辑stage2_awr.yaml；优化器/宏迭代完全由awr字段控制，不经过PPO校验或迭代推导。
旧物理服务的stage9/stage10键只在边界集中生成，内容仅为奖励、时钟、BC数据和60帧
参考合同。预检核对独立工作树、实际文件SHA及GPU占用，不修改环境或安装指向。
"""
from __future__ import annotations

import copy
import json
from pathlib import Path
import subprocess
import yaml
from .io import digest, file_sha256


def git(repo, *args):
    return subprocess.check_output(["git", "-C", str(repo), *args], text=True).strip()


def configuration_assets(config, *, expected=None):
    """自动绑定控制配置内容，避免路径相同但文件改变后仍复用物理身份。"""
    actual = {key: file_sha256(config["paths"][key])
              for key in ("gmt_agent_config", "isaac_contract")}
    if expected is not None and actual != expected:
        raise ValueError("Frozen GMT control/Isaac configuration file contents changed")
    return actual


def validate_measured_deployment_profile(profile, config):
    """复用真实已测时钟时核验原采样器、冻结资产和测量文件，避免把时钟变化算成学习。

    这里只校验及复制，不重新测时、不修改延迟或前缀预算。允许代码文档提交变化，
    但测量数组必须与SHA绑定的原证据完全相同；运行仍记录自己的完整物理身份。
    """
    if profile is None:
        return None
    from gem.closedloop.common.deployment_clock import DeploymentClock
    DeploymentClock(profile)
    sampler = dict(schema="genmo.awr.stage1_ddim.v1", steps=20, eta=0., guidance_scale=2.5,
                   noise="initial_gaussian", actor="same_network_all_steps")
    if profile.get("sampler") != sampler or profile.get("measured_assets") != config["assets"]:
        raise ValueError("Deployment profile sampler/frozen assets differ")
    samples = []
    for reference in profile["provenance"]:
        if file_sha256(reference["path"]) != reference["sha256"]:
            raise ValueError("Deployment measurement evidence SHA differs")
        evidence = json.loads(Path(reference["path"]).read_text())
        if (evidence.get("schema") != "genmo.awr.deployment_calibration.v1"
                or evidence.get("sampler") != sampler or evidence.get("assets") != config["assets"]):
            raise ValueError("Deployment measurement provenance differs")
        samples.extend(evidence["samples_seconds"])
    if samples != profile["samples_seconds"]:
        raise ValueError("Deployment delay samples differ from the measured evidence")
    return copy.deepcopy(profile)


def load_config(path, *, profile=None):
    c = yaml.safe_load(Path(path).read_text())
    if c.get("version") != "genmo.stage2.awr.v1":
        raise ValueError("An independent AWR config is required")
    if profile is not None:
        if profile not in ("smoke", "pilot"):
            raise ValueError("Use the approved smoke or pilot profile")
        c["awr"]["profile"] = profile
    if c["awr"]["profile"] == "smoke":
        s = c["awr"]["smoke"]
        c["awr"]["initial_episodes"] = s["episodes"]
        c["awr"]["replay_capacity"] = s["episodes"]
        c["awr"]["final_all_unique"] = False
        c["runtime"].update(num_envs=s["envs_per_rank"], generation_batch=s["envs_per_rank"])
    if c["model"] != dict(ddim_steps=20, guidance_scale=2.5, history_steps=50):
        raise ValueError("AWR v1 requires original Stage1 DDIM20/CFG2.5/history50")
    from gem.closedloop.common.rewards import NO_ACTIVITY_REWARD_VERSION
    if c["reward"]["version"] != NO_ACTIVITY_REWARD_VERSION:
        raise ValueError("New AWR runs require the reward without activity multiplication")
    if c["awr"]["actor"].get("denoising_guidance_scale") != c["model"]["guidance_scale"]:
        raise ValueError("Replay training CFG must match the deployed sampler")
    if c["awr"]["actor"].get("replay_coefficient", 1.) != 1.:
        raise ValueError("AWR must retain its real Replay objective; BC-only is diagnostic only")
    if not c["awr"].get("retention_gate") or not c["awr"].get("run_ablation"):
        raise ValueError("Self-distillation retention must pass before AWR updates")
    if c["runtime"]["allow_shared_gpu"]:
        raise ValueError("AWR cannot take occupied GPUs")
    if c.get("diagnostics", {}).get("reward_substeps") is not True:
        raise ValueError("AWR requires original GMT contact/tracking substep diagnostics")
    if c["awr"]["max_macro_iterations"] < 1 or c["awr"]["initial_episodes"] > c["awr"]["replay_capacity"]:
        raise ValueError("Invalid macro/FIFO budget")
    completion = c["awr"].get("collection_completion")
    if completion is not None:
        limit = completion.get("max_extra_attempts_total")
        if (type(limit) is not int or limit < 1
                or not isinstance(completion.get("authorization"), str)
                or not completion["authorization"].strip()):
            raise ValueError("Explicit run-wide supplemental attempt budget/authorization required")
    fixed = dict(clock_hz=600, control_ticks=12, decision_ticks=300, physics_ticks=3,
                 warmup_ticks=600, min_prefix_frames=12, lookahead_control_steps=10)
    if any(c["timing"][k] != v for k, v in fixed.items()):
        raise ValueError("AWR must preserve physical timing")
    validate_measured_deployment_profile(c["timing"]["deployment_profile"], c)
    return c


def physical_config(config, *, run_id, rank=0, device="cuda:0"):
    c = copy.deepcopy(config)
    c["runtime"].update(genmo_device=str(device), rank=rank, record_compact_execution_provenance=True)
    c["stage9"] = dict(run_id=run_id, seed=c["awr"]["seed"]+rank*100003,
        reward=copy.deepcopy(c["reward"]), bc_data_root=c["paths"]["data_root"],
        bc_prefix=c["awr"]["bc_prefix"], bc_batch=c["awr"]["actor"]["bc_microbatch"],
        latency_budget_s=c["timing"]["calibration_reservation_seconds"], execution_mode="latency")
    c["stage10"] = dict(generation_horizon=dict(frames=60, fps=30), seed=c["awr"]["seed"],
        dataset=c["dataset"], physical_episode=dict(safety_limit_seconds=None,
        termination="music_end_or_physical_failure"), performance=dict(bc_batch_forward=True))
    return c


def preflight(config, *, check_gpus=True):
    report = dict(schema="genmo.awr.preflight.v1", assets={}, repositories={})
    validate_measured_deployment_profile(config["timing"]["deployment_profile"], config)
    report["configuration_assets"] = configuration_assets(config)
    for name in ("genmo", "gmt"):
        root = Path(config["paths"][name+"_repo"]).resolve()
        branch = git(root, "branch", "--show-current")
        if branch != "feature/stage2-awr":
            raise ValueError(f"{root}: AWR requires its independent feature/stage2-awr worktree")
        status = git(root, "status", "--porcelain")
        if status:
            raise ValueError(f"{root}: commit the reviewed AWR source before physical execution")
        report["repositories"][name] = dict(path=str(root), branch=branch, commit=git(root, "rev-parse", "HEAD"))
    if Path(__file__).resolve().parents[3] != Path(config["paths"]["genmo_repo"]).resolve():
        raise ValueError("Python imported a different GENMO worktree")
    for name, expected in config["assets"].items():
        actual = file_sha256(config["paths"][name])
        if actual != expected:
            raise ValueError(f"Frozen asset fingerprint differs: {name}")
        report["assets"][name] = actual
    if check_gpus:
        apps = subprocess.check_output(["nvidia-smi", "--query-compute-apps=pid,gpu_uuid",
                                        "--format=csv,noheader"], text=True).strip()
        if apps:
            raise RuntimeError("Server-2 GPU resources are occupied; AWR will not preempt:\n"+apps)
        report["gpu_inventory"] = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index,uuid,memory.total", "--format=csv,noheader"], text=True).strip()
    report["config_sha256"] = digest(config)
    return report
