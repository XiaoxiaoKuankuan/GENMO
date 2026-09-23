"""BUMI closed-loop 第4步的小规模训练/验证命令行入口。

该入口完全独立于旧music-only训练脚本，读取已有四库Dataset配置，构建可被后续阶段复用
的Stage1Actor，并运行配置中明确限定的监督步数或采样验证batch数。默认配置为只验证；
指定stage1_train.yaml才执行默认8步的小规模优化，不会隐式启动第5步正式训练。

默认所有运行产物位于系统临时目录，退出后自动删除；显式传入 --output-dir 才保留完整
参数迁移报告、运行配置、验证结果和新接口权重。输出目录必须不存在或为空，旧文件绝不
覆盖。原checkpoint只进行weights-only warm start，旧optimizer和global_step不恢复。
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from gem.closedloop.checkpoint import save_stage1_checkpoint  # noqa: E402
from gem.closedloop.training import (  # noqa: E402
    build_stage1_actor,
    build_stage1_loader,
    build_stage1_losses,
    initialize_stage1_weights,
    load_stage1_config,
    load_stage1_data_config,
    train_stage1_steps,
    validate_stage1_batches,
)


def run(config, output_dir: Path) -> dict:
    """先完整核验配置/资产/权重，再执行指定小规模阶段。"""

    seed = int(config.runtime.seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    data = load_stage1_data_config(config)
    actor = build_stage1_actor(config, data)
    weight_report = initialize_stage1_weights(actor, config)
    (output_dir / "weight_loading_report.json").write_text(
        json.dumps(weight_report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    device = torch.device(str(config.runtime.device))
    actor.to(device)
    losses = build_stage1_losses(actor, config).to(device)
    common = {
        "batch_size": int(config.data_loader.batch_size),
        "num_workers": int(config.data_loader.num_workers),
        "seed": seed,
    }
    if config.mode == "train":
        loader = build_stage1_loader(
            data,
            split="train",
            samples_per_epoch=int(config.train.max_steps) * common["batch_size"],
            **common,
        )
        result = train_stage1_steps(actor, losses, loader, **dict(config.train))
        step = int(result["completed_steps"])
        save_stage1_checkpoint(
            actor,
            output_dir / "stage1_actor.pt",
            config=OmegaConf.to_container(config, resolve=True),
            global_step=step,
            warm_start_report=weight_report,
        )
    elif config.mode == "validate":
        loader = build_stage1_loader(data, split=str(config.validation.split), **common)
        options = dict(config.validation)
        options.pop("split")
        result = validate_stage1_batches(actor, losses, loader, **options)
    else:
        raise ValueError("mode must be train or validate")
    report = {"mode": str(config.mode), "weight_initialization": weight_report, "result": result}
    (output_dir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    OmegaConf.save(config, output_dir / "resolved_config.yaml", resolve=True)
    return report


def main(argv: list[str] | None = None) -> dict:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/closedloop/stage1_validate.yaml")
    parser.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="显式覆盖配置，可重复；例如 model.latent_dim=64",
    )
    parser.add_argument(
        "--output-dir", type=Path, help="保留运行产物的独立空目录；省略则使用并自动清理系统临时目录"
    )
    arguments = parser.parse_args(argv)
    config = load_stage1_config(arguments.config, arguments.set)
    explicit = arguments.output_dir or config.runtime.get("output_dir")
    if explicit:
        output = Path(explicit).expanduser().resolve()
        if output.exists() and any(output.iterdir()):
            raise FileExistsError(f"output directory must be empty: {output}")
        output.mkdir(parents=True, exist_ok=True)
        report = run(config, output)
        report["output_dir"] = str(output)
        report["artifacts_retained"] = True
    else:
        with tempfile.TemporaryDirectory(prefix="genmo-stage1-smoke-") as temporary:
            report = run(config, Path(temporary))
        report["artifacts_retained"] = False
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return report


if __name__ == "__main__":
    main()
