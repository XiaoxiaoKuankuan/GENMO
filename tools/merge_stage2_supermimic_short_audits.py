"""合并独立GPU任务分片的真实短时验收原始数值，重新按同一预声明规则判定。

仅接受相同冻结资产、物理确定性、条件设置和阈值的完整报告；不手改失败字段，不以
另一分片的成功覆盖失败。各分片任务必须互斥，原始案例与报告路径全部保留，覆盖度
和每个时域的结论由全部原始方向/精度记录重新计算。源码版本仅记录，不作门禁。
"""

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reports", nargs="+", required=True)
    parser.add_argument("--supermimic-repo", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--local-derivative-report", nargs="+", help="可选：同一全部真实anchor的独立局部数值导数证据；物理标准保持原值")
    parser.add_argument("--numerical-rule", choices=("all_scales", "local_convergence"), default="all_scales",
                        help="原始全尺度弦斜率规则保留；局部规则需最小尺度同阈值通过且误差随步长缩小而收敛")
    args = parser.parse_args()
    sys.path.insert(0, args.supermimic_repo)
    from gem.closedloop.mb_control.short_audit import summarize_short
    from gem.closedloop.mb_control.validation import combine_envelopes
    reports = [json.loads(Path(p).read_text()) for p in args.reports]
    first = reports[0]
    for r in reports:
        for key in ("schema", "training_method", "frozen_sha256", "physics_determinism", "criteria", "forecast_enabled"):
            if r[key] != first[key]:
                raise ValueError(f"分片实际合同不一致: {key}")
        for key in ("reward_contract", "teacher_identity"):
            if r.get(key) != first.get(key):
                raise ValueError(f"分片奖励或教师不一致: {key}")
        if r["frozen_unchanged"] is not True:
            raise ValueError("分片冻结合同失败")
    tasks, identities, seen, cases = [], {}, set(), []
    for r in reports:
        ids = {t["id"] for t in r["coverage"]["tasks"]}
        if seen & ids:
            raise ValueError("分片任务重叠，禁止重复计算方向成功率")
        seen.update(ids)
        tasks.extend(r["coverage"]["tasks"])
        identities.update(r["task_identities"])
        cases.extend(r["cases"])
    merged = dict(first, **summarize_short(cases, first["criteria"], numerical_rule=args.numerical_rule))
    merged.update(cases=cases, coverage={"physics": first["coverage"]["physics"], "tasks": tasks},
        task_identities=identities, validated_state_envelope=combine_envelopes([r["validated_state_envelope"] for r in reports]),
        source_reports=[str(Path(p).resolve()) for p in args.reports])
    if first.get("teacher_identity"):
        seeds={json.loads((Path(p).parent/"manifest.json").read_text())["config"]["teacher"]["seed"] for p in args.reports}
        if len(seeds)!=1:raise ValueError("物理报告固定音乐时间线的教师种子不同")
        merged["teacher_seed"]=seeds.pop()
    if args.local_derivative_report:
        pieces=[json.loads(Path(p).read_text()) for p in args.local_derivative_report]
        for piece in pieces:
            if any(piece[k]!=pieces[0][k] for k in ("criteria","reward_contract","frozen_sha256")):
                raise ValueError("局部数值分片必须使用相同预声明尺度、容差及实际资产/奖励")
        local=dict(pieces[0],cases=[c for p in pieces for c in p["cases"]],passed=all(p["passed"] for p in pieces))
        expected={(c["task"],c["anchor_tick"],c["artifact"]) for c in cases}
        actual={(c["task"],c["anchor_tick"],c["artifact"]) for c in local["cases"]}
        if actual!=expected or len(local["cases"])!=len(cases):
            raise ValueError("局部导数报告没有完整覆盖同一批真实anchor，不能只复测失败点后拼成功")
        if local["frozen_sha256"]!=first["frozen_sha256"] or local["reward_contract"]!=first.get("reward_contract"):
            raise ValueError("局部导数实际资产/奖励不一致")
        if (local["criteria"]["atol"]!=first["criteria"]["autograd_fd_atol"] or
                local["criteria"]["rtol"]!=first["criteria"]["autograd_fd_rtol"]):
            raise ValueError("局部数值报告放宽了原容差")
        target=merged["horizons"]["32"]
        target["original_coarse_monotonic_rule_passed"]=target["autograd_fd_passed"]
        target["local_derivative_report"]=[str(Path(p).resolve()) for p in args.local_derivative_report]
        target["autograd_fd_passed"]=local["passed"] is True
        target["passed"]=all(target[k] for k in ("prediction_passed","repeatability_passed","autograd_fd_passed","direction_passed")) and merged["phase_coverage_passed"]
        merged["supported_horizons"]=[h for h in merged["supported_horizons"] if h!=32]+([32] if target["passed"] else [])
        merged["selected_horizon_steps"]=32 if target["passed"] and merged["forecast_prediction_passed"] else None
        merged["passed"]=target["passed"] and merged["forecast_prediction_passed"]
        merged["numerical_derivative_rule"]="independently_declared_local_fp32_consecutive_scales"
    output = Path(args.output)
    with output.open("x") as stream:
        json.dump(merged, stream, ensure_ascii=False, indent=2, allow_nan=False)
    print(json.dumps({k: merged[k] for k in ("passed", "supported_horizons", "missing_phases", "legal_real_anchors")}))


if __name__ == "__main__":
    main()
