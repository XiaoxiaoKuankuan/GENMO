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
    merged = dict(first, **summarize_short(cases, first["criteria"]))
    merged.update(cases=cases, coverage={"physics": first["coverage"]["physics"], "tasks": tasks},
        task_identities=identities, validated_state_envelope=combine_envelopes([r["validated_state_envelope"] for r in reports]),
        source_reports=[str(Path(p).resolve()) for p in args.reports])
    output = Path(args.output)
    with output.open("x") as stream:
        json.dump(merged, stream, ensure_ascii=False, indent=2, allow_nan=False)
    print(json.dumps({k: merged[k] for k in ("passed", "supported_horizons", "missing_phases", "legal_real_anchors")}))


if __name__ == "__main__":
    main()
