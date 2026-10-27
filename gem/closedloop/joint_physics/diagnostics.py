"""联合训练逐阶段、互斥终态的样本诊断。

每个请求只登记一次最终结果，同时保留经过的阶段及首个拒绝时刻；阶段进入数可以相互
重叠，终态比例的分母始终是本入口请求总数，禁止把多个检查失败相加当成不同样本。
记录来源、时间、实际噪声、动作来源和越界坐标，帮助区分课程未开启与学生执行不合法。
本模块只读诊断，不选择或替换学生动作，也不改变损失和认证范围。
"""

from collections import Counter


def summarize(rows):
    counts = Counter(row["reason"] for row in rows)
    entered = Counter(stage for row in rows for stage in row.get("stages", []))
    return {"count": len(rows), "terminal_counts": dict(counts),
            "terminal_fraction": {k: v/max(1, len(rows)) for k,v in counts.items()},
            "stage_entered": dict(entered), "rows": rows}


def sample_record(batch, index, timestep, snr):
    meta = batch["meta"][index]
    return {"source": meta["sample_id"], "dataset": meta["dataset_id"],
            "request_tick": int(meta["decision_frame"])*20, "noise_level": int(timestep), "snr": float(snr),
            "prediction_origin": "student_denoised_x0", "action_origin": "not_executed",
            "first_reject_tick": None, "stages": ["sampled"]}
