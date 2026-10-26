"""正式多卡流水线的设备布局与可选资源预算，不设置舞蹈质量停训条件。

主GPU同时承担一个真实Newton采集环境和唯一GENMO AdamW优化器，其余GPU各运行
独立的真实物理采集器；所有环境使用同一Actor版本，回传合法anchor后汇总梯度。
这属于多卡真实采集、集中优化，不是DDP。评价只能在所有采集请求已经返回、Actor
未更新的同步边界复用这些GPU。本模块只检查重复设备和显式复用约定，不评判动作
好坏。资源上限为null或非正数时禁用；连续候选拒绝不属于资源预算或停止理由。
"""


def physical_gpu_layout(training_gpu, worker_gpus, evaluation_gpus, *, share_evaluation=False):
    collection = [training_gpu, *worker_gpus]
    if len(collection) != len(set(collection)) or not 1 <= len(collection) <= 8:
        raise ValueError("真实采集需要1至8张互不重复的GPU")
    if not evaluation_gpus or len(evaluation_gpus) != len(set(evaluation_gpus)) or len(evaluation_gpus) > 8:
        raise ValueError("评价需要1至8张互不重复的GPU")
    if any(not isinstance(gpu, int) or gpu < 0 for gpu in collection + list(evaluation_gpus)):
        raise ValueError("GPU编号必须是非负整数")
    overlap = sorted(set(collection).intersection(evaluation_gpus))
    if overlap and not share_evaluation:
        raise ValueError("评价复用采集GPU时必须显式启用同步边界复用")
    return {"collection_gpus": collection, "optimizer_gpu": training_gpu,
            "evaluation_gpus": list(evaluation_gpus), "shared_evaluation_gpus": overlap,
            "allocated_gpus": sorted(set(collection + list(evaluation_gpus))),
            "gradient_distribution": "central_optimizer_real_parallel_collection",
            "evaluation_runs_while_collection_paused": bool(overlap)}


def resource_stop_reason(settings, control_steps, elapsed_seconds):
    steps = settings.get("max_control_steps")
    hours = settings.get("max_wall_hours")
    if steps is not None and steps > 0 and control_steps >= steps:
        return "explicit_control_step_budget"
    if hours is not None and hours > 0 and elapsed_seconds >= hours * 3600:
        return "explicit_wall_time_budget"
    return None
