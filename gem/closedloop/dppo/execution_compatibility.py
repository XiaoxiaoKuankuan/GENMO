"""第二阶段恢复和比较的实际兼容性检查，源码版本仅作追溯记录。

历史checkpoint、MC成果、固定评估和最佳模型指针都可能保存完整源码哈希表。
这些记录描述文件保存时的代码，不能作为恢复训练或比较模型的通行证。本模块
统一在比较副本中排除源码记录；不改写输入、旧模型或旧报告，也不需要版本迁移。

仅排除明确的源码记录字段source_files、behavior_sources、physical_sources。
模型资产、Gaussian核、奖励、观测/物理配置、执行时钟、任务及价值目标等实际
配置仍参与比较。模型shape和优化器参数对应关系由加载器另行验证；本模块不
替代真实采样概率重算、冻结GMT验证或数值保护，也不把源码不同自动解释为收益。
"""
from __future__ import annotations


SOURCE_RECORDS = frozenset(('source_files', 'behavior_sources', 'physical_sources'))


def without_source_records(value):
    """生成不含源码记录的比较副本，兼容过去存在或缺少源码表的文件。"""
    if isinstance(value, dict):
        return {key: without_source_records(item) for key, item in value.items()
                if key not in SOURCE_RECORDS}
    if isinstance(value, list):
        return [without_source_records(item) for item in value]
    if isinstance(value, tuple):
        return tuple(without_source_records(item) for item in value)
    return value


def same_execution(left, right):
    return without_source_records(left) == without_source_records(right)


def contains_execution(recorded, required):
    """评估加载只比较调用方需要的配置；源码记录即使缺失也不阻塞。"""
    old, new = without_source_records(recorded), without_source_records(required)
    return (isinstance(old, dict) and isinstance(new, dict)
            and all(key in old and old[key] == value for key, value in new.items()))
