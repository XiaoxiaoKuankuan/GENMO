"""当前参考时间轴的合法bootstrap和再规划边界。

严格保留最小前缀、GMT十控制步前瞻、插值右点和差分支持。只根据真实参考覆盖
计算合法栅格；旧按环境配额等待、drain和向末条转移追加奖励的实现已经删除。
"""
from .env_adapter import ExecutionIntegrityError

FRAGMENT_CONTRACT='available_reference_fragment_boundary.v3'


def fragment_reference_boundary(snapshot, *, minimum_prefix=12, grid=300, rolling=False):
    """返回支持最小前缀、GMT十控制步前瞻及差分的最后共同决策格点。"""
    source=int(snapshot['source_end_tick'])
    valid=int(snapshot['reference_valid_end_tick'])
    latest_deadline=((min(valid,source-12)-120)//12)*12
    latest_prefix=source-(minimum_prefix-1)*20
    boundary=(min(latest_deadline,latest_prefix)//grid)*grid
    if boundary<int(snapshot['tick']):
        raise ExecutionIntegrityError('Completed GPU fragment has no causal bootstrap reference')
    return boundary
