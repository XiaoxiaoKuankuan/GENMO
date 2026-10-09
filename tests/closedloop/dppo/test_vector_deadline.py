"""共享GPU场景剩余参考的严格deadline协商测试，只在服务器1运行。

覆盖启动预算足够时不改变请求、轮末参考较短时收紧deadline、30/50Hz边界和
不可生成时拒绝。以原reserve的整数公式独立验算最大边界，不缩短实测延迟，
不修改前缀/差分支持规则，也不把已过期参考补零成可用输入。
"""
import pytest
from gem.closedloop.dppo.vector_environment import bounded_reference_deadline
from gem.closedloop.dppo.env_adapter import ExecutionIntegrityError


def test_unchanged_when_reference_covers_calibration():
    d,r=bounded_reference_deadline(dict(tick=600,source_end_tick=2980,reference_valid_end_tick=2964),1440)
    assert d==1440 and not r['reduced']


def test_continuous_drain_tightens_deadline_without_weakening_prefix():
    d,r=bounded_reference_deadline(dict(tick=3000,source_end_tick=3880,reference_valid_end_tick=3864),3852)
    assert d==3744 and r['reduced']
    protected=d+120
    assert protected<=3864
    assert ((protected+12+19)//20)*20<=3880
    assert d+12+120>3864


@pytest.mark.parametrize('end,valid',[(3210,3192),(3110,3084)])
def test_insufficient_window_is_not_fabricated(end,valid):
    with pytest.raises(ExecutionIntegrityError):
        bounded_reference_deadline(dict(tick=3000,source_end_tick=end,reference_valid_end_tick=valid),3852)
