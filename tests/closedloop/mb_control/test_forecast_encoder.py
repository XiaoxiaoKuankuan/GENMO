"""独立前瞻条件的零初始化、无效屏蔽和分阶段梯度测试源，本轮未执行。

58维预测误差与相对时间只经零末层进入原latent；invalid样本不受任意未来值影响。
首步只有末投影可更新，更新后前层可获得梯度，避免把零初始化行为误判为断图。
"""

import torch

from gem.closedloop.mb_control.contracts import ForecastContext
from gem.closedloop.mb_control.future_encoder import FutureStateEncoder


def test_zero_projection_invalid_mask_and_first_update():
    torch.manual_seed(42)
    encoder = FutureStateEncoder(32)
    context = ForecastContext(torch.randn(2, 58), torch.tensor([True, False]), 0, 120)
    times = torch.arange(60).double()[None].expand(2, -1) / 30
    output = encoder(context, times)
    assert output.count_nonzero() == 0
    output.sum().backward()
    assert encoder.output.weight.grad.abs().sum() > 0
    assert encoder.input.weight.grad.count_nonzero() == 0
    with torch.no_grad():
        encoder.output.weight -= .01 * encoder.output.weight.grad
    encoder.zero_grad(set_to_none=True)
    output = encoder(context, times)
    assert output[1].count_nonzero() == 0
    output[0].square().mean().backward()
    assert encoder.input.weight.grad.abs().sum() > 0
