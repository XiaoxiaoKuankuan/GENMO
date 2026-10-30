"""十帧Adapter导出的动态batch、限幅和外部根姿态开关回归。

使用真实RobotSpec及非零网络输出，分别核对根姿态开启/关闭时ONNX与训练网络的
输出合同。所有ONNX只写入pytest临时目录；测试不加载训练优化器、不运行物理环境，
不把数值一致性当成真实跟踪收益。正式checkpoint另由导出入口进行同样的数值核验。
"""

import pytest
import torch

from supermimic.robots import get_robot_spec
from gem.closedloop.reference_adapter.policy import ReferenceAdapter
from tools.export.export_reference_adapter_onnx import ExportAdapter, export_and_check


@pytest.mark.parametrize("root_orientation", [True, False])
def test_export_dynamic_batch_bounds_and_root_switch(tmp_path, root_orientation):
    torch.manual_seed(42)
    adapter = ReferenceAdapter(get_robot_spec("bumi3_4340"))
    adapter.disable_root_xy = True
    with torch.no_grad():
        # 用小幅学习到的末层权重加大bias覆盖Tanh饱和，避免把任意大权重与
        # 100倍输入共同制造的FP32累加误差混入根通道合同测试。
        adapter.net[-1].weight.normal_(std=.01)
        adapter.net[-1].bias.normal_(std=2.)
    model = ExportAdapter(adapter, root_orientation=root_orientation)
    x = torch.randn(3, adapter.input_spec.dimension)
    with torch.no_grad():
        expected = adapter.forward_packed(x)
        if not root_orientation:
            expected[..., 3:6] = 0
        torch.testing.assert_close(model(x), expected, atol=0, rtol=0)
    report = export_and_check(model, tmp_path/"adapter.onnx")
    assert report["passed"] and len(report["cases"]) == 6
    assert {c["batch"] for c in report["cases"]} == {1, 16}
