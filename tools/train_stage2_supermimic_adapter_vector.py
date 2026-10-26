"""启动一个独立GPU上的多环境Adapter实验、真实压力测试或数值审计。

一个进程创建一个批量物理仿真，所有模型和优化器独立；不使用DDP、torchrun或跨卡
通信。实际配置、数据划分和输出目录必须显式指定；正式训练先做真实WM/Critic预热
和物理方向验证，模型质量监控不自动停训，只有数值/物理正确性故障限制对应更新。
"""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

if __name__ == '__main__':
    from gem.closedloop.reference_adapter.vector_entry import main
    raise SystemExit(main())
