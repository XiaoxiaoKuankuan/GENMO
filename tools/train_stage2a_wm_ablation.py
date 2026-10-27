"""启动固定真实数据的A/B/C/D独立WM消融，每组单GPU、150次有界监督。

数据由Stage2-A的wm-data模式真实采集，标签按train/validation隔离。每25次重新以
当前WM执行Newton方向审计；进程不接受Adapter更新、不训练冻结Actor或GENMO。
所有模型版本、优化器、固定标签SHA及失败证据保存在独立目录，禁止覆盖旧实验。
"""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from gem.closedloop.reference_adapter.wm_ablation import main
if __name__=='__main__':
    raise SystemExit(main())
