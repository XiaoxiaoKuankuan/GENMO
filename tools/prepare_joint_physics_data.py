"""原音乐动作的真实Newton附属缓存入口。

只按配置读取既有split，调用冻结SM连续执行并保存独立分片；不创建新配对库、不加载GENMO
权重、不覆盖缓存、不复位拼接失败轨迹。具体合同与预算由joint_physics.entry统一校验。
"""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gem.closedloop.joint_physics.entry import main

if __name__ == "__main__":
    main("prepare")
