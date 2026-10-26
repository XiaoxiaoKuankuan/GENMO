"""在服务器真实资产/物理接管状态上验证冻结教师与完整DDIM梯度。

输入独立教师配置，输出原始真实请求、损失分量、参数梯度及冻结身份检查；一次
诊断更新只存在于该进程，不恢复正式训练、不宣称物理质量改善。
"""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from gem.closedloop.mb_control.entry import main
if __name__ == "__main__":
    main("teacher_audit")
