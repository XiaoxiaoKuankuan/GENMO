"""GENMO第二阶段离策略优势加权扩散训练。

模块以完整真实物理Episode作为回放单位，先拟合双Critic并交叉标注，再执行
原Stage1去噪监督与真实BC的混合更新。这里不定义PPO概率、链KL或Flow Matching；
采集、训练及评价的运行身份各自持久化，物理效果必须由独立成对评价判断。
"""

VERSION = "genmo.stage2.awr.v1"
