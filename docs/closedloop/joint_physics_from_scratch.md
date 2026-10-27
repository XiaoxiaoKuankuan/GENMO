# 从零音乐扩散与可微物理执行约束

本方法独立于旧Stage2：静态构造随机GENMO，原60帧监督与同一样本去噪x0的执行损失
共用一个优化器、一个run。没有GENMO权重初始化、生成教师、Critic或终点价值。

2026-10-27实际基线：`4a005d14d7e523d43959c21d3a6364de79653462`；
只读SM bridge：`d9164b0e15934806ff8d750ff70911949c7f5171`。
独立工作树：`/home/weili/bumi-closedloop-worktrees/joint-physics/GENMO`；
分支：`feature/joint-diffusion-physics-from-scratch`。原共享分支与运行进程保留。

实现与验收正在分阶段补齐；尚无方法收益结论。真实资产、Newton方向、有效学生更新、
重建、自由采样和完整连续执行会分别列出，未通过项不会用接口测试代替。
