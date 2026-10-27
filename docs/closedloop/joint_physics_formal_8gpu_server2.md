# Server-2 正式八卡联合训练（2026-10-27）

用户最新补充预算为 **10000 次全局 optimizer 更新**；不是每卡10000后相加，覆盖旧600000示例。
配置为 `configs/closedloop/joint_physics_formal_8gpu_server2.yaml`。原1024维16层Transformer随机初始化，
完整4287源、60帧、1000步x0监督不变；只有一个GENMO AdamW，冻结SM/WM及normalizer。

- 数据batch每卡32，全局256；LR峰值1e-5，1000步warmup，梯度裁剪1，FP32。
- 第1000步起每4步抽取每卡1个真实离线请求；噪声独立取50/100/250，lambda由0渐增到0.01，warmup2000。
- 取消历史25%有效率启动门槛。历史WM/动作数值包络、运动组合仅统计；资产、参考时间和实际物理合法性仍检查。
- 目标H32。后续WM非有限、严重非法或SM终止时，保留此前至少8步的可微前缀，按有效折扣质量归一化。
  日志分别记录8至32实际步数；短前缀不算完整H32。参考不连续、无新参考影响等结构错误整窗拒绝。
- 严重状态界限显式写入配置：根高度0.05至3米，关节超机械限制0.5rad以上，速度绝对值超过20m/s、
  角速度30rad/s、关节速度60rad/s，或旋转非法。它们不是此前认证集最大/最小值。
  生成参考硬限位超限只在训练中统计；真实Newton评价仍执行原严格硬限位和SM终止，不接受训练策略。
- 每4步增加每卡1个合法FutureContext监督，权重0.1、独立完整1000步噪声，同一Stage1损失和优化器。
  不运行WM前瞻或Newton。每500步检查两个sidecar目录的index.json增量；新增不可变分片并原子发布index即可热扩展。
  不能改写已有源、分片、split或资产。checkpoint保存当前索引快照，恢复先还原旧抽样游标，再在刷新边界引入增量。
- 每100步控制台监控；metrics.jsonl逐步记录。physics_gradient_norm明确指学生x0处执行梯度，
  不冒充全部参数梯度范数；有效物理更新要求该梯度有限非零、实际权重大于0且完成共同backward/AdamW。
- 每500步保存完整v3 checkpoint（8rank独立数据/物理/Future游标、噪声、CPU及当前rank CUDA RNG）。
  只支持本算法完整恢复；旧v1/v2保留评价读取。每10000步仅数值去噪及完整DDIM20诊断，不做训练期Newton评价。

启动命令（在Server-2独立工作树中，输出必须为全新目录）：

```bash
export JOINT_PYTHON=/data0/user/liwei/envs/genmo_sm_runtime/bin/python
export JOINT_GPUS=8 CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
bash scripts/train_joint_physics.sh configs/closedloop/joint_physics_formal_8gpu_server2.yaml \
  --max-steps 10000 --output /data0/user/liwei/GENMO_outputs/joint_physics_formal_8gpu_20261027/run
```

完整恢复使用相同配置和8卡布局，加 `--resume /原输出/joint_00000500.pt`，输出指定新目录，
`--max-steps 10000`仍表示绝对全局步数上限。正式任务仅启动一次；恢复前先确认原任务已退出。
训练暂时无合法物理样本仍继续监督。正式启动不宣称自由生成、真实整曲或动力学一致性已经改善。
