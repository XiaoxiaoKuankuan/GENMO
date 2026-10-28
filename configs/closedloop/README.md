# 当前配置入口（2026-10-28）

本目录只保留当前分支最后一次实验及基础依赖。旧实验参数、sweep和repair入口已从工作树移除，历史可在Git中追溯。保留入口不代表已取得物理或整歌质量提升。

## 本分支实验

本分支目前没有活动训练，仅保留最后一次实验入口，不能替代Adapter/Joint Physics当前研究基线。

- `stage2_awr_candidate_nstep.yaml`

## 基础依赖

`stage1_*` 保存冻结Stage1模型的训练/数据/契约支持；`stage8_frozen_isaac.yaml` 与 `sim2sim_mujoco.yaml` 是独立评估基础。`configs` 的其他子目录是模型、机器人资产、数据和Hydra组合模块，不按时间删除。JSON格式示例保留用于资产和任务准备。

## 回归与历史

仅为仍维护的算法合同保留少量 `tests/closedloop/fixtures/*_contract.yaml`，禁止用作训练入口；旧实验扫描未迁移到测试目录。针对旧实验参数的断言已退役，通用梯度、恢复、概率与物理协议回归继续维护。历史文档描述当时实验，其旧命令不作为当前使用说明。
