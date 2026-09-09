# 用户长期 Git 同步授权

本文件记录用户于 2026-09-02 对 `/home/weili/GENMO` 仓库给出的持续授权，供后续 Codex
任务在保护既有工作的前提下执行常规代码同步。它补充 `AGENTS.md` 的仓库工作约定，不替代
测试、记录、分支检查和服务器只做快进同步等要求。

- Codex 为完成用户请求而修改代码、配置、测试或文档后，可以直接在当前工作分支执行
  `git add`、使用中文详细说明的 `git commit`、`git push`，无需再次询问。
- 该授权也覆盖任务开始时工作树中已经存在、经检查确认属于用户要求且需要一起交付的有效
  改动。Codex 必须保留这些改动并在提交说明和 `记录文本.md` 中如实记录，不得静默丢弃。
- 推送成功后，可以在对应训练服务器仓库执行 `git pull --ff-only`，无需再次询问；服务器
  仍不得形成长期未提交的源码修改。
- 长期授权不包含 `git push --force`、`git reset`、`git rebase`、历史改写、删除分支或标签、
  覆盖远端等高风险操作。此类操作仍需用户逐次明确授权。
- 每次同步前必须检查当前分支、HEAD、工作树和远端差异；发生冲突或发现来源不明的改动时，
  必须先保护双方有效历史，不得用覆盖式命令解决。

## 服务器2八卡训练固定环境

用户于 2026-09-03 明确要求：服务器2启动任何 GENMO 八卡 DDP 训练或八卡训练 smoke 前，
必须同时设置以下四个环境变量；缺少其中任意一项时不得把首次 NCCL collective 失败误判为
模型、数据或显存问题，也不得绕过多卡 smoke 直接启动正式训练。

```bash
export NCCL_CUMEM_HOST_ENABLE=0
export NCCL_IB_DISABLE=1
export NCCL_SOCKET_IFNAME=lo
export TORCH_NCCL_BLOCKING_WAIT=1
```

其中关键项是 `NCCL_CUMEM_HOST_ENABLE=0`，不能误写成语义不同的
`NCCL_CUMEM_ENABLE=0`。本约定只用于服务器运行环境，不写入公开 Hydra 实验配置；
`agent.md` 必须继续由 Git 忽略，不提交或推送到 GitHub。
