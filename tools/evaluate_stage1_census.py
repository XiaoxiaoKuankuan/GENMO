"""启动冻结原Stage1的全量音乐真实物理评价工作进程。

本入口复用GENMO现有Newton装配、原20步DDIM及异步参考调度；具体任务从全量评价
专用SQLite队列领取，逐曲输出真实轨迹和完成/失败记录。不加载Stage2训练权重，
不创建任何优化器，不改变已有正式训练目录；任务准备与多卡监督由stage1_census模块提供。
"""
from gem.closedloop.mb_control.entry import main

if __name__ == '__main__':
    main('census')
