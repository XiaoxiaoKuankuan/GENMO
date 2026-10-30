2026-10-30 当前根位置/连续性模型：64首训练集真实物理评价与五首视频
============================================================================================

结果：训练集内存在明确真实跟踪收益。两模型各重复两遍64首，均128/128完整完成。
根位置RMSE由0.333750m降为0.138147m（-58.61%），Tracking提高8.25%，关节q误差降低15.88%。
jerk提高3.09%、接触脚滑提高1.93%、Beat F1降低1.86%；因此不能称所有指标均改善。

范围、配对与候选选择
------------------------------------------------------------------------------

- 本次只评价现有权重，没有训练、重新拟合WM、更新归一化、修改奖励或调整模型参数。
- 评价对象为formal_root_continuity_20261029_2157八组实验，不与上一轮不同奖励/约束的历史分数直接混比。
- 2026-10-30 10:57:46（北京时间）锁定八组validation best和最新已保存权重，共16个不可变快照。
- 每个候选覆盖同64首train、同冻结GENMO生成缓存、真实初态和物理参数；50Hz控制、200Hz物理，无动作探索与观测噪声。
- 筛选顺序：完整完成率降序、世界根RMSE升序、完整音乐预算Tracking降序、jerk升序。
- 选择GPU7/u2600后单独复跑两遍候选、两遍零Adapter；每首两遍平均后按64首等权平均。
- 此为训练集内验证及模型选择，不把结果称为独立测试集泛化，也没有重写原有validation/evaluation。
- GPU7初筛根RMSE为0.136872m，GPU6为0.137333m，差0.46毫米，接近并列；GPU6的Tracking更高。GPU7并非各指标绝对最佳。
- 八组权重更新数不同，候选选择不能作为等预算时域或正则强度的因果消融。

选中模型：H64、根辅助lambda=1.4910286903381347、参考连续性强档（弱档各项权重乘3）、seed42、Critic关闭。
GENMO、SuperMimic Actor及normalizer冻结；根XY输出关闭，仍为2654→512→512→256→270。

保全权重（weights-only，不能冒称完整优化器恢复包）：::

  /data0/user/liwei/GENMO_outputs/stage2_supermimic_adapter/train64_rootcont_20261030_105746/snapshots/g7_latest_u002600.pt

文件SHA256：``437c88295b504c46da6e1f57cd49c4db0243da4c8a80b6c4dac483b266f628fa``。

64首整体结果
------------------------------------------------------------------------------

.. list-table:: 两遍逐歌平均再等权汇总；变化以基线为分母
   :header-rows: 1

   * - 指标
     - 零Adapter
     - GPU7/u2600
     - 变化
     - 改善歌曲数
   * - 完成次数
     - 128/128
     - 128/128
     - 相同
     - 两侧全部完成
   * - 完整预算Tracking
     - 0.097236
     - 0.105255
     - +8.246%
     - 64/64
   * - 世界根位置RMSE / m
     - 0.333750
     - 0.138147
     - -58.608%
     - 58/64
   * - 根姿态RMSE / rad
     - 0.054961
     - 0.048918
     - -10.997%
     - 57/64
   * - 关节q RMSE / rad
     - 0.100645
     - 0.084664
     - -15.879%
     - 64/64
   * - 关节qd RMSE / rad/s
     - 0.715618
     - 0.677525
     - -5.323%
     - 63/64
   * - 身体姿态RMSE / rad
     - 0.129278
     - 0.110478
     - -14.542%
     - 64/64
   * - 关节jerk RMS / rad/s^3
     - 1137.669148
     - 1172.853486
     - +3.093%
     - 12/64
   * - 接触脚滑 / m/s
     - 0.063217
     - 0.064435
     - +1.926%
     - 17/64
   * - Beat F1
     - 0.353548
     - 0.346980
     - -1.858%
     - 20/64

Tracking以相同六项公式、误差尺度和求和权重计算，包含原dt，不混入各实验不同的根辅助/连续性成本。
完成率按每首实际缓存的完整音乐预算计算；没有用失败后的伪造状态补齐。
两次候选评价根RMSE分别0.138213和0.138080m。逐歌根RMSE两遍绝对差的平均为：基线0.018813m、候选0.009668m；不能用总体均值接近声称每首完全可重复。

原SM六项（各原始0～1项，未乘求和权重与dt）：

.. list-table::
   :header-rows: 1

   * - 项目
     - 零Adapter
     - GPU7/u2600
     - 相对变化
   * - body_pos
     - 0.929562
     - 0.944488
     - +1.606%
   * - body_ori
     - 0.530654
     - 0.631808
     - +19.062%
   * - body_linvel
     - 0.965298
     - 0.966982
     - +0.174%
   * - body_angvel
     - 0.907900
     - 0.914269
     - +0.702%
   * - root_pos
     - 0.468210
     - 0.693370
     - +48.090%
   * - root_ori
     - 0.966545
     - 0.973145
     - +0.683%

根位置未改善的6首
------------------------------------------------------------------------------

- AIOZ-GDANCE_h8-0X7fMIqM_02_0_810_dancer_03：0.096687→0.108667m（+12.39%）。
- AIST++_gLO_sBM_cAll_d15_mLO3_ch04：0.031869→0.043648m（+36.96%）。
- AIST++_gMH_sBM_cAll_d24_mMH2_ch03：0.055130→0.083922m（+52.22%）。
- AIST++_gPO_sBM_cAll_d11_mPO4_ch03：0.044788→0.053598m（+19.67%）。
- AIST++_gWA_sBM_cAll_d26_mWA4_ch03：0.064520→0.065553m（+1.60%）。
- FineDance_115：0.321094→0.336858m（+4.91%）。

八组最新已保存候选的初筛结果
------------------------------------------------------------------------------

以下各候选一遍、对应worker零基线两遍；与最终确认表的重复次数不同。全部64/64完成。

.. list-table::
   :header-rows: 1

   * - GPU / 更新
     - 根RMSE变化
     - Tracking变化
     - q误差变化
     - jerk变化
     - 脚滑变化
   * - GPU0 / 5000
     - -37.00%
     - +6.88%
     - -23.02%
     - +7.84%
     - +7.04%
   * - GPU1 / 5000
     - -39.06%
     - +6.61%
     - -21.59%
     - +6.56%
     - +6.44%
   * - GPU2 / 5000
     - -45.82%
     - +7.23%
     - -22.40%
     - +7.31%
     - +7.56%
   * - GPU3 / 5000
     - -44.50%
     - +6.94%
     - -20.62%
     - +6.61%
     - +6.05%
   * - GPU4 / 2600
     - -55.29%
     - +8.52%
     - -17.84%
     - +3.30%
     - +2.64%
   * - GPU5 / 2800
     - -56.17%
     - +8.34%
     - -16.91%
     - +2.90%
     - +3.09%
   * - GPU6 / 2600
     - -58.92%
     - +8.89%
     - -16.82%
     - +3.05%
     - +2.32%
   * - GPU7 / 2600
     - -59.06%
     - +8.36%
     - -15.92%
     - +3.01%
     - +1.85%

五首视频与真实轨迹
------------------------------------------------------------------------------

五首从双方两遍均完整完成、根误差及Tracking均改善的样本中选取，按Tracking绝对增益排序并去重音频。
它们是改善明显的展示正例；总体结论来自前述全部64首。视频使用首遍真实轨迹，下面视频数值不混用两遍均值。
左右共用同一名义GENMO目标、同一世界相机、同一音乐时钟。左橙色为零Adapter真实执行，右绿色为候选真实执行，蓝色影子为共同原始名义目标。
世界位置不重新对齐，不平滑执行轨迹，不改变播放速度；通过MuJoCo mj_forward回放Newton实测状态，非另一次MuJoCo动力学测试。
全部视频1600×842、50FPS、H264/AAC，含原音乐；5首完整视频加50.022秒精选合集（每首10秒）。

.. list-table:: 视频首遍、相同完整音乐区间
   :header-rows: 1

   * - 文件
     - 时长 / s
     - 零Adapter根RMSE / cm
     - 候选根RMSE / cm
   * - 01_Mine_dance_3__Abracadabra.mp4
     - 57.16
     - 37.602
     - 8.258
   * - 02_AIOZ-GDANCE_sD2oK_kPCE4_11_0_1620_dancer_01.mp4
     - 53.96
     - 28.507
     - 6.825
   * - 03_AIOZ-GDANCE_8QvKh-jvI1I_05_0_1290_dancer_04.mp4
     - 42.96
     - 30.976
     - 8.223
   * - 04_FineDance_157.mp4
     - 165.82
     - 49.425
     - 19.436
   * - 05_Mine_dance_3__Foreveryoung.mp4
     - 104.26
     - 78.702
     - 13.419

代码、验证、进程与留存
------------------------------------------------------------------------------

- 3963883404990f1db90c55dee3858743b8058c2f：evaluate_adapter_train_matrix.py新增16候选/重复基线/冻结资产实测入口；render_adapter_paired_ghost.py增加共同nominal影子选项，默认corrected模式兼容旧调用。
- 10874fb56aaa7a2ad9a79b685b289eb93b391a87：summarize_adapter_train_matrix.py新增64首配对身份核验、逐歌等权重复汇总、16候选排序和五首去重选择。
- 本次报告提交只增加证据文档与记录文本，不改变任何活跃训练代码。AGENTS.md原用户修改保留且不提交。
- CPU配对身份/失败前缀测试4 passed；入口AST检查及64行合成算术/初态不匹配检查通过。
- 实际筛选16×64次候选＋4×64次零基线，确认2×64次候选＋2×64次零基线，共1536次完整episode。
- 共4390344条有效评价转移（只计活跃歌曲），物理引擎包含已结束空闲env的总转移计数19749888；两者不能混淆。所有completed.json记录optimizer_steps=0和actor_unchanged=true。
- 5个完整MP4和合集均ffprobe与ffmpeg全量解码通过；本地5个MP4、5个PNG和合集11文件SHA一致，胜出权重及10条NPZ逐一SHA通过。已目视检查Abracadabra/FineDance157/ForeverYoung峰值图、精选合集抽帧和64首总图。
- 评价/渲染临时测试目录已精确清理；正式报告、实际轨迹、视频及保全模型为用户交付产物，永久保留。
- 11:59:37检查八组原训练PID均在，更新数依GPU0～7为5591/5617/5595/5609/3010/3023/2819/2945；本轮未停止、重启或修改原任务。实际快照见training_status_at_delivery.json。

服务器结果根：``/data0/user/liwei/GENMO_outputs/stage2_supermimic_adapter/train64_rootcont_20261030_105746``。
本地交付根：``/home/weili/bumi-closedloop-worktrees/adapter/evaluation_train64_rootcont_20261030_105746``。
完整数值位于report/summary.json、per_song_comparison.csv、candidate_comparison.csv；5首轨迹/视频元数据分别位于confirmation/与media/。
胜出weights快照保存在snapshots/；配置保存在configs/。所有256条确认轨迹都留在服务器，本地保留五首两侧共10条轨迹。

已实际执行的命令
------------------------------------------------------------------------------

::

  nice -n 10 /data0/user/liwei/envs/genmo_adapter_runtime_20261026/bin/python -B -u /data0/user/liwei/GENMO_outputs/stage2_supermimic_adapter/train64_rootcont_20261030_105746/source_genmo/tools/eval/evaluate_adapter_train_matrix.py --manifest /data0/user/liwei/GENMO_outputs/stage2_supermimic_adapter/train64_rootcont_20261030_105746/candidates.json --output /data0/user/liwei/GENMO_outputs/stage2_supermimic_adapter/train64_rootcont_20261030_105746/worker0 --worker-index 0 --num-workers 2 --visualizer none --deterministic

::

  nice -n 10 /data0/user/liwei/envs/genmo_adapter_runtime_20261026/bin/python -B -u /data0/user/liwei/GENMO_outputs/stage2_supermimic_adapter/train64_rootcont_20261030_105746/source_genmo/tools/eval/evaluate_adapter_train_matrix.py --manifest /data0/user/liwei/GENMO_outputs/stage2_supermimic_adapter/train64_rootcont_20261030_105746/candidates.json --output /data0/user/liwei/GENMO_outputs/stage2_supermimic_adapter/train64_rootcont_20261030_105746/worker1 --worker-index 1 --num-workers 2 --visualizer none --deterministic

::

  nice -n 10 /data0/user/liwei/envs/genmo_adapter_runtime_20261026/bin/python -B -u /data0/user/liwei/GENMO_outputs/stage2_supermimic_adapter/train64_rootcont_20261030_105746/source_genmo/tools/eval/evaluate_adapter_train_matrix.py --manifest /data0/user/liwei/GENMO_outputs/stage2_supermimic_adapter/train64_rootcont_20261030_105746/confirmation_candidates.json --output /data0/user/liwei/GENMO_outputs/stage2_supermimic_adapter/train64_rootcont_20261030_105746/confirmation --repeat-candidate 2 --save-trajectories --visualizer none --deterministic

::

  /data0/user/liwei/envs/genmo_adapter_runtime_20261026/bin/python -B -u /home/user/liwei/bumi-closedloop-worktrees/adapter/GENMO/tools/eval/render_adapter_paired_ghost.py --trajectories /data0/user/liwei/GENMO_outputs/stage2_supermimic_adapter/train64_rootcont_20261030_105746/confirmation/zero_trajectories/033_Mine_dance_3__Abracadabra.json /data0/user/liwei/GENMO_outputs/stage2_supermimic_adapter/train64_rootcont_20261030_105746/confirmation/g7_latest_u002600_trajectories/033_Mine_dance_3__Abracadabra.json --output /data0/user/liwei/GENMO_outputs/stage2_supermimic_adapter/train64_rootcont_20261030_105746/media/01_Mine_dance_3__Abracadabra.mp4 --ghost-reference nominal --font /data0/user/liwei/GENMO_outputs/stage2_supermimic_adapter/train64_best700_20261029/NotoSansCJK-Regular.ttc --labels '零Adapter：冻结GENMO + SuperMimic' '训练集候选第一：GPU7 / u2600'
