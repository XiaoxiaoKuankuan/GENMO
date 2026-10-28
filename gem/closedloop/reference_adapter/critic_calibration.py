"""Critic 的真实回报锚定、可证明上界和独立物理校准。

本模块不启动训练，不修改冻结 Actor，不把想象 GAE 当真实标签。真实 MC 从同一
episode 的连续奖励向真正终止回算；随机中途起点只能称完整后缀，不能称整歌。
未观测到真实终止、发生时间缺口或跨 reset 的样本一律不生成完整 MC 标签。
当前 Stage2 正式配置关闭终端价值；这里为后续显式重新启用提供独立接口：训练集
真实回报监督、独立 validation 逐歌曲校准，以及由原奖励上界推导的价值上界。
上界是数值保护，不是校准的替代；不会仅因裁剪后的值合法就批准 Critic。
"""

import torch


def discounted_value_upper_bound(*, gamma=.99, reward_upper=.1, remaining_steps=None):
    """原六项 Tracking 权重总和 5、dt=.02；其他现用项非正，故 r<=.1。"""
    if not 0 <= gamma < 1 or reward_upper < 0:
        raise ValueError('折扣和每步奖励上界非法')
    if remaining_steps is None:
        return reward_upper / (1-gamma)
    if isinstance(remaining_steps, torch.Tensor):
        if bool((remaining_steps < 0).any()):
            raise ValueError('剩余真实参考步数不能为负')
        return reward_upper * (1-gamma**remaining_steps) / (1-gamma)
    if remaining_steps < 0:
        raise ValueError('剩余真实参考步数不能为负')
    return reward_upper * (1-gamma**remaining_steps) / (1-gamma)


def bounded_terminal_value(value, *, gamma=.99, remaining_steps=None, reward_upper=.1):
    """仅在已独立校准后使用；超上界处梯度真实为零，不使用直通估计。"""
    bound = torch.as_tensor(discounted_value_upper_bound(gamma=gamma, reward_upper=reward_upper,
        remaining_steps=remaining_steps), device=value.device, dtype=value.dtype)
    return torch.minimum(value, bound)


@torch.no_grad()
def real_mc_suffixes(reward, episode, frame, terminated, valid, *, gamma=.99):
    """输入 [T,N]，输出同形真实回报与完整后缀 mask，不从截断处 bootstrap。

    terminated 仅允许真实物理失败或歌曲/参考真实结束，行政停止必须传 False。
    起点可以在歌曲中途；是否整歌必须由调用者根据 start_frame 单独报告。
    """
    shape = reward.shape
    if reward.ndim != 2 or any(x.shape != shape for x in (episode, frame, terminated, valid)):
        raise ValueError('真实 MC 需要同形 [T,N] 奖励、episode、frame、终止和有效性')
    if not 0 <= gamma < 1:
        raise ValueError('gamma 必须位于 [0,1)')
    result = torch.zeros_like(reward)
    complete = torch.zeros_like(valid, dtype=torch.bool)
    following = reward.new_zeros(shape[1])
    known = torch.zeros(shape[1], dtype=torch.bool, device=reward.device)
    for t in range(shape[0]-1, -1, -1):
        finite = valid[t].bool() & torch.isfinite(reward[t])
        terminal = finite & terminated[t].bool()
        continuous = torch.zeros_like(known)
        if t+1 < shape[0]:
            continuous = (episode[t] == episode[t+1]) & (frame[t+1] == frame[t]+1)
        usable = finite & (terminal | (known & continuous))
        value = reward[t] + gamma*torch.where(terminal, torch.zeros_like(following), following)
        result[t] = torch.where(usable, value, torch.zeros_like(value))
        complete[t] = usable
        following, known = result[t], usable
    return result, complete


def real_anchor_loss(critic, features, returns, *, split, valid=None):
    """只监督真实 train 完整 MC；验证歌曲与想象 target 不得调用此接口拟合。"""
    if split != 'train':
        raise ValueError('Critic 真实锚定仅允许 train 歌曲')
    mask = torch.ones_like(returns, dtype=torch.bool) if valid is None else valid.bool()
    if not bool(mask.any()) or not bool(torch.isfinite(returns[mask]).all()):
        raise ValueError('没有有限且完整的真实回报标签')
    return (critic(features[mask])-returns[mask].detach()).square().mean()


@torch.no_grad()
def physical_calibration(predicted, actual, songs, *, split, train_return_mean, min_samples_per_song=32,
                         max_normalized_rmse=.25, min_ev=0., gamma=.99, reward_upper=.1):
    """逐歌曲独立真实回报校准，先检查未裁剪值，不能靠 clamp 伪造通过。

    train_return_mean 必须来自真实 train 标签；验证集不拟合常数、统计量或网络。
    标准差极小的歌曲不强求 EV，但仍须满足绝对尺度误差和 train 常数基线。
    """
    if split not in ('validation', 'audit'):
        raise ValueError('Critic 重新启用需要独立 validation/audit 真实回报')
    if predicted.ndim != 1 or predicted.shape != actual.shape or songs.shape != actual.shape:
        raise ValueError('Critic 校准需要一维等长预测、真实回报和歌曲 ID')
    upper = discounted_value_upper_bound(gamma=gamma, reward_upper=reward_upper)
    finite = torch.isfinite(predicted) & torch.isfinite(actual)
    rows = []
    for song in songs.unique().tolist():
        mask = songs == song
        y, p = actual[mask], predicted[mask]
        enough = len(y) >= min_samples_per_song
        okay = bool(finite[mask].all())
        variance = y.var(unbiased=False)
        mse = (p-y).square().mean()
        baseline = (y-float(train_return_mean)).square().mean()
        ev = float(1-(p-y).var(unbiased=False)/variance) if variance > 1e-10 else None
        violations = int((p > upper+1e-6).sum())
        passed = (enough and okay and violations == 0 and float(mse.sqrt()) <= max_normalized_rmse*upper
                  and float(mse) <= float(baseline)+1e-8 and (ev is None or ev >= min_ev))
        rows.append(dict(song_id=int(song), samples=len(y), mse=float(mse), rmse=float(mse.sqrt()),
            ev=ev, train_constant_baseline_mse=float(baseline), upper_bound_violations=violations,
            mean_predicted=float(p.mean()), mean_real=float(y.mean()), passed=passed))
    return dict(qualified=bool(rows) and all(r['passed'] for r in rows), split=split,
        target='independent_real_complete_MC_suffix', reward_upper=reward_upper, value_upper=upper,
        samples=actual.numel(), songs=rows, terminal_value_requires_explicit_enable=True)


def terminal_value_allowed(requested, report, *, real_anchor_updates):
    """只有明确开启、真实锚定及逐歌校准均满足才允许终端梯度。"""
    return bool(requested and real_anchor_updates > 0 and report and report.get('qualified')
                and report.get('target') == 'independent_real_complete_MC_suffix'
                and report.get('split') in ('validation', 'audit'))


@torch.no_grad()
def preserve_critic_input_function(critic, new_mean, new_var, new_count, *, optimizer=None):
    """更新 Critic 输入统计时同步重参数化首层，避免仅改 norm 造成价值突跳。

    Adam 在首层 W/b 的耦合仿射换元下不能由逐元素一阶/二阶矩精确搬迁。若已有
    optimizer，显式只重建受影响首层的 Adam 状态，保留其余层，调用者必须记录此边界。
    """
    from .wm_distribution import rebase_input_normalizer
    return rebase_input_normalizer(critic.norm, critic.net[0], new_mean, new_var, new_count,
                                   optimizer=optimizer)
