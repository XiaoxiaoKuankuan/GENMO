"""用当前整段留出验证选择在线Critic，旧自举标签只作漂移诊断。

真实 rollout 先按物理 episode 分成训练与留出集，同一段的相邻状态不能交叉。
只对训练集反传，所有目标保持本轮采集时固定的半马尔可夫 GAE。每个 epoch
检查留出MSE并记录初始短评估固定标签误差，按当前留出误差保存最佳参数与Adam；
连续无改善则提前结束并恢复最佳状态。固定集从不用于反传，也不重新采完整 MC。
这些指标仍含 bootstrap 偏差，仅防止已见的拟合过头，不能证明价值真值或物理收益。
冻结 Stage1 条件只编码一次，历史和已提交未来参考编码器仍在每次前向重新计算。
所有 collective 在本地异常同步块之外执行，支持不均匀分片和空本地留出集。
"""
from __future__ import annotations

import torch
from .learning_guards import capture_learning, restore_learning, split_episodes
from .optimization_support import _local_phase, _owned, balanced_epoch_order


def guarded_critic_update(critic, optimizer, rows, targets, cache, manifest, *, distributed,
                          generator, epochs, batch, grad_clip, guard, fixed_reference):
    critic.train()
    selected = [i for i, m in enumerate(manifest) if m['valid']]
    with _local_phase(distributed, 'critic_holdout_identities'):
        local_ids = {(distributed.rank, i): r.identity for i, r in enumerate(rows)}
    identities = {k: v for shard in distributed.all_gather_object(local_ids) for k, v in shard.items()}
    train, validation = split_episodes(selected, manifest, identities, guard['holdout_fraction'])
    from .critic_diagnostics import GROUPS, subgroup_labels
    with _local_phase(distributed, 'critic_subgroup_labels'):
        local_groups = {(distributed.rank, i): flags for i, flags in enumerate(subgroup_labels(rows))}
    group_labels = {k: v for shard in distributed.all_gather_object(local_groups) for k, v in shard.items()}
    subgroups = {name: [i for i in validation if name in group_labels[(manifest[i]['owner_rank'],
                  manifest[i]['local_index'])]] for name in GROUPS}
    indices = [i for _, i in _owned(selected, manifest, distributed)]
    lookup = {i: j for j, i in enumerate(indices)}
    with _local_phase(distributed, 'critic_guard_encode'):
        encoded = critic.encode_frozen(cache.critic_context([rows[i] for i in indices])) if indices else {}
        target = torch.as_tensor(targets['returns'], device=cache.device, dtype=torch.float32)
        fixed = []
        if fixed_reference is None:
            raise ValueError('Guarded Critic requires the durable initial short-evaluation reference')
        records = fixed_reference['records']
        for start in range(0, len(records), 32):
            chunk = records[start:start+32]
            context = {k: torch.cat([r['critic_context'][k] for r in chunk]).to(cache.device)
                       for k in chunk[0]['critic_context']}
            fixed.append((critic.encode_frozen(context),
                          torch.tensor([r['remaining_music_seconds'] for r in chunk], device=cache.device),
                          torch.tensor([r['target'] for r in chunk], device=cache.device),
                          torch.tensor([r['initial_prediction'] for r in chunk], device=cache.device)))

    def forward(local):
        ix = torch.tensor([lookup[i] for i in local], device=cache.device)
        features = {k: v.index_select(0, ix) for k, v in encoded.items()}
        return critic.forward_encoded(features, cache.get('remaining_music_seconds', [rows[i] for i in local]))

    def summarize(totals, allow_empty=False):
        n, s, s2, e, e2, absolute = distributed.sum_tensor(totals).cpu().tolist()
        if n < 1:
            if allow_empty:
                return dict(count=0, mse=None, rmse=None, mae=None, bias=None, explained_variance=None, target_variance=None)
            raise ValueError('Empty global Critic validation set')
        variance = max(0., s2/n-(s/n)**2)
        return dict(mse=e2/n, rmse=(e2/n)**.5, mae=absolute/n, bias=e/n, target_variance=variance,
                    explained_variance=None if variance <= 1e-12 else 1-(e2/n-(e/n)**2)/variance, count=int(n))

    def moments(expected, prediction):
        expected, prediction = expected.double(), prediction.double()
        error = expected-prediction
        return torch.stack((expected.new_tensor(len(expected)), expected.sum(), expected.square().sum(),
                            error.sum(), error.square().sum(), error.abs().sum()))

    @torch.no_grad()
    def metrics(global_indices, allow_empty=False):
        with _local_phase(distributed, 'critic_guard_validation'):
            totals = torch.zeros(6, device=cache.device, dtype=torch.float64)
            local = [i for _, i in _owned(global_indices, manifest, distributed)]
            for start in range(0, len(local), 128):
                chunk = local[start:start+128]
                totals += moments(target[chunk], forward(chunk))
        return summarize(totals, allow_empty=allow_empty)

    @torch.no_grad()
    def fixed_metrics(initial=False):
        with _local_phase(distributed, 'critic_guard_fixed_validation'):
            totals = torch.zeros(6, device=cache.device, dtype=torch.float64)
            for features, remaining, expected, original in fixed:
                totals += moments(expected, original if initial else critic.forward_encoded(features, remaining))
        return summarize(totals)

    initial, holdout_before = metrics(selected), metrics(validation)
    subgroup_before = {name: metrics(indices, allow_empty=True) for name, indices in subgroups.items()}
    fixed_initial, fixed_before = fixed_metrics(initial=True), fixed_metrics()
    best_loss, best_epoch, best_steps, stale = holdout_before['mse'], 0, 0, 0
    with _local_phase(distributed, 'critic_guard_snapshot'):
        best = capture_learning(critic, optimizer)
    checks, losses, norms = [], [], []
    visits = 0
    for epoch in range(epochs):
        order = balanced_epoch_order(train, manifest, generator) if distributed.rank == 0 else None
        order = distributed.broadcast_object(order)
        for start in range(0, len(order), batch):
            chosen = order[start:start+batch]
            local = [i for _, i in _owned(chosen, manifest, distributed)]
            optimizer.zero_grad(set_to_none=True)
            with _local_phase(distributed, 'critic_guard_backward'):
                loss = torch.zeros((), device=cache.device)
                if local:
                    loss = .5*(forward(local)-target[local]).square().sum()/len(chosen)
                    loss.backward()
            distributed.sum_gradients(critic)
            with _local_phase(distributed, 'critic_guard_step'):
                norm = torch.nn.utils.clip_grad_norm_([p for p in critic.parameters() if p.requires_grad],
                                                      grad_clip, error_if_nonfinite=True)
                optimizer.step()
            losses.append(loss.detach()); norms.append(float(norm)); visits += len(chosen)
        holdout, stable = metrics(validation), fixed_metrics()
        accepted = holdout['mse'] < best_loss
        checks.append(dict(epoch=epoch+1, holdout=holdout, fixed=stable, selected=accepted))
        if accepted:
            with _local_phase(distributed, 'critic_guard_best_snapshot'):
                best = capture_learning(critic, optimizer)
            best_loss, best_epoch, best_steps, stale = holdout['mse'], epoch+1, len(losses), 0
        else:
            stale += 1
        if stale >= guard['patience_epochs']:
            break
    with _local_phase(distributed, 'critic_guard_restore_best'):
        restore_learning(critic, optimizer, best)
    final, holdout_after, fixed_after = metrics(selected), metrics(validation), fixed_metrics()
    subgroup_after = {name: metrics(indices, allow_empty=True) for name, indices in subgroups.items()}
    return dict(before=initial, after=final, optimizer_steps=best_steps, attempted_optimizer_steps=len(losses),
                epochs=len(checks), selected_epoch=best_epoch, validation_epochs=checks,
                holdout_before=holdout_before, holdout_after=holdout_after,
                subgroup_before=subgroup_before, subgroup_after=subgroup_after,
                subgroup_scope='current_holdout_observed_intervals_bootstrapped_targets_not_MC_truth',
                fixed_before=fixed_before, fixed_after=fixed_after, fixed_initial=fixed_initial,
                fixed_reference_role='diagnostic_only', selection_metric='current_episode_holdout_mse',
                training_actions=len(train), validation_actions=len(validation),
                split_scope='whole_physical_episodes_no_adjacent_state_leakage',
                losses=distributed.sum_tensor(torch.stack(losses)).cpu().tolist() if losses else [],
                gradient_norms=norms, global_sample_visits=visits, frozen_encoder_passes=int(bool(indices)),
                feature_bytes=sum(v.numel()*v.element_size() for v in encoded.values()),
                critic_feature_contract='frozen_music_prefix_cached_trainable_history_and_future_recomputed',
                metric_scope='fixed_bootstrapped_targets_not_unbiased_MC_truth',
                best_model_and_adam_restored=True)
