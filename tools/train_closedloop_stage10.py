"""第十步完整数据集上的单环境连续DPPO训练、真正续训及独立评估入口。

本入口不读取200首选择清单。四库全部train/val/test先建立身份并逐文件审计，训练
只从完整train池抽任务；随机音乐起点与配对活动度同步。每轮用当前Actor重新采集，
固定旧价值/优势，分别更新Critic和Actor，复核真实联合KL及GMT冻结状态后发布完整
checkpoint。恢复重新创建物理session，不复用旧Buffer或假称恢复PhysX内部状态。

默认配置是正式训练前有限检查，外层轮数与实际optimizer候选次数分别计账。异常
停止当前session，保留执行证据和消耗，后续显式从最近已发布checkpoint恢复；不在
半更新的Critic/BC状态上静默继续。SIGINT/SIGTERM只请求在下一完整轮次边界停止。

eval必须显式传入本入口的完整checkpoint，恢复其Actor/Critic并禁止优化；固定
val/test歌曲与种子输出逐episode证据。源码、模型、奖励、完整数据身份严格绑定，
跨Stage9初始化仅允许明确的Actor/Critic权重迁移，优化器和采样器重新建立。

恢复身份显式绑定基础种子、模型/时钟/运行时/诊断配置；预算和磁盘容量另由运行账本
管理，不依赖某个固定准备YAML的源码哈希。decision、attempt、episode_count共同
参与请求和显式噪声种子，必须在初始及逐轮checkpoint中保存并完整恢复；缺失字段
的旧Stage10状态明确拒绝，不猜测为零。每个session另保存实际输入和解析配置SHA。

八卡入口可显式注入同步learner：主进程保持单GMT采集、全局GAE、预算与发布，
Actor/Critic更新交给多GPU共同计算。普通单卡入口不接受八卡配置，避免静默降级。
"""
from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path
import random
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
import yaml

from gem.closedloop.baseline_provenance import collect_source_provenance, verify_source_provenance
from gem.closedloop.dppo.budget import atomic_json
from gem.closedloop.dppo.buffer import RolloutBuffer
from gem.closedloop.dppo.checkpoint import (VERSION as CHECKPOINT_VERSION, load_checkpoint,
    save_checkpoint, _validate_model_state)
from gem.closedloop.dppo.critic import UpperCritic
from gem.closedloop.dppo.env_adapter import UpperEnvironment
from gem.closedloop.dppo.evaluation import build_evaluation_tasks, evaluate_policy
from gem.closedloop.dppo.full_dataset import FullMusicCatalog, FullMusicSampler
from gem.closedloop.dppo.policy import DPPODiffusionPolicy
from gem.closedloop.dppo.rewards import resolve_reward_config
from gem.closedloop.dppo.rpc import AcknowledgedBackend
from gem.closedloop.dppo.run_management import RunManager, GuardedStepJournal, RolloutWriter, StopSignal
from gem.closedloop.dppo.long_run import LongRunMaintenance, validate_long_run_settings
from gem.closedloop.dppo.trainer import (load_actor, fixed_targets, critic_update, actor_update,
    probability_check, analytic_kl, SupervisedAnchor, populate_values)
from gem.closedloop.evaluation_music import sha256_file
from gem.closedloop.frozen_actor import _fingerprint
from gem.closedloop.online_conditions import OnlineConditionBuilder
from gem.robots.bumi.feature_codec import BumiMotionFeatureCodec
from gem.robots.bumi.kinematics import BumiKinematics
from tools.eval.run_closedloop_baseline import Workers, preflight

VERSION = 'genmo.closedloop.stage10.v1'
VERSION_V2 = 'genmo.closedloop.stage10.v2'


class _PreflightComplete(Exception):
    """只用于跳过模型阶段；统一finally仍可把源码/资产检查失败变为非零退出。"""


def _positive(value, name, *, integer=False):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
        raise ValueError(f'{name} must be positive and finite')
    if integer and type(value) is not int:
        raise ValueError(f'{name} must be an integer')


def configuration(path):
    config = yaml.safe_load(Path(path).read_text())
    if 'base_config' in config or config.get('stage10', {}).get('version') not in (VERSION, VERSION_V2):
        raise ValueError('Stage10 requires its own explicit full-dataset configuration')
    stage = config['stage10']
    from gem.closedloop.dppo.training_scale import derive_training_scale
    derive_training_scale(config)
    parallel_v2 = stage['version'] == VERSION_V2
    if type(stage['seed']) is not int or not 0 <= stage['seed'] < 2**32:
        raise ValueError('Stage10 base seed must be an integer in [0, 2**32)')
    train = stage['training']
    forbidden = {'music_selection', 'selection_path', 'groups_per_dataset'}
    if forbidden.intersection(train) or forbidden.intersection(stage['dataset']):
        raise ValueError('Stage10 cannot use a preselected music subset')
    for key in ('rollout_upper_steps', 'critic_steps', 'critic_batch', 'bc_batch', 'denoising_steps'):
        _positive(train[key], key, integer=True)
    for key in ('actor_lr', 'critic_lr', 'grad_clip_norm', 'critic_grad_clip_norm', 'episode_seconds',
                'latency_budget_s', 'std_floor', 'eta', 'kl_stop_joint'):
        _positive(train[key], key)
    for key in ('gamma_upper', 'lambda_upper', 'gamma_denoising'):
        _positive(train[key], key)
        if train[key] > 1:
            raise ValueError(f'{key} must be <= 1')
    if not 0 < train['ppo_clip'] < 1:
        raise ValueError('PPO clip must be between zero and one')
    if not parallel_v2 and (train['ppo_epochs'] != 1 or train['denoising_microbatch'] != 1):
        raise ValueError('This version preserves one accumulated Actor step and microbatch=1')
    if train['log_probability_reduction'] != 'joint_sum' or train['execution_mode'] != 'latency':
        raise ValueError('Stage10 preserves joint_sum and the verified latency execution semantics')
    candidates = train.get('actor_lr_candidates', [train['actor_lr']])
    if (not isinstance(candidates, list) or not 1 <= len(candidates) <= 3
            or any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v)
                   or not 0 < v <= 1e-6 for v in candidates)
            or any(a >= b for a, b in zip(candidates, candidates[1:]))):
        raise ValueError('Require one to three increasing finite Actor learning-rate candidates')
    if train['actor_lr'] not in candidates:
        raise ValueError('Initial Actor learning rate must belong to the tested candidates')
    if parallel_v2:
        from gem.closedloop.dppo.parallel_support import validate_v2_configuration
        validate_v2_configuration(config)
    if not math.isfinite(train['bc_weight']) or train['bc_weight'] < 0:
        raise ValueError('BC weight must be finite and nonnegative')
    if train['denoising_steps'] != config['model']['ddim_steps'] or train['guidance_scale'] != config['model']['guidance_scale']:
        raise ValueError('Policy and environment model settings differ')
    prefix = train['bc_prefix']
    if (type(prefix['min_frames']) is not int or type(prefix['max_frames']) is not int
            or not 1 <= prefix['min_frames'] <= prefix['max_frames'] < 120
            or not 0 <= prefix['zero_probability'] < 1):
        raise ValueError('Invalid supervised prefix coverage')
    limits = stage['limits']
    if set(limits) != {'accepted_iterations', 'optimizer_attempts', 'generations', 'control_steps', 'physics_steps'}:
        raise ValueError('Stage10 requires all five independent budget limits')
    for key, value in limits.items():
        _positive(value, key, integer=True)
    if limits['physics_steps'] != 4 * limits['control_steps']:
        raise ValueError('Physics budget must preserve four substeps per control interval')
    if limits['optimizer_attempts'] < limits['accepted_iterations']:
        raise ValueError('Optimizer attempt budget cannot be less than accepted iteration budget')
    storage = stage['storage']
    for key in ('min_free_bytes', 'max_run_bytes', 'checkpoint_reserve_bytes', 'rollout_chunk_size'):
        _positive(storage[key], key, integer=True)
    validate_long_run_settings(stage)
    if stage['dataset']['split'] != 'train' or not isinstance(stage['dataset']['random_start'], bool):
        raise ValueError('Training must use the complete train split with an explicit start policy')
    probabilities = stage['dataset']['source_probabilities']
    if len(probabilities) != 4 or any(not math.isfinite(p) or p <= 0 for p in probabilities) or not math.isclose(sum(probabilities), 1.):
        raise ValueError('Full training source probabilities must contain four positive values summing to one')
    reward = resolve_reward_config(stage['reward'])
    # UpperEnvironment/SupervisedAnchor沿用已验收接口；这是同值适配，不是另一份可漂移配置。
    config['stage9'] = dict(train, reward=reward, seed=stage['seed'],
                            run_id='assigned_from_run', bc_data_root=config['paths']['data_root'])
    config['diagnostics']['reward_substeps'] = True
    return config


def runtime_preflight(config, *, check_gpu):
    """只复用资产/GPU/物理契约检查；数据选择完全由全量目录审计负责。"""
    bridge = copy.deepcopy(config)
    bridge['evaluation'] = dict(split='train', datasets=['AIST++', 'AIOZ-GDANCE', 'FineDance', 'Mine'],
        seeds=[config['stage10']['seed']], seconds=30, modes=['latency'], groups_per_dataset=1,
        require_audio=False, selection_seed=config['stage10']['seed'])
    result = preflight(bridge, [], check_gpu=check_gpu)
    result.pop('selected', None)
    result['data_scope'] = 'assets_only; complete dataset checked separately'
    return result


def _sources(config, check):
    stage = config['stage10']
    names = ('policy', 'buffer', 'rewards', 'critic', 'returns', 'music_tasks', 'target_activity',
             'env_adapter', 'rpc', 'budget', 'trainer', 'checkpoint', 'lr_calibration',
             'full_dataset', 'run_management', 'evaluation', 'long_run', 'archive_store')
    if config['stage10']['version'] == VERSION_V2:
        names += ('parallel_support', 'parallel_training', 'execution_profile', 'periodic_monitor',
                  'updater_v2', 'position_repair', 'archives', 'archive_process', 'optional_diagnostics',
                  'performance', 'tensor_cache', 'asset_cache', 'rollout_storage', 'budget_ledger',
                  'batch_execution', 'execution_checks', 'journal_codec', 'dual_collector',
                  'numerical_execution', 'training_scale', 'fixed_tile_linear', 'prepaid_budget', 'update_observation',
                  'effect_checkpoint', 'compensated_gemm', 'denoising_sampling')
        if stage.get('performance', {}).get('fixed_work_probe_iteration') is not None:
            names += ('fixed_work_probe',)
    vector = config.get('runtime',{}).get('backend') == 'gpu_vectorized.v1'
    if vector:
        names += ('vector_collector','vector_boundary','vector_environment','vector_runtime','vector_evaluation',
                  'rollback_audit','sampling_graph','condition_sampling_graph','vector_metrics',
                  'vector_reward_math','vector_reward_adapter','columnar_reward','vector_devices','deployment_clock','vector_generation',
                  'world_flow','world_collector')
    gmt_additional = [f'source/NoetixRobot/NoetixRobot/tasks/mimic/mimic_noetix_bumi4340_mha_sonic/closedloop/{name}.py'
                      for name in ('execution_journal', 'execution_copy')]
    if vector:
        gmt_additional += [f'source/NoetixRobot/NoetixRobot/tasks/mimic/mimic_noetix_bumi4340_mha_sonic/closedloop/{name}.py'
            for name in ('vector_backend','vector_env','vector_reference','vector_diagnostics','vector_service','vector_journal','vector_columns')]
        gmt_additional += [f'scripts/rsl_rl/{name}.py' for name in
            ('serve_frozen_gmt_vector','bumi4340_frozen_torch_policy','vector_app_resources')]
    return collect_source_provenance(config['paths'], repository_state=check['repositories'], additional_files={
        'genmo_repo': [*(f'gem/closedloop/dppo/{n}.py' for n in names),
            'tools/train_closedloop_stage10.py', 'gem/runtime/trajectory_blocks.py', 'gem/closedloop/__init__.py',
            *(['tools/verify_stage10_saved_learning.py','tools/verify_stage10_gradient_reductions.py',
               'tools/stage10_step_gradient_report.py'] if stage.get('performance', {}).get('fixed_work_probe_iteration') is not None else []),
            'configs/closedloop/stage1_dataset_server1_fourset_90505_v1.yaml',
            'gem/closedloop/stage1_dataset.py', 'gem/closedloop/losses.py',
            *(['gem/closedloop/dppo/distributed_runtime.py', 'tools/train_closedloop_stage10_8gpu.py']
              if config['stage10'].get('distributed') else [])],
        'gmt_repo': gmt_additional})


def identity(config, check, provenance, catalog, data_audit, actor):
    return dict(stage10=config['stage10']['version'], assets=check['asset_sha256'], actor_interface=dict(actor.interface_config),
        base_seed=config['stage10']['seed'], execution_contract=dict(protocol_version=config['version'],
            model=copy.deepcopy(config['model']), timing=copy.deepcopy(config['timing']),
            runtime=copy.deepcopy(config['runtime']), diagnostics=copy.deepcopy(config['diagnostics']),
            interpreters={name:config['paths'][name] for name in ('genmo_python', 'isaac_python')}),
        training_contract=config['stage10']['training'], reward=config['stage9']['reward'],
        environment=config['environment'], termination=config['termination'],
        dataset=catalog.identity, data_content_sha256=data_audit['data_content_sha256'],
        sampling=config['stage10']['dataset'], source_manifest_sha256=provenance['source_manifest_sha256'],
        **({'performance_contract': copy.deepcopy(config['stage10']['performance'])}
           if config['stage10'].get('performance') else {}),
        **({'distributed_training': copy.deepcopy(config['stage10']['distributed'])}
           if config['stage10'].get('distributed') else {}))


def validate_execution_state(state):
    """旧状态缺失噪声计数时拒绝恢复；不能把显式采样种子的组成部分猜成零。"""
    for name in ('decision', 'attempt', 'episode_count'):
        if name not in state or type(state[name]) is not int or state[name] < 0:
            raise ValueError(f'Stage10 checkpoint requires explicit nonnegative integer execution state: {name}')
    if state['decision'] > state['attempt']:
        raise ValueError('Stage10 decision count cannot exceed generation attempts')
    _positive(state.get('latency_budget_s'), 'checkpoint latency_budget_s')


def capture_execution_state(env):
    """初始校准后及每轮发布前共用的执行计数快照，不保存PhysX内部状态。"""
    result = {name:getattr(env,name) for name in ('decision', 'attempt', 'episode_count', 'latency_budget_s')}
    clock = getattr(env, 'deployment_clock', None)
    if clock is not None: result['deployment_profile_sha256'] = clock.sha256
    validate_execution_state(result)
    return result


def restore_execution_state(env, state, *, spent_generations=None):
    """恢复显式种子计数；失败尝试已消耗的generation编号仍不复用。"""
    validate_execution_state(state)
    if spent_generations is not None and (type(spent_generations) is not int or spent_generations < 0):
        raise ValueError('Spent generation counter must be a nonnegative integer')
    env.policy_version, env.iteration = state['policy_version'], state['iteration']
    env.decision, env.episode_count = state['decision'], state['episode_count']
    env.attempt = max(state['attempt'], state['attempt'] if spent_generations is None else spent_generations)
    clock = getattr(env, 'deployment_clock', None)
    if state.get('deployment_profile_sha256') != (None if clock is None else clock.sha256):
        raise ValueError('Checkpoint deployment clock/profile differs from this run')
    if clock is not None and state['latency_budget_s'] != clock.budget_seconds:
        raise ValueError('Checkpoint changed modeled prefix budget')
    env.latency_budget_s = state['latency_budget_s']


def load_stage10_checkpoint(path, **kwargs):
    """在原完整恢复前先检查执行计数；公共Stage9 checkpoint读取契约保持原样。"""
    payload = torch.load(path, map_location='cpu', weights_only=False, mmap=True)
    validate_execution_state(payload.get('state', {}))
    del payload
    return load_checkpoint(path, **kwargs)


def initialize_stage9_weights(path, actor, critic, expected):
    """明确的跨阶段权重初始化：验证核心契约，不恢复旧优化器、200首采样器或计数。"""
    saved = torch.load(path, map_location='cpu', weights_only=False, mmap=True)
    old = saved.get('identity', {})
    if saved.get('version') != CHECKPOINT_VERSION or 'stage10' in old:
        raise ValueError('Initialization requires a verified Stage9 full-state checkpoint')
    for key in ('assets', 'actor_interface', 'reward', 'environment', 'termination'):
        if old.get(key) != expected[key]:
            raise ValueError(f'Stage9 weight migration changes {key}')
    stochastic = expected['training_contract']
    if old.get('sampler') != {k: stochastic[v] for k, v in
            (('steps', 'denoising_steps'), ('eta', 'eta'), ('std_floor', 'std_floor'), ('guidance_scale', 'guidance_scale'))}:
        raise ValueError('Stage9 weight migration changes the stochastic kernel')
    _validate_model_state(actor, saved['actor'], 'Actor')
    _validate_model_state(critic, saved['critic'], 'Critic')
    actor.load_state_dict(saved['actor'], strict=True)
    critic.load_state_dict(saved['critic'], strict=True)
    return dict(mode='stage9_actor_critic_weights_only', path=str(Path(path).resolve()), sha256=sha256_file(path),
                source_iteration=saved['state']['iteration'], optimizer_restored=False,
                sampler_restored=False, global_step_restored=False)


def validate_resume_budget(saved, live):
    from gem.closedloop.dppo.run_management import validate_budget_progress
    return validate_budget_progress(saved, live)


def collect_rollout(env, sampler, count, writer, *, check_disk=None, value_snapshot=None, batch_value_snapshot=None):
    """完整train池采集；仅自然终止、行政窗口及整批边界截断，不含首三曲特例。"""
    buffer = RolloutBuffer(count)
    tasks, sources = [], {}
    while len(buffer) < count:
        task = sampler.next_task()
        sample = task['sample']
        if sample['row']['split'] != 'train':
            raise ValueError('Evaluation samples cannot enter training rollouts')
        env.reset_task(sample, task['music'], seed=env.config['stage9']['seed'] + env.episode_count,
                       phase='train', music_start_frame=task['music_start_frame'])
        tasks.append(dict(dataset=sample['dataset'], sample_id=sample['row']['sample_id'],
                          music_start_frame=task['music_start_frame']))
        while len(buffer) < count:
            if check_disk is not None:
                check_disk()
            transition = env.step()
            if not transition.transition_valid or transition.identity['policy_version'] != env.policy_version:
                raise ValueError('Training requires valid data from the current policy version')
            if len(buffer) + 1 == count and not transition.terminated:
                transition.truncated = True
                transition.reason = transition.reason or 'rollout_boundary'
            transition.metadata['training_task'] = dict(dataset=sample['dataset'],
                sample_id=sample['row']['sample_id'], split='train', music_start_frame=task['music_start_frame'],
                manifest_sha256=sample['manifest_sha256'])
            if value_snapshot is not None:
                value_snapshot(transition)
            buffer.append(transition)
            if batch_value_snapshot is None:
                writer.append(buffer.transitions[-1])
            sampler.record_execution(task, transition.executed_control_steps)
            source = sample['dataset']
            item = sources.setdefault(source, dict(transitions=0, control_steps=0))
            item['transitions'] += 1
            item['control_steps'] += transition.executed_control_steps
            print(f'[COLLECT] iteration={env.iteration+1} {len(buffer)}/{count} {source} '
                  f'm={transition.executed_control_steps} start={task["music_start_frame"]}', flush=True)
            if transition.terminated or transition.truncated:
                break
    if batch_value_snapshot is not None:
        batch_value_snapshot(buffer.transitions)
        for row in buffer.transitions:
            writer.append(row)
    manifest = writer.finish()
    return buffer, dict(full_train_pool=True, transition_count=len(buffer),
        control_steps=sum(t.executed_control_steps for t in buffer.transitions), source_counts=sources,
        tasks=tasks, coverage=sampler.coverage(), rollout_manifest=str(manifest))


def _relative(path, root):
    return str(Path(path).resolve().relative_to(Path(root).resolve()))


def _assert_frozen(result):
    if result.get('policy_unchanged') is not True or result.get('runtime_parameters_unchanged') is not True:
        raise RuntimeError('Frozen GMT policy or runtime parameters changed')
    journal = result.get('execution_journal', {})
    if journal.get('executed_seq') != journal.get('acked_seq'):
        raise RuntimeError('Checkpoint boundary contains an unacknowledged physical mutation')


def calibrate(env, catalog, output, warmup, samples):
    """独立校准任务不消费训练采样器；只用于确定latency预算，不能冒称硬实时。"""
    source = next(iter(catalog.samples['train']))
    sample = catalog.samples['train'][source][0]
    env.reset_task(sample, catalog.load_music(sample), seed=env.config['stage9']['seed'], phase='calibration')
    durations, commit_durations = [], []
    timing_v2 = env.config['runtime'].get('timing_contract') == 'deployment_critical.v2'
    for i in range(warmup + samples):
        generated = env.generate()
        if generated['rejection'] or generated['prepared'] is None:
            raise RuntimeError('Stochastic calibration rejected a reference')
        started = time.perf_counter()
        env.backend.call('commit_plan', prepared_plan_id=generated['prepared']['prepared_plan_id'],
                         expected_control_tick=env.snapshot['tick'])
        commit_seconds = time.perf_counter() - started
        elapsed = generated['critical_ready_seconds'] if timing_v2 else generated['elapsed'] + commit_seconds
        env.snapshot = env.backend.call('snapshot')
        env.decision += 1
        if i >= warmup:
            durations.append(elapsed)
            commit_durations.append(commit_seconds)
        print(f'[CALIBRATION] {i+1}/{warmup+samples} {elapsed:.4f}s', flush=True)
    peak = max(durations)
    env.latency_budget_s = math.ceil((peak + max(.04, .25*peak))*50)/50
    report = dict(durations=durations, commit_seconds=commit_durations, latency_budget_s=env.latency_budget_s,
                  timing_contract=env.config['runtime'].get('timing_contract', 'legacy_audit_inclusive.v1'),
                  scope='deployment_critical_ready' if timing_v2 else 'measured_generation_prepare_commit; not hard realtime',
                  source='full_train_catalog')
    atomic_json(output / 'calibration.json', report)
    return report


def main(argv=None, *, learner=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=ROOT/'configs/closedloop/stage10_prepare_server1.yaml')
    parser.add_argument('--mode', choices=('preflight', 'train', 'eval'), default='preflight')
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--resume', help='Same Stage10 run: latest or an exact complete checkpoint path')
    parser.add_argument('--checkpoint', type=Path, help='Required model to evaluate; never selects Stage1 implicitly')
    parser.add_argument('--initialize-stage9', type=Path, help='Explicit Actor/Critic weights-only migration for a new run')
    parser.add_argument('--extend-budget-reason', help='Explicit same-run resume budget extension reason; used counts never reset')
    parser.add_argument('--stop-after-iteration', type=int, help='Stop at this accepted logical iteration boundary')
    parser.add_argument('--eval-count', help='all or count from the complete held-out catalog')
    parser.add_argument('--eval-split', choices=('val', 'test'))
    args = parser.parse_args(argv)
    if learner is not None and (args.mode != 'train' or args.initialize_stage9):
        parser.error('Distributed training supports new runs and full-state resume only')
    if args.resume and args.mode != 'train':
        parser.error('--resume is only for continued train mode')
    if bool(args.checkpoint) != (args.mode == 'eval'):
        parser.error('eval requires --checkpoint; other modes cannot use it')
    if args.initialize_stage9 and (args.mode != 'train' or args.resume):
        parser.error('--initialize-stage9 is only for new training runs')
    if args.extend_budget_reason is not None and (args.mode != 'train' or args.resume != 'latest'
                                                  or not args.extend_budget_reason.strip()):
        parser.error('--extend-budget-reason requires train --resume latest and a nonempty reason')
    if (args.eval_count is not None or args.eval_split is not None) and args.mode != 'eval':
        parser.error('Evaluation overrides are only valid in eval mode')
    config = configuration(args.config)
    if config['stage10'].get('distributed') and learner is None and args.mode == 'train':
        parser.error('Use train_closedloop_stage10_8gpu.py for distributed configurations')
    stage, s = config['stage10'], config['stage9']
    storage = stage['storage']
    output = args.output_dir.resolve()
    stop_at = stage['limits']['accepted_iterations'] if args.stop_after_iteration is None else args.stop_after_iteration
    if not 1 <= stop_at <= stage['limits']['accepted_iterations']:
        parser.error('stop iteration must fit the explicit run budget')
    manager = RunManager(output, resume=bool(args.resume), min_free_bytes=storage['min_free_bytes'],
                         max_run_bytes=storage['max_run_bytes'])
    try:
        session = output / 'sessions' / manager.session_id
        session.mkdir(parents=True, exist_ok=False)
        config['stage9']['run_id'] = manager.run_id
        config['output_root'] = str(output)
        config_path = session/'resolved_config.yaml'
        config_path.write_text(yaml.safe_dump(config, allow_unicode=True, sort_keys=False))
        budget = manager.budget(stage['limits'], for_extension=args.extend_budget_reason is not None)
        maintenance = LongRunMaintenance(manager, stage)
    except BaseException:
        manager.close()
        raise
    report = dict(schema=VERSION, mode=args.mode, status='running', session_id=manager.session_id,
                  iterations=[], evaluations=[], initialization=None, resume=None, budget_extension=None,
                  input_config_path=str(args.config.resolve()), input_config_sha256=sha256_file(args.config),
                  resolved_config_sha256=sha256_file(config_path), long_run_policy=maintenance.policy)
    workers = journal = backend = None
    provenance = check = None
    code = 0
    stop = StopSignal()
    try:
        stop.install()
        random.seed(stage['seed']); np.random.seed(stage['seed']); torch.manual_seed(stage['seed'])
        torch.set_num_threads(config['runtime']['torch_threads'])
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        check = (runtime_preflight(config, check_gpu=args.mode != 'preflight') if learner is None
                 else learner.preflight(config, check_gpu=True))
        atomic_json(session/'preflight.json', check)
        if not check['ready']:
            raise RuntimeError('Runtime assets did not pass preflight')
        catalog = FullMusicCatalog(config['paths']['data_root'])
        audited = 0
        def data_progress(record):
            nonlocal audited
            audited += 1
            if audited % 100 == 0:
                print(f'[DATA_AUDIT] {audited} {record["dataset"]}/{record["split"]}', flush=True)
        data_audit = catalog.audit_files(data_progress, require_audio=stage['dataset']['require_audio'])
        atomic_json(session/'data_audit.json', data_audit)
        report['data_audit'] = _relative(session/'data_audit.json', output)
        provenance = _sources(config, check)
        atomic_json(session/'source_identity.json', provenance)
        if args.mode == 'preflight':
            report.update(status='passed', full_dataset_checked=True, no_model_loaded=True)
            raise _PreflightComplete
        actor, train_config, loading = load_actor(config)
        atomic_json(session/'actor_loading.json', loading)
        critic = UpperCritic(qpos_mean=actor.endecoder.mean, qpos_std=actor.endecoder.std,
            proprio_scales=tuple(train_config.model.proprio_scales)).to(config['runtime']['genmo_device'])
        actor_optimizer = torch.optim.AdamW(actor.parameters(), lr=s['actor_lr'], weight_decay=0.)
        critic_optimizer = torch.optim.AdamW(critic.parameters(), lr=s['critic_lr'], weight_decay=0.)
        policy = DPPODiffusionPolicy(actor, steps=s['denoising_steps'], eta=s['eta'],
                                     std_floor=s['std_floor'], guidance_scale=s['guidance_scale'],
                                     cfg_batch=s.get('cfg_batch', False), std_schedule=s.get('std_schedule'))
        sampler = FullMusicSampler(catalog, split='train', seed=stage['seed'], window_seconds=s['episode_seconds'],
            random_start=stage['dataset']['random_start'], source_probabilities=stage['dataset']['source_probabilities'])
        bc = SupervisedAnchor(config, actor, train_config)
        generator = torch.Generator().manual_seed(stage['seed']+2002)
        samplers, generators = dict(music=sampler, bc=bc), dict(critic=generator)
        expected = identity(config, check, provenance, catalog, data_audit, actor)
        state = dict(iteration=0, policy_version=0, actor_updates=0, critic_updates=0,
            buffer_size=0, pending_plan=False, selected_actor_lr=s['actor_lr'], optimizer_attempts=0,
            latency_budget_s=s['latency_budget_s'], decision=0, attempt=0, episode_count=0, session_id=manager.session_id)
        if args.initialize_stage9:
            report['initialization'] = initialize_stage9_weights(args.initialize_stage9, actor, critic, expected)
        else:
            report['initialization'] = dict(mode='stage1_actor_weights_only', critic='new', optimizer_restored=False)
        resume_path = None
        if args.resume or args.checkpoint:
            resume_path = (manager.latest_checkpoint() if args.resume == 'latest' else
                           Path(args.resume or args.checkpoint).resolve())
            if args.resume:
                allowed = manager.latest_checkpoint() if (output/'latest.json').exists() else output/'checkpoints/initial.pt'
                if resume_path.resolve() != allowed.resolve():
                    raise ValueError('Training resume must use the latest published checkpoint of this run')
            weights_eval = args.mode == 'eval' and stage['version'] == VERSION_V2
            if weights_eval:
                from gem.closedloop.dppo.checkpoint import load_weights_checkpoint
                loaded = load_weights_checkpoint(resume_path, actor=actor, critic=critic, identity=expected)
                state.update(loaded['state'])
                # 独立评估从新执行计数开始，仅沿用训练时已定义的延迟预算与计算方式。
                state.update(decision=0, attempt=0, episode_count=0,
                    latency_budget_s=loaded['rank_execution_states'][0]['latency_budget_s'])
                policy.cfg_batch = state['execution_profile']['cfg_batch']
            else:
                state = load_stage10_checkpoint(resume_path, actor=actor, critic=critic,
                    actor_optimizer=actor_optimizer, critic_optimizer=critic_optimizer,
                    identity=expected, samplers=samplers, generators=generators)
            if args.resume:
                validate_resume_budget(state['budget'], budget.state_dict())
            if any(group['lr'] != state['selected_actor_lr'] for group in actor_optimizer.param_groups):
                raise RuntimeError('Restored optimizer learning rate differs from accepted state')
            report['resume'] = dict(checkpoint=str(resume_path), sha256=sha256_file(resume_path),
                restored_full_state=not weights_eval, restore_mode='weights_only' if weights_eval else 'full_state',
                initial_iteration=state['iteration'], old_buffer_discarded=True,
                actor_optimizer_lrs=[g['lr'] for g in actor_optimizer.param_groups],
                critic_optimizer_lrs=[g['lr'] for g in critic_optimizer.param_groups],
                training_resume=bool(args.resume))
        report['initial_iteration'] = state['iteration']
        if args.mode == 'train' and state['iteration'] >= stop_at:
            raise ValueError('Requested stop iteration must exceed the restored iteration')
        if learner is not None:
            learner.attach(config, actor, critic, actor_optimizer, critic_optimizer, policy)
        if args.resume:
            previous = json.loads((output/'run.json').read_text())
            if previous['identity'] != expected:
                raise ValueError('Run identity differs from the restored training configuration')
        else:
            atomic_json(output/'run.json', dict(schema='genmo.closedloop.stage10.run.v1', identity=expected,
                config_sha256=sha256_file(args.config), dataset_identity=catalog.identity,
                initialization=report['initialization'], mode=args.mode))
        if args.extend_budget_reason is not None:
            report['budget_extension'] = budget.extend_limits(stage['limits'], reason=args.extend_budget_reason.strip(),
                checkpoint_sha256=report['resume']['sha256'], config_sha256=sha256_file(args.config))
        workers = Workers(config, session)
        socket = Path(workers.temp.name)/'gmt.sock'
        client = workers.start('gmt', [config['paths']['isaac_python'], '-B',
            str(Path(config['paths']['gmt_repo'])/'scripts/rsl_rl/serve_frozen_gmt.py'),
            '--config', str(config_path), '--socket', str(socket), '--headless'], config['paths']['gmt_repo'], socket)
        journal = GuardedStepJournal(session/'bootstrap_journal.sqlite', manager.disk_guard)
        backend = AcknowledgedBackend(client, journal, socket_path=socket, timeout_s=config['runtime']['rpc_timeout_s'])
        builder = OnlineConditionBuilder(BumiMotionFeatureCodec(BumiKinematics(config['paths']['kinematics'])))
        env = UpperEnvironment(config, backend, builder, policy, budget, session/'bootstrap')
        env.disk_guard = manager.disk_guard
        restore_execution_state(env, state, spent_generations=budget.state_dict()['used']['generations'])
        if report['resume'] is not None:
            report['resume']['new_backend_session_id'] = backend.session_id
        else:
            report['calibration'] = calibrate(env, catalog, session,
                config['timing']['calibration_warmup'], config['timing']['calibration_samples'])
        manager.append_metrics(dict(event='session_start', mode=args.mode, initial_iteration=state['iteration'],
                                    resume=report['resume'], backend_session_id=backend.session_id))
        if args.mode == 'eval':
            evaluation = stage['evaluation']
            count = args.eval_count or evaluation['eval_count']
            count = count if count == 'all' else int(count)
            plan = build_evaluation_tasks(catalog, split=args.eval_split or evaluation['split'], eval_count=count,
                seeds=evaluation['seeds'], selection_seed=evaluation['selection_seed'],
                start_mode=evaluation['start_mode'], episode_seconds=evaluation['episode_seconds'])
            before = dict(actor=_fingerprint(actor), critic=_fingerprint(critic))
            env.output = session/'evaluation'
            env.output.joinpath('raw_samples').mkdir(parents=True)
            result = evaluate_policy(env, policy, plan, session/'evaluation_report', catalog=catalog,
                episode_seconds=evaluation['episode_seconds'], actor_identity=dict(checkpoint=str(resume_path),
                    sha256=sha256_file(resume_path), iteration=state['iteration']), frozen_modules={'critic':critic},
                progress=lambda item: print('[EVAL] '+str(item), flush=True))
            if before != dict(actor=_fingerprint(actor), critic=_fingerprint(critic)):
                raise RuntimeError('Read-only evaluation changed a trainable network')
            report['evaluations'].append(_relative(session/'evaluation_report/report.json', output))
            report['evaluation_status'] = result.get('status')
            manager.check_disk(refresh=True)
        else:
            # 保存迭代零的完整状态，首次候选失败也有明确的新任务恢复边界。
            if not args.resume:
                state.update(budget=budget.state_dict(), **capture_execution_state(env))
                initial = output/'checkpoints'/'initial.pt'
                manager.check_disk(storage['checkpoint_reserve_bytes'], refresh=True)
                save_checkpoint(initial, actor=actor, critic=critic, actor_optimizer=actor_optimizer,
                    critic_optimizer=critic_optimizer, state=state, identity=expected, config=config,
                    samplers=samplers, generators=generators)
                manager.disk_guard.account_file(initial)
            while state['iteration'] < stop_at and not stop.stop_requested:
                if maintenance.expired():
                    report['stop_reason'] = 'walltime_limit'
                    break
                capacity = budget.iteration_capacity(len(s['actor_lr_candidates']))
                if not capacity['can_start']:
                    report.update(stop_reason='budget_exhausted', budget_stop_details=capacity)
                    break
                index = state['iteration'] + 1
                iteration_output = session/'iterations'/f'{index:06d}'
                iteration_output.mkdir(parents=True, exist_ok=False)
                manager.check_disk(storage['checkpoint_reserve_bytes'], refresh=True)
                journal.close()
                journal = GuardedStepJournal(iteration_output/'execution_journal.sqlite', manager.disk_guard)
                backend.journal = journal
                env.output = iteration_output
                env.output.joinpath('raw_samples').mkdir()
                writer = RolloutWriter(iteration_output/'rollout', policy_version=env.policy_version,
                    chunk_size=storage['rollout_chunk_size'], disk_guard=manager.disk_guard)
                step_report = dict(iteration=index, policy_version_before=state['policy_version'], status='running',
                                   kl_limit=s['kl_stop_joint'])
                published = False
                try:
                    buffer, collection = collect_rollout(env, sampler, s['rollout_upper_steps'], writer,
                        check_disk=manager.check_disk, value_snapshot=lambda row:populate_values(
                            [row],critic,config['runtime']['genmo_device']))
                    step_report.update(collection=collection, collected_upper_transitions=len(buffer),
                        rollout_manifest=_relative(collection['rollout_manifest'], output),
                        execution_journal=_relative(journal.path, output))
                    step_report['probability_check'] = probability_check(policy, buffer.transitions, clip=s['ppo_clip'])
                    targets = fixed_targets(buffer.transitions, critic, config['runtime']['genmo_device'],
                                             gamma_upper=s['gamma_upper'], lambda_upper=s['lambda_upper'])
                    targets.update(old_values=torch.tensor([t.old_value for t in buffer.transitions], dtype=torch.float64),
                                   next_values=torch.tensor([t.next_value for t in buffer.transitions], dtype=torch.float64))
                    targets_path = iteration_output/'fixed_targets.pt'
                    torch.save(targets, targets_path)
                    step_report['targets_path'] = _relative(targets_path, output)
                    initial_actor = _fingerprint(actor)
                    step_report['critic'] = (critic_update if learner is None else learner.critic_update)(critic, critic_optimizer, buffer.transitions, targets,
                        steps=s['critic_steps'], batch_size=s['critic_batch'], generator=generator,
                        grad_clip_norm=s['critic_grad_clip_norm'])
                    step_report['critic']['actor_unchanged'] = initial_actor == _fingerprint(actor)
                    if not step_report['critic']['actor_unchanged']:
                        raise RuntimeError('Critic value loss changed Actor')
                    initial_critic = _fingerprint(critic)
                    events = []
                    def progress(event):
                        events.append(event)
                        atomic_json(iteration_output/'lr_progress.json', dict(events=events))
                    step_report['actor'] = (actor_update if learner is None else learner.actor_update)(policy, actor_optimizer, buffer.transitions, targets,
                        bc=bc, bc_weight=s['bc_weight'], clip=s['ppo_clip'], gamma_denoising=s['gamma_denoising'],
                        grad_clip_norm=s['grad_clip_norm'], learning_rate_candidates=s['actor_lr_candidates'],
                        kl_limit=s['kl_stop_joint'], reserve_attempt=lambda:budget.reserve('update',optimizer_attempts=1),
                        calibration_progress=progress)
                    step_report['actor']['critic_unchanged'] = initial_critic == _fingerprint(critic)
                    step_report['kl'] = (analytic_kl if learner is None else learner.analytic_kl)(policy, buffer.transitions)
                    if not step_report['actor']['critic_unchanged'] or step_report['kl']['mean_joint_kl'] > s['kl_stop_joint']:
                        raise RuntimeError('Actor isolation or accepted KL verification failed')
                    step_report['gmt_frozen'] = backend.call('verify_frozen')
                    _assert_frozen(step_report['gmt_frozen'])
                    step_report['source_unchanged'] = verify_source_provenance(provenance)
                    if not step_report['source_unchanged']['unchanged']:
                        raise RuntimeError('Runtime source changed during training')
                    if any(sha256_file(config['paths'][key]) != digest for key,digest in check['asset_sha256'].items()):
                        raise RuntimeError('Original assets changed during training')
                    buffer.clear()
                    budget.accept_iteration()
                    state.update(iteration=index, policy_version=state['policy_version']+1,
                        actor_updates=state['actor_updates']+1, critic_updates=state['critic_updates']+s['critic_steps'],
                        selected_actor_lr=float(actor_optimizer.param_groups[0]['lr']),
                        optimizer_attempts=budget.state_dict()['used']['optimizer_attempts'],
                        budget=budget.state_dict(), **capture_execution_state(env),
                        session_id=manager.session_id, buffer_size=0, pending_plan=False)
                    env.policy_version, env.iteration = state['policy_version'], state['iteration']
                    checkpoint = output/'checkpoints'/f'stage10_{index:06d}_{manager.session_id}.pt'
                    manager.check_disk(storage['checkpoint_reserve_bytes'], refresh=True)
                    save_checkpoint(checkpoint, actor=actor, critic=critic, actor_optimizer=actor_optimizer,
                        critic_optimizer=critic_optimizer, state=state, identity=expected, config=config,
                        samplers=samplers, generators=generators)
                    step_report.update(status='accepted', policy_version_after=state['policy_version'],
                        checkpoint=_relative(checkpoint, output), budget=budget.state_dict())
                    atomic_json(iteration_output/'summary.json', step_report)
                    manager.publish_checkpoint(index, checkpoint, metadata=dict(
                        iteration_summary=_relative(iteration_output/'summary.json', output)))
                    published = True
                    report['iterations'].append(_relative(iteration_output/'summary.json', output))
                    manager.append_metrics(dict(event='iteration_accepted', iteration=index,
                        summary=report['iterations'][-1], mean_joint_kl=step_report['kl']['mean_joint_kl'],
                        actor_lr=state['selected_actor_lr'], critic_mse=step_report['critic']['mse']))
                    print(f'[ACCEPTED] iteration={index} lr={state["selected_actor_lr"]} '
                          f'KL={step_report["kl"]["mean_joint_kl"]:.8g}', flush=True)
                    # 只有已发布 checkpoint 的完整轮次可进入归档/保留；SQLite 必须先关闭。
                    journal.close()
                    archive = maintenance.archive_iteration(iteration_output)
                    retired = maintenance.prune_checkpoints()
                    if archive is not None:
                        print(f'[ARCHIVED] iteration={index} bytes={archive["archive_size_bytes"]} '
                              f'retired_checkpoints={len(retired)}', flush=True)
                except BaseException as exc:
                    if not published:
                        step_report.update(status='failed', error=dict(type=type(exc).__name__, message=str(exc)),
                            continuation='stop_session; resume_last_published_complete_checkpoint; discard_incomplete_rollout')
                        if hasattr(exc, 'lr_calibration_report'):
                            step_report['lr_calibration_failure'] = exc.lr_calibration_report
                        atomic_json(iteration_output/'summary.json', step_report)
                    raise
        report.update(status='passed', final_iteration=state['iteration'], stopped_on_signal=stop.stop_requested,
                      full_dataset_checked=True, sampling_coverage=sampler.coverage(),
                      stop_reason=report.get('stop_reason', 'signal' if stop.stop_requested else
                                             ('evaluation_complete' if args.mode == 'eval' else 'iteration_target')))
    except _PreflightComplete:
        pass
    except BaseException as exc:
        code = 1
        report.update(status='failed', error=dict(type=type(exc).__name__, message=str(exc), traceback=traceback.format_exc()))
        traceback.print_exc()
    finally:
        try:
            if workers is not None:
                if backend is not None:
                    workers.entries[-1]['client'] = backend.client
                workers.close()
                report['worker_shutdown'] = workers.shutdown
                gmt = workers.shutdown.get('gmt', {})
                if (gmt.get('close_error') or gmt.get('forced_shutdown') or gmt.get('process_exit_code') != 0
                        or gmt.get('policy_unchanged') is not True or gmt.get('runtime_parameters_unchanged') is not True):
                    report['status'], code = 'failed', 1
            if journal is not None:
                journal.close()
            if provenance is not None:
                report['source_unchanged'] = verify_source_provenance(provenance)
                if not report['source_unchanged']['unchanged']:
                    report['status'], code = 'failed', 1
            if check is not None:
                report['original_assets_unchanged'] = all(sha256_file(config['paths'][key]) == digest
                                                         for key,digest in check['asset_sha256'].items())
                if not report['original_assets_unchanged']:
                    report['status'], code = 'failed', 1
            report.update(exit_code=code, budget=budget.state_dict())
            if learner is not None:
                report['distributed_checks'] = learner.evidence
            atomic_json(session/'summary.json', report)
            manager.append_metrics(dict(event='session_end', status=report['status'], exit_code=code,
                                        summary=_relative(session/'summary.json', output)))
            atomic_json(session/'completion.json', dict(
                schema='genmo.closedloop.stage10.session_completion.v1',
                summary_sha256=sha256_file(session/'summary.json'), status=report['status'], exit_code=code))
        finally:
            stop.restore()
            manager.close()
        print(json.dumps(dict(status=report['status'], output=str(session), exit_code=code)), flush=True)
    return code


if __name__ == '__main__':
    raise SystemExit(main())
