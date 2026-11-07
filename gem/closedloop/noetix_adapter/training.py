"""在用户指定noetix实现上训练冻结底层Actor之前的音乐参考Adapter。

先校验ONNX内嵌源PT哈希、机器人控制契约及66组CPU数值等价，再严格加载同一
PT的Actor和WM。真实环境、关节顺序、执行器延迟/滤波/PD、奖励、终止与WM监督
直接使用该源码。Actor和输入归一化永久冻结，Adapter通过Actor输入与WM反传。

每组保留原64首GENMO训练参考，2048环境、约百万转移Replay、H8×10次WM更新、
预热轮数和Critic由实验配置控制；Critic可载入同源网络并独立真实TD预热。
真实与想象奖励
都针对未修正名义动作，避免Adapter改目标给自己打分。旧Goal570的标定权重不迁移。
保存完整WM/Adapter/Critic/三优化器/Replay/RNG/采样器；恢复时新开物理episode。
状态、实际Adam次数、源模型身份及训练指标持续落盘；有限更新不代表物理质量改善。
"""

from dataclasses import asdict, is_dataclass
import copy
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import torch

from .data import digest
from .model import ReferenceLibrary, ReferencePolicy


def fill_config(instance, values):
    for key, value in values.items():
        old = getattr(instance, key)
        if is_dataclass(old):
            fill_config(old, value)
        else:
            setattr(instance, key, value)
    return instance


def verify_actor(settings, source):
    """检查实际ONNX与可微Actor一致；不根据相似文件名猜测模型。"""
    import onnxruntime as ort
    path = Path(source) / 'scripts/export_policy_onnx.py'
    spec = importlib.util.spec_from_file_location('noetix_export_verification', path)
    exporter = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(exporter)
    ck, actor, checkpoint_sha = exporter.load_policy(Path(settings['checkpoint']))
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    session = ort.InferenceSession(settings['onnx'], options, providers=['CPUExecutionProvider'])
    meta = json.loads(session.get_modelmeta().custom_metadata_map['noetix_supermimic.policy'])
    if meta['checkpoint_sha256'] != checkpoint_sha:
        raise ValueError('指定ONNX不是此PT导出的Actor')
    if digest(settings['onnx']) != settings['onnx_sha256'] or checkpoint_sha != settings['checkpoint_sha256']:
        raise ValueError('模型内容与实验记录不一致')
    if {x.name: x.shape for x in session.get_inputs()} != {'obs_hist': [1, 750], 'goal': [1, 606]}:
        raise ValueError('需要指定的750/606 BUMI4340 Actor')
    errors = []
    with torch.no_grad():
        for label, inputs in exporter.samples(actor, 64, 40000):
            actual = session.run(None, dict(zip(('obs_hist', 'goal'), [x.numpy() for x in inputs])))[0]
            expected = actor(*inputs).numpy()
            np.testing.assert_allclose(actual, expected, atol=1e-4, rtol=1e-4, err_msg=label)
            errors.append(float(np.abs(actual - expected).max()))
    return ck, dict(cases=len(errors), max_abs_action_error=max(errors),
                    onnx_sha256=settings['onnx_sha256'], checkpoint_sha256=checkpoint_sha,
                    actor_parameters_and_normalizers='frozen', status='PASS')


def make_config(saved, settings):
    from supermimic.config import TrainConfig
    saved = copy.deepcopy(saved)
    # 指定40k保存时尚有此采样器等待字段；当前源码已删除，且本实验关闭动作退休。
    saved.get('env', {}).get('sampler', {}).pop('patience_iterations', None)
    cfg = fill_config(TrainConfig(), saved)
    # 当前源码新增的推搡/质量/PD扰动不在指定24k的配置里，不能因默认值而悄悄启用。
    if hasattr(cfg.env, 'perturbations') and 'perturbations' not in saved.get('env', {}):
        cfg.env.perturbations.torso_mass_range = (1., 1.)
        cfg.env.perturbations.kp_range = cfg.env.perturbations.kd_range = (1., 1.)
        cfg.env.perturbations.push_enabled = False
        cfg.env.perturbations.joint_velocity_noise = 0.
    cfg.run_name = settings['run_name']
    cfg.log_root = Path(settings['root'])
    cfg.seed = settings['seed']
    cfg.device = 'cuda:0'
    cfg.resume = None
    cfg.amp = 'none'
    # 本入口逐轮读取真实episode并自行原子保存完整包，不进入原生延迟统计/异步保存循环。
    if hasattr(cfg, 'defer_metrics'):
        cfg.defer_metrics = False
        cfg.async_checkpoint = False
    cfg.data.library = Path(settings['library'])
    cfg.data.npz_dir = cfg.data.library.parent
    cfg.data.manifest = cfg.data.clips = cfg.data.limit = None
    cfg.env.num_envs = settings.get('num_envs', 2048)
    cfg.env.replay_capacity = settings.get('replay_steps', 512)
    cfg.env.action_noise_std = 0.
    cfg.env.obs_noise = False
    cfg.env.sampling_mode = 'uniform'
    cfg.env.sampler.retire_enabled = False
    cfg.env.sampler.start_margin = max(100, settings['horizon'] + 10)
    cfg.env.compact_library = True
    cfg.wm_trainer.lr = 3e-5
    cfg.wm_trainer.window = 8
    cfg.wm_trainer.batch_size = settings.get('wm_batch', 2048)
    cfg.wm_trainer.batches_per_update = settings.get('wm_batches', 10)
    cfg.wm_trainer.compile = settings.get('compile', 'default')
    cfg.policy_trainer.horizon = settings['horizon']
    cfg.policy_trainer.batch_size = settings.get('actor_batch', 1024)
    cfg.policy_trainer.microbatch_size = settings.get('microbatch', 256)
    cfg.policy_trainer.lr = settings['adapter_lr']
    cfg.policy_trainer.use_critic = settings.get('use_critic', False)
    cfg.policy_trainer.critic_lr = settings.get('critic_lr', 3e-5)
    cfg.policy_trainer.critic_epochs = settings.get('critic_epochs', 2)
    cfg.policy_trainer.critic_minibatches = settings.get('critic_minibatches', 4)
    cfg.policy_trainer.obs_noise = False
    cfg.policy_trainer.start_margin = settings['horizon'] + 10
    cfg.policy_trainer.compile = settings.get('compile', 'default')
    # 加速版的compact参考预计算直接编码MotionLibrary，绕过ReferenceLibrary的
    # 音乐和12个姿态支持点。Adapter必须逐步走扩展goal_window；其余加速保持。
    if hasattr(cfg.policy_trainer, 'prepare_references'):
        cfg.policy_trainer.prepare_references = False
        cfg.policy_trainer.reference_compile = None
    cfg.schedule.warmup_env_steps = settings.get('warmup_env_steps', 64)
    cfg.schedule.wm_warmup_iterations = settings.get('wm_warmup', 100)
    cfg.schedule.updates_per_env_step = .4
    cfg.schedule.total_iterations = settings['target_updates'] + cfg.schedule.wm_warmup_iterations
    cfg.checkpoint_mode = 'full'
    return cfg


def run(settings, source):
    from supermimic.training.loop import Trainer
    from supermimic.training.checkpoints import atomic_json_save, atomic_torch_save, capture_rng_state, restore_rng_state
    from .critic import AdapterPolicyTrainer
    from supermimic.robots import get_robot_spec

    if torch.cuda.device_count() != 1 or int(os.environ.get('WORLD_SIZE', 1)) != 1:
        raise ValueError('各组为独立单卡实验，每组必须只可见一张GPU')
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    ck, verification = verify_actor(settings, source)
    cfg = make_config(ck['config'], settings)

    class AdapterTrainer(Trainer):
        @torch.no_grad()
        def act(self, obs):
            packed = self.adapter_lib.goal_window(self.command.frame, 10)
            packed = torch.cat((packed, obs['goal'][:, -6:]), dim=-1)
            return self.policy(obs['policy'], packed)

    trainer = AdapterTrainer(cfg)
    output = trainer.log_dir
    trainer.policy.load_state_dict(ck['policy'], strict=True)
    trainer.wm.load_state_dict(ck['wm'], strict=True)
    trainer.critic.load_state_dict(ck['critic'], strict=True)
    # 所有优化器从零开始；Actor、WM及Critic网络使用同一个指定底座。
    torch.manual_seed(cfg.seed)
    trainer.policy = ReferencePolicy(trainer.policy, get_robot_spec('bumi4340').joint_names).to(cfg.device)
    packed_library = torch.load(cfg.data.library, map_location='cpu', weights_only=False)
    trainer.adapter_lib = ReferenceLibrary(trainer.lib, packed_library['music'])
    trainer.policy_trainer = AdapterPolicyTrainer(trainer.wm, trainer.policy, trainer.critic,
                                          trainer.adapter_lib, cfg.policy_trainer)
    trainer.policy_trainer.opt = torch.optim.Adam(trainer.policy.net.parameters(),
                                                 lr=settings['adapter_lr'], betas=cfg.policy_trainer.betas, fused=True)
    frozen = {k: v.detach().cpu().clone() for k, v in trainer.policy.actor.state_dict().items()}
    initial_adapter = {k: v.detach().cpu().clone() for k, v in trainer.policy.net.state_dict().items()}
    atomic_json_save(verification, output / 'actor_onnx_parity.json')
    atomic_json_save(settings | dict(source=source, pid=os.getpid(),
        cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),
        input_dimension=1948, residual_shape=[10, 27], actor_input=[750, 606],
        algorithm='noetix_music_reference_adapter_WM_BPTT', critic=cfg.policy_trainer.use_critic,
        architecture='stateless_future_pose_residual_current_nominal_anchor',
        source_checkpoint_iteration=ck['iteration'], train_songs=len(packed_library['names'])), output/'adapter_manifest.json')
    (output/'pid').write_text(str(os.getpid())+'\n')
    del ck, packed_library
    start_time = time.time()
    trainer._finished_returns, trainer._finished_lens = [], []
    trainer._obs, _ = trainer.env.reset()
    trainer._t_start = start_time

    def status(phase, **fields):
        atomic_json_save(dict(phase=phase, pid=os.getpid(), gpu=os.environ.get('CUDA_VISIBLE_DEVICES'),
            adapter_updates=trainer.num_updates, wm_iterations=trainer.iteration,
            wm_optimizer_updates=trainer.iteration*cfg.wm_trainer.batches_per_update,
            critic_optimizer_updates=trainer.policy_trainer.critic_steps,
            real_critic_optimizer_updates=trainer.policy_trainer.real_critic_steps,
            env_steps=trainer.env_steps, real_transitions=trainer.env_steps*cfg.env.num_envs,
            updated_unix=time.time(), target_updates=settings['target_updates'], **fields), output/'status.json')

    def check_frozen():
        for name, value in trainer.policy.actor.state_dict().items():
            if not torch.equal(value.detach().cpu(), frozen[name]):
                raise RuntimeError(f'冻结Actor或归一化发生变化：{name}')

    def save(full=False):
        check_frozen()
        payload = dict(schema='noetix.reference_adapter.v1', settings=settings,
            config=asdict(cfg), verification=verification,
            adapter=trainer.policy.net.state_dict(), bounds=trainer.policy.bounds,
            adapter_updates=trainer.num_updates, wm_iterations=trainer.iteration,
            env_steps=trainer.env_steps, step_acc=trainer._step_acc,
            critic_steps=trainer.policy_trainer.critic_steps,
            real_critic_steps=trainer.policy_trainer.real_critic_steps,
            physics_restored=False)
        path = output/f'adapter_{trainer.num_updates:06d}.pt'
        atomic_torch_save(payload, path)
        if full:
            payload.update(wm=trainer.wm.state_dict(), critic=trainer.critic.state_dict(),
                adapter_optimizer=trainer.policy_trainer.opt.state_dict(), wm_optimizer=trainer.wm_trainer.opt.state_dict(),
                critic_optimizer=trainer.policy_trainer.critic_opt.state_dict(),
                replay=trainer.buffer.state_dict(), rng=capture_rng_state(), sampler=trainer.command.sampler.state_dict())
            path = output/'checkpoint_latest.pt'
            atomic_torch_save(payload, path)
        atomic_json_save(dict(path=str(path), adapter_updates=trainer.num_updates,
                             wm_iterations=trainer.iteration, full=full), output/'latest_checkpoint.json')

    save()
    # 零Adapter在真实状态上必须复现冻结控制器；并检查完整606维目标保持一致。
    with torch.no_grad():
        native = trainer.policy.actor(trainer._obs['policy'], trainer._obs['goal'])
        zero = trainer.act(trainer._obs)
        packed = torch.cat((trainer.adapter_lib.goal_window(trainer.command.frame, 10),
                            trainer._obs['goal'][:, -6:]), -1)
        corrected, residual = trainer.policy.references(trainer._obs['policy'], packed)
        nominal = torch.cat((packed[:, :600], packed[:, -6:]), -1)
        delta = (corrected - trainer._obs['goal']).abs()
        worst = int(delta.flatten().argmax())
        row, col = divmod(worst, delta.shape[1])
        atomic_json_save(dict(nominal_goal_max_error=float((nominal-trainer._obs['goal']).abs().max()),
            corrected_goal_max_error=float(delta.max()), residual_max=float(residual.abs().max()),
            worst_goal_index=[row, col], nominal_value=float(nominal[row, col]),
            corrected_value=float(corrected[row, col]), native_value=float(trainer._obs['goal'][row, col]),
            action_max_error=float((zero-native).abs().max())), output/'zero_reference_diagnostic.json')
        torch.testing.assert_close(zero, native, atol=2e-5, rtol=2e-5)
        verification['zero_adapter_real_batch_max_error'] = float((zero-native).abs().max())
        atomic_json_save(verification, output/'actor_onnx_parity.json')

    if settings.get('resume'):
        saved = torch.load(settings['resume'], map_location='cpu', weights_only=False)
        for key in ('onnx_sha256', 'checkpoint_sha256', 'library', 'horizon', 'adapter_lr', 'seed'):
            if saved['settings'][key] != settings[key]:
                raise ValueError(f'恢复配置不匹配：{key}')
        if saved['settings'].get('use_critic', False) != cfg.policy_trainer.use_critic:
            raise ValueError('恢复时Critic开关必须与原运行一致')
        trainer.policy.net.load_state_dict(saved['adapter'])
        trainer.wm.load_state_dict(saved['wm'])
        trainer.policy_trainer.opt.load_state_dict(saved['adapter_optimizer'])
        trainer.wm_trainer.opt.load_state_dict(saved['wm_optimizer'])
        trainer.critic.load_state_dict(saved['critic'])
        if cfg.policy_trainer.use_critic:
            trainer.policy_trainer.critic_opt.load_state_dict(saved['critic_optimizer'])
        trainer.policy_trainer.critic_steps = saved.get('critic_steps', 0)
        trainer.policy_trainer.real_critic_steps = saved.get('real_critic_steps', 0)
        trainer.command.sampler.load_state_dict(saved['sampler'])
        trainer.buffer.load_state_dict(saved['replay'])
        trainer.buffer.break_episodes()
        trainer.num_updates, trainer.iteration = saved['adapter_updates'], saved['wm_iterations']
        trainer.env_steps, trainer._step_acc = saved['env_steps'], saved['step_acc']
        restore_rng_state(saved['rng'])
        del saved
    try:
        status('real_warmup')
        for _ in range(cfg.schedule.warmup_env_steps):
            trainer._env_step()
        while trainer.num_updates < settings['target_updates']:
            tick = time.time()
            status('real_collect')
            trainer._step_acc += 1 / cfg.schedule.updates_per_env_step
            n_steps = int(trainer._step_acc)
            trainer._step_acc -= n_steps
            for _ in range(n_steps):
                trainer._env_step()
            status('WM_H8_updates')
            wm = trainer.wm_trainer.update(trainer.buffer)
            metrics = dict(wm=wm)
            critic_start = max(0, cfg.schedule.wm_warmup_iterations - settings.get('critic_warmup', 200))
            if cfg.policy_trainer.use_critic and trainer.iteration >= critic_start:
                status('Critic_real_TD')
                metrics['critic'] = trainer.policy_trainer.update_real_critic(trainer.buffer,
                    settings.get('critic_real_batch', 2048), settings.get('critic_real_batches', 2))
            if trainer.iteration >= cfg.schedule.wm_warmup_iterations:
                status('Adapter_BPTT')
                decays = sum(trainer.num_updates >= value for value in settings['lr_milestones'])
                lr = settings['adapter_lr'] * .5**decays
                trainer.policy_trainer.set_lr(lr)
                metrics['adapter'] = trainer.policy_trainer.update(trainer.buffer)
                trainer.num_updates += 1
                metrics['adapter_lr'] = lr
                metrics['critic_lr'] = trainer.policy_trainer.critic_opt.param_groups[0]['lr']
            trainer.iteration += 1
            if any(not np.isfinite(v) for group in (wm, metrics.get('adapter', {}), metrics.get('critic', {})) for v in group.values()):
                raise FloatingPointError('训练指标出现非有限值')
            metrics.update(iteration=trainer.iteration, adapter_updates=trainer.num_updates,
                env_steps=trainer.env_steps, seconds=time.time()-tick, elapsed=time.time()-start_time)
            if trainer._finished_returns:
                metrics['real_episode_count'] = sum(value.numel() for value in trainer._finished_returns)
                metrics['real_episode_return'] = float(torch.cat(trainer._finished_returns).mean())
                metrics['real_episode_length'] = float(torch.cat(trainer._finished_lens).mean())
                trainer._finished_returns, trainer._finished_lens = [], []
            with (output/'training.jsonl').open('a') as stream:
                stream.write(json.dumps(metrics, allow_nan=False)+'\n')
            for group in ('wm', 'adapter', 'critic'):
                for key, value in metrics.get(group, {}).items():
                    trainer.writer.add_scalar(f'{group}/{key}', value, trainer.iteration)
            if trainer.iteration % 10 == 0 or trainer.num_updates == 1:
                print(json.dumps(metrics, allow_nan=False), flush=True)
                trainer.writer.flush()
            if trainer.num_updates == 1:
                check_frozen()
                delta = sum(float((v.detach().cpu()-initial_adapter[k]).square().sum())
                            for k,v in trainer.policy.net.state_dict().items())**.5
                atomic_json_save(dict(adapter_parameter_change_l2=delta, frozen_actor_equal=True,
                    real_transitions=trainer.env_steps*cfg.env.num_envs, metrics=metrics), output/'first_update.json')
                save(full=True)
            elif trainer.num_updates and trainer.num_updates % settings.get('checkpoint_every', 1000) == 0:
                save(full=True)
            elif trainer.num_updates and trainer.num_updates % 100 == 0:
                save()
            if settings.get('evaluation_every') and trainer.num_updates and trainer.num_updates % settings['evaluation_every'] == 0:
                status('fixed_validation')
                eval_dir = output/'evaluations'
                eval_dir.mkdir(exist_ok=True)
                report = eval_dir/f'paired_{trainer.num_updates:06d}.json'
                with (eval_dir/f'paired_{trainer.num_updates:06d}.log').open('w') as log:
                    result = subprocess.run([sys.executable, '-B', '-u',
                        str(Path(__file__).resolve().parents[3]/'tools/train_noetix_reference_adapter.py'),
                        '--source', str(source), '--evaluate-checkpoint',
                        str(output/f'adapter_{trainer.num_updates:06d}.pt'),
                        '--evaluation-library', settings['evaluation_library'],
                        '--evaluation-output', str(report)], stdout=log, stderr=subprocess.STDOUT)
                atomic_json_save(dict(adapter_updates=trainer.num_updates, exit_code=result.returncode,
                                      report=str(report)), output/'evaluation_latest.json')
                if result.returncode == 0:
                    score = json.loads(report.read_text())['results']['adapter']
                    best_path = output/'validation_best.json'
                    previous = json.loads(best_path.read_text()) if best_path.exists() else None
                    if previous is None or (score['completed'], score['mean_full_return']) > (previous['completed'], previous['mean_full_return']):
                        atomic_json_save(dict(adapter_updates=trainer.num_updates, report=str(report),
                            checkpoint=str(output/f'adapter_{trainer.num_updates:06d}.pt'),
                            completed=score['completed'], mean_full_return=score['mean_full_return']), best_path)
                else:
                    print(f'固定验证失败，exit={result.returncode}，详见{eval_dir}', flush=True)
            if (output/'stop_requested.json').is_file():
                save(full=True)
                status('stopped_by_user')
                return
            status('running', latest_metrics=metrics)
        save(full=True)
        status('completed')
    except BaseException as error:
        status('failed', error_type=type(error).__name__, error=str(error))
        raise
    finally:
        trainer.writer.close()
