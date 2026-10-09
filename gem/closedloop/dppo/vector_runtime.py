"""把GPU多环境采集接入既有Stage10运行管理、预算、执行证据与完整断点。

一个rank只建立一个Isaac GPU世界；每个环境拥有独立音乐采样器、上层状态、日志与
单线程预算账本。根TrainingBudget先为整rank预占，再为各环境划定不能互相透支的
额度；确认所有环境结束后才汇总实际用量并结算，失败不退款。每轮封存世界及环境
两层journal，再交给原RunManager/异步归档器，不复制checkpoint或训练曲线格式。

物理世界在普通轮之间连续存在。断点只保存逻辑游标、环境身份、随机计数及已耗预算，
恢复明确开始新的物理episode。封存/换日志必须在所属线程及一致边界进行；不能在
后台压缩的同时继续改写旧目录。此模块不修改DPPO、BC、GAE、KL或全局160条的定义。
"""
from __future__ import annotations

import copy
from pathlib import Path
from types import SimpleNamespace
import time
import os
import torch
import yaml
from .budget import atomic_json
from .budget_ledger import IncrementalBudget
from .buffer import RolloutBuffer
from .dual_collector import fixed_fragment_targets
from .full_dataset import FullMusicSampler
from .parallel_support import begin_lease, finish_lease, collection_credit, local_call, build_global_manifest
from .returns import normalize_advantages_global
from .run_management import GuardedStepJournal
from .rollout_storage import BlockRolloutWriter
from .trainer import populate_values
from .vector_collector import VectorEnvironmentCollector, VectorWorldClient, VectorLaneBackend
from .vector_environment import VectorUpperEnvironment
from gem.closedloop.online_conditions import OnlineConditionBuilder
from gem.robots.bumi.feature_codec import BumiMotionFeatureCodec
from gem.robots.bumi.kinematics import BumiKinematics


class VectorTrainingRuntime:
    def __init__(self, context):
        from tools.eval.run_closedloop_baseline import Workers
        self.c = c = context
        self.iteration, self.policy_version = c.state['iteration'], c.state['policy_version']
        self.num_envs = c.config['runtime']['num_envs']
        if self.num_envs > c.settings['rollout_upper_steps_per_rank']:
            raise ValueError('Production cannot allocate more environments than real local rollout transitions')
        self.phase = None
        self.closed_journals = []
        self.collector = None
        child = copy.deepcopy(c.config)
        child['runtime'].update(physics_device='cuda:0', asset_conversion_dir=str(c.rank_dir/'usd_assets'))
        config_path = c.rank_dir/'vector_config.yaml'
        config_path.write_text(yaml.safe_dump(child, allow_unicode=True))
        c.workers = Workers(c.config, c.rank_dir)
        socket = Path(c.workers.temp.name)/'vector.sock'
        visible = os.environ['CUDA_VISIBLE_DEVICES'].split(',')[c.distributed.rank].strip()
        client = local_call(c.distributed, lambda: c.workers.start('gmt',
            [c.config['paths']['isaac_python'], '-B', str(Path(c.config['paths']['gmt_repo'])/
             'scripts/rsl_rl/serve_frozen_gmt_vector.py'), '--config', str(config_path), '--socket', str(socket), '--headless'],
            c.config['paths']['gmt_repo'], socket, environment={'CUDA_VISIBLE_DEVICES':visible}, strip_distributed=True))
        def world_factory():
            journal = self._journal(c.rank_dir/'bootstrap_world.sqlite')
            return VectorWorldClient(client, journal, socket_path=socket, timeout_s=c.config['runtime']['rpc_timeout_s'])
        c.workers.entries[0]['client'] = None
        self.collector = VectorEnvironmentCollector(c.policy, self._factory, world_factory,
            num_envs=self.num_envs, batch_wait_seconds=c.config['runtime']['vector_batch_wait_s'])

    def _journal(self, path):
        return GuardedStepJournal(path, self.c.guard, format='genmo.execution_journal.ndarray.v2')

    def _bind(self, slot, resource):
        if self.phase is None:raise RuntimeError('No pre-reserved vector resource phase')
        directory = self.phase.path/f'env{slot:03d}'
        directory.mkdir(parents=True, exist_ok=True)
        previous = getattr(resource, 'journal', None)
        if previous is not None:previous.close()
        previous_budget = getattr(resource.env, 'budget', None)
        if previous_budget is not None:previous_budget.close()
        resource.journal = self._journal(directory/'execution_journal.sqlite')
        resource.env.backend.journal = resource.journal
        resource.env.budget = IncrementalBudget(directory/'budget.json', dict(accepted_iterations=1,
            optimizer_attempts=1, **self.phase.slot_credits[slot]), disk_guard=self.c.guard)
        resource.env.output = directory
        directory.joinpath('raw_samples').mkdir(exist_ok=True)
        resource.env.iteration = self.c.state['iteration']
        resource.env.policy_version = self.c.state['policy_version']

    def _factory(self, slot, proxy):
        c = self.c
        config = copy.deepcopy(c.config)
        config['stage9']['seed'] = (c.stage['seed']+1000003*c.distributed.rank+100003*slot) % 2**32
        backend = VectorLaneBackend(self.collector.lane_client(slot), None)
        builder = OnlineConditionBuilder(BumiMotionFeatureCodec(BumiKinematics(config['paths']['kinematics'])))
        env = VectorUpperEnvironment(config, backend, builder, proxy, None, self.phase.path/f'env{slot:03d}')
        env.disk_guard = c.guard
        resource = SimpleNamespace(env=env, journal=None, sampler=FullMusicSampler(c.catalog, split='train',
            seed=config['stage9']['seed'], window_seconds=c.settings['episode_seconds'],
            random_start=c.stage['dataset']['random_start'], source_probabilities=c.stage['dataset']['source_probabilities']))
        self._bind(slot, resource)
        def close():
            resource.journal.close(); resource.env.budget.close(); backend.client.close()
        resource.close = close
        return resource

    def _open_phase(self, directory, lease, slot_credits):
        c = self.c
        if self.phase is not None:raise RuntimeError('Previous vector phase is not sealed')
        directory.mkdir(parents=True, exist_ok=False)
        credit = {key:sum(item[key] for item in slot_credits) for key in slot_credits[0]}
        per_rank = c.distributed.all_gather_object(credit)
        parent = begin_lease(c.distributed, c.manager, c.budget, directory, lease, per_rank, c.guard)
        self.phase = SimpleNamespace(path=directory, lease=lease, slot_credits=slot_credits,
            per_rank=per_rank, parent=parent)
        def rotate(world):
            world.journal.close()
            world.journal = self._journal(directory/'world_journal.sqlite')
        local_call(c.distributed, lambda:self.collector.world_idle(rotate))
        def rebind():
            for slot, state in enumerate(self.collector.states):
                if state is not None:
                    self.collector.executors[slot].submit(self._bind, slot, state.resource).result()
        local_call(c.distributed, rebind)

    def _finish_phase(self):
        c, phase = self.c, self.phase
        def close_local():
            used = {key:0 for key in phase.per_rank[c.distributed.rank]}
            paths = []
            def close_slot(state):
                resource = state.resource
                record = resource.env.budget.state_dict()
                resource.journal.close()
                resource.env.budget.close()
                return record['used'], [str(resource.journal.path),str(resource.env.budget.database)]
            for executor, state in zip(self.collector.executors, self.collector.states):
                if state is not None:
                    values, local_paths = executor.submit(close_slot, state).result()
                    for key in used:used[key] += values[key]
                    paths.extend(local_paths)
            def close_world(world):
                world.journal.close()
                return str(world.journal.path)
            paths.append(self.collector.world_idle(close_world))
            phase.parent.reserve('completed_vector_execution', **used)
            return paths
        paths = local_call(c.distributed, close_local)
        self.closed_journals = paths
        c.state['budget'] = finish_lease(c.distributed, c.budget, phase.parent, phase.lease, phase.per_rank)
        self.phase = None
        return paths

    @property
    def latency_budget_s(self):
        values = [state.resource.env.latency_budget_s for state in self.collector.states if state is not None]
        return max(values) if values else self.c.settings['latency_budget_s']

    def calibrate(self, saved=None):
        c = self.c
        warmup, samples = c.config['timing']['calibration_warmup'], c.config['timing']['calibration_samples']
        credits = [collection_credit(warmup+samples, c.settings['episode_seconds'], self.latency_budget_s)]*self.num_envs
        name = 'vector_restore' if saved is not None else 'vector_calibration'
        self._open_phase(c.session/'phases'/name/f'rank{c.distributed.rank:02d}', f'{c.session.name}/{name}', credits)
        if saved is not None:
            # 崩溃尾部已预占的generation仍算消耗；宁可跳过编号，也不能复用旧噪声身份。
            saved = copy.deepcopy(saved)
            for record in saved['states']:
                if record is not None:
                    record['execution']['attempt'] = max(record['execution']['attempt'], c.vector_spent_generations)
            local_call(c.distributed, lambda:self.collector.load_state_dict(saved))
            result = dict(restored=True, latency_budget_s=self.latency_budget_s,
                spent_generation_floor=c.vector_spent_generations,contract='fresh_PhysX_episodes_no_hidden_calibration')
        else:
            result = local_call(c.distributed, lambda:self.collector.calibrate(
                count_per_rank=c.settings['rollout_upper_steps_per_rank'], warmup=warmup, samples=samples))
        self._finish_phase()
        return result

    def collect(self, index):
        c = self.c
        directory = c.session/'iterations'/f'{index:06d}'
        path = directory/f'rank{c.distributed.rank:02d}'
        count = c.settings['rollout_upper_steps_per_rank']
        counts = [count//self.num_envs+(slot<count % self.num_envs) for slot in range(self.num_envs)]
        credits = [collection_credit(n, c.settings['episode_seconds'], self.latency_budget_s) for n in counts]
        self._open_phase(path, f'{c.session.name}/iteration{index}', credits)
        started = time.perf_counter()
        fragments, report = local_call(c.distributed, lambda:self.collector.collect(count_per_rank=count,
            policy_version=c.state['policy_version']))
        report['local_compute_seconds'] = report['seconds']
        report['seconds'] = time.perf_counter()-started
        report['synchronization_wait_seconds'] = max(0.,report['seconds']-report['local_compute_seconds'])
        report.update(rank=c.distributed.rank, gpu_peak_allocated_bytes=torch.cuda.max_memory_allocated(c.distributed.device))
        buffer = RolloutBuffer(count)
        for fragment in fragments:
            for row in fragment:buffer.append(row)
        def freeze_targets():
            populate_values(buffer.transitions,c.critic,c.distributed.device,critic_version=c.state['critic_updates'],
                batch_size=c.stage['performance']['value_snapshot_batch_size'])
            return fixed_fragment_targets(fragments,c.critic,c.distributed.device,
                critic_version=c.state['critic_updates'],gamma_upper=c.settings['gamma_upper'],lambda_upper=c.settings['lambda_upper'])[1]
        targets = normalize_advantages_global(local_call(c.distributed,freeze_targets),distributed=c.distributed)
        manifest = build_global_manifest(buffer.transitions,c.distributed)
        if len(manifest)!=c.settings['rollout_upper_steps']:raise RuntimeError('Global vector batch changed')
        def persist():
            writer = BlockRolloutWriter(path/'rollout',policy_version=c.state['policy_version'],
                chunk_size=c.stage['storage']['rollout_chunk_size'],disk_guard=c.guard)
            for row in buffer.transitions:writer.append(row)
            writer.finish()
            torch.save(targets,path/'fixed_targets.pt');c.guard.account_file(path/'fixed_targets.pt')
            atomic_json(path/'collection.json',report)
        local_call(c.distributed,persist)
        self._finish_phase()
        return buffer,targets,manifest,directory,path,report

    def checkpoint_state(self):
        return self.collector.state_dict()

    def verify_frozen(self):
        return self.collector.world_call('verify_frozen')

    def close(self):
        if self.collector is not None:self.collector.close()
