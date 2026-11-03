#!/usr/bin/env python3
"""将联合GENMO完整断点和固定训练/测试样本导出到空闲机器，生成带音乐的双栏预览。

prepare仅在CPU读取本流程完整checkpoint、原划分、原音乐特征与动作：按数据来源分层、
固定种子抽样并按音频SHA去重，选样在任何生成之前完成，不依据生成结果换歌。仅导出
GENMO及FutureEncoder推理参数和合同，不将该精简产物冒充可恢复训练的完整断点。
源文件SHA、原地面元数据和完整有效帧数一并保存；不运行Newton、不写训练缓存。

generate在独立空闲GPU读取该包，复用原60帧codec和因果滚动生成器，执行完整DDIM20、
CFG2.5、eta0。明确给定原始连续12帧启动动作，此后条件只含音乐及自身生成历史，
没有真实在线状态，因此FutureContext显式无效。原未知动作从不输入生成网络。
输出原始/生成PT和NPZ、音频、数值与身份记录；可调用原MuJoCo渲染器生成同步双栏MP4。
这些是完整数据样本长度的运动学预览，不是Newton执行、在线闭环或正式质量统计。
所有正式产物只写全新目录；此工具不修改训练模型、优化器、原数据、SM/WM或运行进程。
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from gem.closedloop.joint_physics.checkpoint import atomic_save, read_checkpoint
from gem.closedloop.joint_physics.config import build_actor, seed_all
from gem.closedloop.joint_physics.contracts import digest

SCHEMA = 'genmo.joint_physics.portable_preview.v1'


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')


def prepare(args):
    from gem.closedloop.joint_physics.data import SourceCatalog
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    value = read_checkpoint(args.checkpoint)
    cfg = copy.deepcopy(value['config'])
    cfg['data']['source_ids'] = None
    # 准备包含test的便携包，不使用只包含train的监督特征缓存。
    cfg['data']['loading'] = {}
    rng = np.random.default_rng(args.seed)
    used_audio, selections = set(), []
    for split, count in [('train', args.train_count), ('test', args.test_count)]:
        catalog = SourceCatalog(cfg, split)
        groups = {}
        for dataset, row in catalog.sources():
            groups.setdefault(dataset.dataset_name, []).append((dataset, row))
        for group_number, name in enumerate(sorted(groups)):
            quota = count // len(groups) + (group_number < count % len(groups))
            if quota == 0:
                continue
            chosen = 0
            for index in rng.permutation(len(groups[name])):
                dataset, row_index = groups[name][int(index)]
                row = dataset.rows[row_index]
                sha = row['source_audio_sha256']
                if sha in used_audio:
                    continue
                selections.append((split, dataset, row_index))
                used_audio.add(sha)
                chosen += 1
                if chosen == quota:
                    break
            if chosen != quota:
                raise ValueError(f'{split}/{name}没有足够的不同音频')
    manifest = dict(schema=SCHEMA, step=value['global_step'], seed=args.seed,
                    checkpoint=str(Path(args.checkpoint).resolve()), checkpoint_sha256=digest(args.checkpoint),
                    sampling=cfg['sampling'], assets=value['contract']['assets'],
                    selection='seeded_source_stratified_before_inference_distinct_audio_sha256',
                    history_source='own_generated_causal_kinematic_proxy', forecast_valid=False,
                    physics='not_run_kinematic_preview_only', sources=[])
    # 首先固定选样清单，后续加载或生成失败不能换成更好看的样本。
    write_json(output/'selection.json', [dict(split=s, dataset=d.dataset_name, source=d.rows[i]['sample_id'])
                                       for s,d,i in selections])
    for number, (split, dataset, row_index) in enumerate(selections, 1):
        row = dataset.rows[row_index]
        sequence = dataset.reader.load_aligned_sequence(row)
        ground = dataset._ground_supervision(sequence)
        original = sequence['qpos'].clone().cpu()
        original[:, 2] -= ground['ground_height_world_m']
        meta = dataset.get_window(row_index, 0)['meta']
        audio = Path(meta['motion_path']).parent.parent / row['audio_path']
        # source_motion_sha256是重定向之前的原动作身份，原reader已核对PT内部元数据；
        # 不能误当成当前机器人qpos文件SHA。当前文件的内容身份另存motion_file_sha256。
        for field, path in [('source_music_feature_sha256', meta['music_feature_path']),
                            ('source_audio_sha256', audio)]:
            if digest(path) != row[field]:
                raise ValueError(f'原数据内容身份改变: {row["sample_id"]}/{field}')
        directory = output/f'{number:02d}'
        directory.mkdir()
        shutil.copy2(audio, directory/'audio.wav')
        atomic_save(dict(qpos=original, music=sequence['music'].cpu(), ground=ground, source_row=row),
                    directory/'input.pt')
        result = dict(number=number, directory=directory.name, source=row['sample_id'],
                      dataset=dataset.dataset_name, split=split, frames=len(original), seconds=len(original)/30,
                      input_sha256=digest(directory/'input.pt'), audio_sha256=row['source_audio_sha256'],
                      source_motion_sha256=row['source_motion_sha256'],
                      motion_file_sha256=digest(meta['motion_path']),
                      music_feature_sha256=digest(meta['music_feature_path']),
                      seed=args.seed+number*1000, ground_metadata=ground)
        manifest['sources'].append(result)
        print(json.dumps(result, ensure_ascii=False), flush=True)
    stats = Path(cfg['stage1']['endecoder']['stats_path'])
    fk = ROOT / cfg['stage1']['endecoder']['kinematics_path']
    if digest(stats) != manifest['assets']['stats_sha256'] or digest(fk) != manifest['assets']['fk_sha256']:
        raise ValueError('stats/FK与断点资产合同不一致')
    shutil.copy2(stats, output/'stats.json')
    atomic_save(dict(schema=SCHEMA, actor=value['actor'], config=cfg, contract=value['contract'],
                     global_step=value['global_step'], checkpoint_sha256=manifest['checkpoint_sha256']),
                output/'model.pt')
    manifest['model_sha256'] = digest(output/'model.pt')
    write_json(output/'manifest.json', manifest)


def generate(args):
    from gem.closedloop.mb_control.contracts import ForecastContext
    from tools.eval.generate_stage1_music_dances import generate as generate_sequence
    from tools.eval.generate_stage1_music_dances import verify_video
    inputs, output = args.inputs.resolve(), args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    manifest = json.loads((inputs/'manifest.json').read_text())
    if manifest['schema'] != SCHEMA or digest(inputs/'model.pt') != manifest['model_sha256']:
        raise ValueError('便携模型内容身份错误')
    value = torch.load(inputs/'model.pt', map_location='cpu', mmap=True, weights_only=False)
    if value['schema'] != SCHEMA or value['global_step'] != manifest['step']:
        raise ValueError('便携模型版本/步数错误')
    cfg = copy.deepcopy(value['config'])
    cfg['stage1']['endecoder']['stats_path'] = str(inputs/'stats.json')
    if (digest(inputs/'stats.json') != manifest['assets']['stats_sha256'] or
        digest(ROOT/cfg['stage1']['endecoder']['kinematics_path']) != manifest['assets']['fk_sha256']):
        raise ValueError('本地stats/FK身份不同')
    seed_all(manifest['seed'])
    actor, _ = build_actor(cfg, args.device)
    actor.load_state_dict(value['actor'], strict=True)
    actor.eval().requires_grad_(False)
    del value
    names = list(actor.stage1.endecoder.kinematics.joint_order)

    class Runtime:
        def sample(self, conditions, steps, guidance_scale, noise):
            if steps != 20 or guidance_scale != 2.5:
                raise ValueError('必须完整DDIM20/CFG2.5')
            context = ForecastContext.invalid(noise, round(float(conditions['decision_time'][0])*600))
            result = actor.sample_differentiable(conditions, context, noise)
            return dict(result, qpos=result['qpos28'])

    manifest['input_bundle'] = str(inputs)
    manifest['generation'] = 'offline_autoregressive_ddim20_given_continuous_demo_prefix12'
    manifest['startup'] = dict(source='original_continuous_first_12_frames', frames=12,
                              duration_seconds=12/30, hold_seconds=0., first_generated_frame=12)
    results = []
    write_json(output/'manifest.json', manifest)
    for source in manifest['sources']:
        src = inputs/source['directory']
        if digest(src/'input.pt') != source['input_sha256'] or digest(src/'audio.wav') != source['audio_sha256']:
            raise ValueError('便携样本内容身份错误')
        data = torch.load(src/'input.pt', map_location='cpu', weights_only=False)
        directory = output/source['directory']
        directory.mkdir()
        shutil.copy2(src/'audio.wav', directory/'audio.wav')
        n, started = source['frames'], time.perf_counter()
        generated, logits, contact_valid, windows = generate_sequence(
            actor.stage1, Runtime(), data['music'], n+9, source['seed'], directory,
            window_frames=60, prefix_frames=12, startup_qpos=data['qpos'][:12])
        common = dict(robot_name='bumi', quaternion_convention='wxyz', qpos_order='mujoco_native',
                      joint_names=names, fps=30)
        atomic_save(dict(common, qpos=data['qpos'], evidence='original_dataset_reference'), directory/'original.pt')
        atomic_save(dict(common, qpos=generated[:n], evidence='ddim20_given_demo_prefix12_joint_genmo_reference',
                         startup=manifest['startup'],
                         model_step=manifest['step'], generated_halo=generated[n:], windows=windows,
                         contact_logits=logits[:n], contact_valid=contact_valid[:n]), directory/'generated.pt')
        np.savez_compressed(directory/'generated.npz', qpos=generated[:n].numpy(), fps=30,
                            joint_names=np.asarray(names), quaternion_convention='wxyz',
                            model_step=manifest['step'], contact_logits=logits[:n].numpy(),
                            contact_valid=contact_valid[:n].numpy())
        result = dict(source, generation_seconds=time.perf_counter()-started, generated_windows=len(windows),
                      startup_hold_seconds=0., demonstration_frames_used=12,
                      startup_prefix_seconds=12/30, first_generated_frame=12)
        if args.mjcf:
            subprocess.run([sys.executable, str(ROOT/'tools/eval/render_bumi_motion.py'),
                '--motion', str(directory/'original.pt'), '--compare-motion', str(directory/'generated.pt'),
                '--mjcf', str(args.mjcf), '--audio', str(directory/'audio.wav'), '--font', str(args.font),
                '--output', str(directory/'comparison.mp4'), '--width', '640', '--height', '540',
                '--title', f'{source["number"]:02d} {source["split"]} / {source["dataset"]}'], check=True)
            result['video'] = verify_video(directory/'comparison.mp4', n)
        result['wall_seconds'] = time.perf_counter()-started
        write_json(directory/'result.json', result)
        results.append(result)
        write_json(output/'results.json', results)
        print(json.dumps(result, ensure_ascii=False), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='mode', required=True)
    prep = sub.add_parser('prepare')
    prep.add_argument('--checkpoint', type=Path, required=True)
    prep.add_argument('--train-count', type=int, default=10)
    prep.add_argument('--test-count', type=int, default=5)
    prep.add_argument('--seed', type=int, default=42)
    gen = sub.add_parser('generate')
    gen.add_argument('--inputs', type=Path, required=True)
    gen.add_argument('--device', default='cuda:0')
    gen.add_argument('--mjcf', type=Path)
    gen.add_argument('--font', type=Path, default=Path('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'))
    for command in (prep, gen):
        command.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    (prepare if args.mode == 'prepare' else generate)(args)


if __name__ == '__main__':
    main()
