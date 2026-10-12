"""将Stage10 v3已完成验收的原始JSON整理为可导出的中文对照图。

本工具只读计算标定、配对物理评估和完整episode价值审计，不加载模型、占用GPU、
重算奖励或补造实验数据。计算图使用相同真实链和完整更新次数的Actor阶段时间；
物理图直接使用原报告的音乐簇置信区间；Critic图使用无自举的MC样本误差，明确
其有限轨迹边界。三类量分别展示，不将计算提速解释为学习收益。

输入可来自不同已完成实验，但计算输入须具有相同case和更新工作量，物理输入须
来自同一初始化报告，MC输入须是独立完整回报审计。输出到不存在的新目录，同时
保存PNG、SVG和全部输入SHA清单。中文字体显式指定，便于服务器无桌面环境复现。
该脚本只生成报告图表；不修改checkpoint、训练配置、原日志或既有验收结果。
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--compute', nargs=2, action='append', metavar=('LABEL', 'REPORT'), required=True)
    parser.add_argument('--physical', nargs=2, action='append', metavar=('LABEL', 'REPORT'), required=True)
    parser.add_argument('--mc-audit', type=Path, required=True)
    parser.add_argument('--font', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists() or not args.font.is_file():
        parser.error('输出目录必须不存在，中文字体必须存在')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    import numpy as np

    font_manager.fontManager.addfont(str(args.font))
    plt.rcParams.update({'font.family': [font_manager.FontProperties(fname=str(args.font)).get_name(), 'DejaVu Sans'],
        'font.size': 10, 'axes.unicode_minus': False, 'svg.fonttype': 'path'})
    inputs = []

    def read(path):
        path = Path(path)
        raw = path.read_bytes()
        inputs.append(dict(path=str(path.resolve()), sha256=hashlib.sha256(raw).hexdigest()))
        return json.loads(raw)

    compute = [(label, read(path)) for label, path in args.compute]
    physical = [(label, read(path)) for label, path in args.physical]
    mc = read(args.mc_audit)
    case_ids, work = set(), set()
    for _, report in compute:
        if len(report['candidates']) != 1 or not report['candidates'][0]['evaluation_publication_admissible']:
            raise ValueError('计算图要求单一且通过完整更新验收的候选')
        candidate = report['candidates'][0]
        case_ids.add(report['source_case_sha256'])
        work.add((report['real_global_chains'], candidate['update']['optimizer_steps'],
                  candidate['update']['actual_global_bc_samples']))
    if len(case_ids) != 1 or len(work) != 1:
        raise ValueError('计算候选的真实输入或更新工作量不匹配')
    baseline_ids = {report['inputs'][0]['sha256'] for _, report in physical}
    if len(baseline_ids) != 1 or any(report['comparison_mode'] != 'algorithm' for _, report in physical):
        raise ValueError('物理图必须是同一初始化上的算法对照')
    if mc['target_scope'] != 'actual_discounted_complete_episode_returns_without_critic_bootstrap':
        raise ValueError('Critic图要求完整回报、无价值自举')

    fig, axes = plt.subplots(2, 2, figsize=(14, 9.5), layout='constrained')
    colors = ['#227a9f', '#db933d', '#689c60', '#a97199']
    ax = axes[0, 0]
    times = [r['candidates'][0]['actor_seconds'] for _, r in compute]
    bars = ax.bar([label for label, _ in compute], times, color=colors[:len(compute)])
    ax.bar_label(bars, fmt='%.1f秒', padding=4)
    ax.set_ylim(0, max(times) * 1.2)
    ax.set_ylabel('Actor更新阶段墙钟 / 秒')
    chain_count, update_count, bc_count = next(iter(work))
    ax.set_title(f'相同{chain_count}条真实链、{update_count}次Adam、{bc_count}个BC样本\n'
                 '单次固定工作量对照；不含物理采样与最终KL')

    ax = axes[0, 1]
    values = np.array([r['source_balanced_reward_delta'] for _, r in physical])
    limits = np.array([r['reward_delta_cluster_bootstrap_95_percent'] for _, r in physical])
    positions = np.arange(len(physical))
    ax.errorbar(values, positions, xerr=np.stack((values-limits[:, 0], limits[:, 1]-values)),
                fmt='o', capsize=5, color=colors[0])
    ax.axvline(0, linestyle='--', color='#777777', linewidth=1)
    ax.set_yticks(positions, [label for label, _ in physical])
    ax.invert_yaxis()
    ax.set_xlabel('相对初始化的实际四来源等权回报变化')
    ax.set_title('固定32任务、16首音乐、两种子\n误差线为95%音乐簇区间，单轮结果不证明收敛')

    names = list(mc['models'])
    model_labels = {'initial': '初始Critic', 'warm8_critic32': '8轮预热',
                    'warm32_long180': '长轨迹32轮预热'}
    ax = axes[1, 0]
    errors = [mc['models'][name]['overall']['mse'] for name in names]
    bars = ax.bar([model_labels.get(name, name) for name in names], errors, color=colors[:len(names)])
    ax.bar_label(bars, fmt='%.1f', padding=4)
    ax.set_ylim(0, max(errors) * 1.2)
    ax.set_ylabel('完整折扣回报样本MSE')
    count = mc['models'][names[0]]['overall']['transition_count']
    ax.set_title(f'完整episode独立回报：{count}条因果条件\n无末端价值自举；有限MC样本不是期望V的真值')

    ax = axes[1, 1]
    sources = list(mc['models'][names[0]]['by_source'])
    positions = np.arange(len(sources)); width = .8 / len(names)
    for index, name in enumerate(names):
        ev = [mc['models'][name]['by_source'][source]['explained_variance'] for source in sources]
        ax.bar(positions + (index-(len(names)-1)/2)*width, ev, width,
               label=model_labels.get(name, name), color=colors[index % len(colors)])
    ax.set_xticks(positions, sources)
    ax.set_ylabel('Explained Variance')
    ax.axhline(0, color='#777777', linewidth=.8)
    ax.legend(loc='best', frameon=False)
    ax.set_title('分来源完整回报预测\n每来源仅一首歌；FineDance发生真实航向终止')
    for ax in axes.flat:
        ax.spines[['top', 'right']].set_visible(False)
        ax.grid(axis='y', alpha=.15)
        ax.set_axisbelow(True)
    fig.suptitle('GENMO s595000 → DPPO＋BC｜冻结GMT 140500｜八张RTX 6000D有限验收', fontsize=15)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    fig.savefig(args.output_dir / 'acceptance.png', dpi=180)
    fig.savefig(args.output_dir / 'acceptance.svg')
    plt.close(fig)
    manifest = dict(schema='genmo.stage10.acceptance_figure.v1', inputs=inputs,
        physical_statistical_scope='paired_music_cluster_intervals_without_multiple_comparison_correction',
        compute_work=dict(zip(('real_chains', 'adam_steps', 'bc_samples'), next(iter(work)))))
    (args.output_dir / 'inputs.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
    print(args.output_dir)


if __name__ == '__main__':
    main()
