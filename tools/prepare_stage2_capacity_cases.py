"""为服务器2生成可审计的128环境容量实验配置，不启动训练、不覆盖既有产物。

以已验收非对称Stage2配置为基础，保持奖励、15+4+1随机核、学习率、PPO两epoch、
物理频率与全部保护条件。先生成新源码身份下的完整MC预热配置：独立训练音乐数
从每来源12增为24，仍用独立验证音乐和种子选择MC/TD比例。旧Critic初始化被明确
移除，不能用修改SHA的方式迁移。预热完成后传入真实新artifact，生成四个独立run。

容量矩阵为128环境/卡、4或8秒世界窗口、Actor微批128或256；全局optimizer
minibatch统一2048条链，BC按真实名义采集容量从512升到1024/2048，Critic全局
批量2048且仍32epoch。每组总计16秒世界时间：4秒四轮、8秒两轮，实际有效样本、
失败/重置与物理控制数必须由报告核对，不能把名义容量当作实际完成工作。
有限实验遇到硬KL拒绝时整轮回滚并退出，不通过追加rollout获得更多交互预算；
各组停止轮数和optimizer尝试上限单独绑定，保留原资源、信号及物理保护。

本工具只写完整配置及manifest，包含实际源码提交、基础配置SHA、实验边界和
可复制启动参数。默认稳定数值核、CUDA Graph关闭，所有run都重新初始化PhysX，
不恢复旧Actor、优化器或rollout。路径必须为独立实验目录，已存在文件明确拒绝。
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import yaml

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from gem.closedloop.dppo.training_config import configuration


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()


def prepare(base,output,*,critic_artifact=None,kl_microbatch=128,mc_music=24):
    base,output=Path(base).resolve(),Path(output).resolve()
    config=yaml.safe_load(base.read_text())
    if kl_microbatch not in (128,256,512) or mc_music<12:
        raise ValueError('Use an explicitly scanned KL batch and at least the baseline MC music coverage')
    precision=config['stage10']['precision']
    if (precision['linear_execution']!='stable_tf32x3' or precision['mode']!='native_fp32'
            or precision.get('cuda_graph',False) or precision['compile']):
        raise ValueError('Capacity experiments keep the audited stable eager numerical kernel')
    config['runtime'].update(num_envs=128,generation_batch=128)
    stage=config['stage10'];stage['critic']['preheat_artifact']=None
    stage['critic']['global_batch']=2048
    stage['critic']['complete_mc']['training_samples_per_source']=mc_music
    stage['scale']['actor_minibatch_chains']=2048
    stage['training'].update(denoising_microbatch=128,kl_microbatch=kl_microbatch,bc_samples_per_rollout=1024)
    stage['limits']['accepted_iterations']=4
    stage['run_control']['max_training_iterations']=4
    directory=output/('preheat_config' if critic_artifact is None else 'capacity_configs')
    directory.mkdir(parents=True,exist_ok=False)
    identity=dict(base_config=str(base),base_config_sha256=sha(base),
        source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        source_branch=subprocess.check_output(['git','branch','--show-current'],cwd=ROOT,text=True).strip(),
        scope='server2_finite_capacity_not_formal_training',shared_numerical_kernel=True,
        maximum_world_seconds_per_case=16,configs=[])
    if critic_artifact is None:
        cases=[('preheat',4,128,0)]
    else:
        artifact=Path(critic_artifact).resolve()
        stage['critic']['preheat_artifact']=dict(path=str(artifact),sha256=sha(artifact))
        cases=[(f'env128_world{seconds}_actor{micro}',seconds,micro,16//seconds)
            for seconds in (4,8) for micro in (128,256)]
    for name,seconds,micro,iterations in cases:
        c=copy.deepcopy(config);s=c['stage10']
        s['ppo_rollout']['world_seconds']=float(seconds)
        s['training'].update(denoising_microbatch=micro,bc_samples_per_rollout=256*seconds)
        if iterations:
            s['training']['kl_rejection']='abort'
            s['limits']['accepted_iterations']=iterations
            s['run_control']['max_training_iterations']=iterations
            # 额外一个合法边界只预留可能出现的变长尾批，不增加已接受轮数。
            s['limits']['optimizer_attempts']=iterations*2*(seconds+1)
        path=directory/(name+'.yaml')
        with path.open('x') as stream:yaml.safe_dump(c,stream,allow_unicode=True,sort_keys=False)
        parsed=configuration(path)
        args=['--config',str(path),'--output-dir',str(output/name)]
        args+=['--critic-preheat-only'] if not iterations else ['--stop-after-iteration',str(iterations)]
        identity['configs'].append(dict(name=name,path=str(path),sha256=sha(path),
            launch_args=args,derived=parsed['stage10']['derived'],stop_after_iteration=iterations))
    with (directory/'manifest.json').open('x') as stream:json.dump(identity,stream,ensure_ascii=False,indent=2)
    return identity


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base',type=Path,default=ROOT/'configs/closedloop/stage2_asymmetric_server2.yaml')
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--critic-artifact',type=Path)
    parser.add_argument('--kl-microbatch',type=int,choices=[128,256,512],default=128)
    parser.add_argument('--mc-music-per-source',type=int,default=24)
    args=parser.parse_args()
    print(json.dumps(prepare(args.base,args.output,critic_artifact=args.critic_artifact,
        kl_microbatch=args.kl_microbatch,mc_music=args.mc_music_per_source),ensure_ascii=False,indent=2))
