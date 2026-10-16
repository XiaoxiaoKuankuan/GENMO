"""生成统一Actor学习率的30组独立有限实验配置，不启动任何GPU作业。

固定5个统一LR、坐标DPPO/joint-score两种目标以及1/3/10倍KL门槛。每份配置都完整
绑定实际优化参数、有限回溯和独立输出身份；原配置、旧模型、Critic与rollout不覆盖。
默认每组最多8轮，只有固定数据和物理评估筛选通过后才由调用方显式启动在线实验。
"""
from __future__ import annotations
import argparse
import copy
import json
from pathlib import Path
import sys
import yaml

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from gem.closedloop.dppo.actor_experiment import optimization_binding
from gem.closedloop.dppo.kl_calibration import digest
from gem.closedloop.dppo.training_config import configuration
from gem.closedloop.dppo.training_checkpoint import file_sha256

RATES=(5e-8,3e-7,1e-6,3e-6,1e-5)


def case_name(objective,rate,multiplier):
    return f'{objective}_lr{rate:g}_kl{multiplier}'


def prepare(base,output):
    config=yaml.safe_load(Path(base).read_text())
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    cases=[]
    for objective in ('coordinate_dppo','joint_score_pg'):
        for rate in RATES:
            for multiplier in (1,3,10):
                c=copy.deepcopy(config);s=c['stage10'];t=s['training']
                t.update(actor_lr=rate,actor_objective=objective)
                s['actor_experiment']['kl_multiplier']=multiplier
                for k,v in dict(mean_coordinate=.0003,maximum_step_mean_coordinate=.0006,p95_coordinate=.0007).items():
                    s['kl'][k]=v*multiplier
                # 新rollout只有一次更新，soft KL仍检查更新前行为位置；不把它当放宽后门槛。
                s['kl']['calibration']=dict(status='exploratory_not_physically_validated',scope='four_trainable_steps_only',
                    optimization_sha256=digest(optimization_binding(s)))
                name=case_name(objective,rate,multiplier);path=output/(name+'.yaml')
                with path.open('x') as f:yaml.safe_dump(c,f,allow_unicode=True,sort_keys=False)
                configuration(path)
                cases.append(dict(name=name,path=str(path.resolve()),sha256=file_sha256(path),
                    objective=objective,actor_lr=rate,kl_multiplier=multiplier,optimization=optimization_binding(s)))
    report=dict(schema='genmo.stage2.uniform_lr_candidate_manifest.v1',base_sha256=file_sha256(base),
        scope='independent_finite_experiment_configurations_not_physical_acceptance',candidates=cases)
    with (output/'manifest.json').open('x') as f:json.dump(report,f,ensure_ascii=False,indent=2)
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base',type=Path,default=ROOT/'configs/closedloop/stage2_actor_lr_server2.yaml')
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();print(json.dumps(prepare(args.base,args.output),ensure_ascii=False,indent=2))
