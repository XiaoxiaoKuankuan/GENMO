"""真实消费来源与固定噪声探针连接的独立解析测试。

构造两个已知原计划及相邻源帧插值，检查报告严格按真实支持计算关节参考增量；
缺少任一来源计划时只记录不完整，不把未知增量补零。全部文件写临时目录，
不执行物理、不改变PPO概率，不将反事实参考变化作为机器人收益。
"""
import gzip
import json
import torch
from tools.summarize_stage2_batch_comparison import consume_reference_changes


def test_consumed_reference_uses_all_real_source_weights_and_excludes_unknown(tmp_path):
    directory=tmp_path/'4096';directory.mkdir()
    (directory/'result.json').write_text('{}')
    before=dict(qpos=torch.zeros(2,3,9),identities=[{'plan_id':'A'},{'plan_id':'B'}])
    after=dict(qpos=before['qpos'].clone())
    after['qpos'][0,:,7:]=2.;after['qpos'][1,:,7:]=4.
    torch.save(dict(before=before,after=after),directory/'probe_rank00.pt')
    path=tmp_path/'support.jsonl.gz'
    with gzip.open(path,'wt') as f:
        for sources in ([dict(plan_id='A',frame=0,weight=.25),dict(plan_id='B',frame=1,weight=.75)],
                        [dict(plan_id='A',frame=0,weight=.5),dict(plan_id='unknown',frame=1,weight=.5)]):
            f.write(json.dumps(dict(pose_source=sources))+'\n')
    report=consume_reference_changes(tmp_path,path)['4096']
    assert report['covered_center_controls']==1
    assert report['joint_rms_rad']==3.5
    assert report['common_ordinary_joint_rms_rad']==3.5
    assert report['incomplete_probed_support_controls']==1
