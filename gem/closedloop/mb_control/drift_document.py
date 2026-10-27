"""将服务器2真实漂移诊断证据整理为可独立阅读的中文HTML报告。

输入是本轮独立目录中的只读JSON、完整真实评价和候选审计结果；输出为HTML、PNG
曲线与机器可读的实验状态。所有指标读取实际产物，未完成的25/50明确标成未完成，
不会根据训练计数推定评价成功。旧目标与新独立曲线分开，保留负结果、支持域拒绝、
假设和未验证事项；本工具不修改模型、奖励、正式训练状态或原始物理证据。
"""
import argparse
from collections import defaultdict
from datetime import datetime,timezone
import html
import json
from pathlib import Path
import statistics


def load(root,name,default=None):
    p=root/name
    return json.loads(p.read_text()) if p.exists() else default


def table(headers,rows):
    esc=lambda x:html.escape(str(x))
    return '<table><thead><tr>'+''.join('<th>'+esc(h)+'</th>' for h in headers)+'</tr></thead><tbody>'+''.join(
        '<tr>'+''.join('<td>'+esc(x)+'</td>' for x in row)+'</tr>' for row in rows)+'</tbody></table>'


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--root',required=True)
    args=parser.parse_args();root=Path(args.root)
    obj=load(root,'objective_summary.json');grad=load(root,'gradient_conflict_summary.json')
    log=load(root,'offline_v3/log_audit.json');trust=load(root,'trust_drift_summary.json')
    phys=load(root,'physical_direction_summary.json');dist=load(root,'sampling_unique_summary.json')
    cond=load(root,'conditional_drift/summary.json');ori=load(root,'body_orientation_comparison.json')
    timecover=load(root,'music_time_coverage.json',{})
    new_audit=load(root,'new_trajectory_audit/summary.json',{})|load(root,'new50_trajectory_audit/summary.json',{})
    evals=[]
    base=obj['0']['B_episode']
    evals.append({'update':0,'loss':base,'completed':8,'episodes':8,'origin':'逐张量核对Stage1后复用原完整8曲物理基线'})
    for version,folder in [(10,'episode_multianchor'),(25,'episode_continuation'),(50,'episode_continuation')]:
        path=f'{folder}/evaluations/actor_{version:06}/physical_summary.json'
        result=load(root,path)
        if result:
            evals.append({'update':version,'loss':statistics.mean(r['physical_metrics_mean']['dance_loss'] for r in result['physical_results']),
                'completed':result['completed'],'episodes':result['episodes'],'origin':path,
                'per_song':[{k:r[k] for k in ['task','complete','physical_seconds','music_seconds','reason']}|
                    {'loss':r['physical_metrics_mean']['dance_loss']} for r in result['physical_results']]})
        else:evals.append({'update':version,'loss':None,'completed':None,'episodes':8,'origin':'完整评价尚未落盘，不填预测值'})
    manifest=load(root,'episode_multianchor/manifest.json')['config']
    baseline_folder=Path(manifest['diagnostic']['source_run'])/'evaluations/fixed_delay_replay/initial_genmo_baseline'
    baseline_rows={r['task']:r for r in json.loads((baseline_folder/'physical_summary.json').read_text())['physical_results']}
    for e in evals[1:]:
        if e['loss'] is not None:
            e['songs_better_than_stage1']=sum(r['loss']<baseline_rows[r['task']]['physical_metrics_mean']['dance_loss'] for r in e['per_song'])
    paired=[]
    for e in evals[1:]:
        if e['loss'] is None:continue
        for row in load(root,e['origin'])['physical_results']:
            before=baseline_rows[row['task']]
            events=lambda r:{x['request_tick']:tuple(x.get(k) for k in ['actual_ready_tick','commit_tick','rejection_tick','accepted']) for x in r['events']}
            a,b=events(before),events(row)
            paired.append({'update':e['update'],'task':row['task'],
                'input_music_initial_identity_equal':before['input_identity']==row['input_identity'],
                'seed_noise_rule_equal':all(before[k]==row[k] for k in ['seed','noise_seed','noise_rule']),
                'same_request_ticks':set(a)==set(b),'delivery_events_equal':a==b,
                'same_complete_physical_duration':before['complete'] and row['complete'] and before['physical_seconds']==row['physical_seconds'],
                'timing_mode':row['timing_mode']})
    (root/'paired_input_audit.json').write_text(json.dumps(paired,ensure_ascii=False,indent=2))
    accepted=[];last_evaluated=max(e['update'] for e in evals if e['loss'] is not None)
    for folder,minimum in [('episode_multianchor',0),('episode_continuation',10)]:
        for line in (root/folder/'training.jsonl').open():
            row=json.loads(line)
            if minimum<row['accepted_actor_updates']<=last_evaluated and row['accepted_execution_feedback_update']:
                accepted.append(row)
    update_deltas=defaultdict(list)
    for row in accepted:
        metrics=row['output_trust_region']['attempts'][-1]['metrics']
        for term in ['dance','episode']:
            update_deltas[term].append(metrics['losses_after'][term]-metrics['losses_before'][term])
    local_checks={k:{'updates':len(v),'up':sum(x>0 for x in v),'mean_delta':statistics.mean(v)} for k,v in update_deltas.items()}
    state={'generated_utc':datetime.now(timezone.utc).isoformat(),'independent_evaluations':evals,
        'original_run_untouched':True,'large_training_expanded':False,
        'quality_improvement_established':False,'music_time_coverage':timecover,'accepted_candidate_model_loss':local_checks,
        'paired_input_audit':paired,'new_full_trajectory_objectives':new_audit}
    (root/'decisive_report.json').write_text(json.dumps(state,ensure_ascii=False,indent=2))
    sections=[]
    def add(title,body):sections.append('<section><h2>'+title+'</h2>'+body+'</section>')
    add('结论与证据边界','''<p><b>主要问题是：短时优化只看局部、更新更容易改善监督和先验，Stage1生成行为发生改变后，又被在线历史反馈放大为整曲转向与相位漂移。</b>机器人仍有活动，不代表保留了原音乐舞蹈。世界模型存在接触阶段反向案例，但当前证据不支持把全部退化归咎于世界模型。</p>
    <p>已完成旧5版本×8曲完整轨迹重算、5140个H16 anchor、700次接受更新/1225个候选、真实Newton参数候选对照，以及从Stage1新初始化的独立实验。原正式进程、源码工作树和检查点未被覆盖或中断。新实验负结果完整保留；尚未形成扩大正式训练的依据。</p>''')
    rows=[]
    for v in ['0','100','300','600','700']:
        d=obj[v];roots=statistics.mean(x['C_world_root_rms_m'] for x in d['song_details'])
        rows.append([v,f"{d['A_anchor_H16']:.6f}",f"{d['B_episode']:.6f}",f"{d['C_world']:.6f}",f'{roots:.5f}',f"{d['world_yaw_rms_rad']:.5f}"])
    add('1. 训练与评价目标：局部和整曲都已退化',table(['更新','A 每anchor重对齐H16均值','B episode一次对齐','C 世界坐标Loss','世界根位置RMS / m','世界yaw RMS / rad'],rows)+'''
    <p>八曲等权，全部完整执行。A是H16每步均值，原优化使用折扣积分，二者尺度不能混用。B=C是因为本次配对评价初始SE2为严格恒等。第700次A仍是基线约2.03倍，因此不能解释成“局部一直改善，只是全局评分不一致”。第600次比300次恢复，但各训练版本均劣于Stage1，并非每个检查点严格单调恶化。</p>
    <p>原训练在每个anchor重新SE2对齐；真实评价比较冻结原音乐世界时间线。global prior也按当时记录参考重新对齐，并只对本H16实际支持点约束。局部目标丢失已经积累的平移/yaw偏差；当前prior不等于episode累计目标。</p>
    <p>身体位置以根部为参照，身体姿态却用世界旋转（dance_losses.py:70–82）。世界yaw因此进入根姿态与身体姿态两组。只在离线改为根部相对身体姿态，0/700误差为0.02108/0.04393，仍明显退化；不把重定义评分后的下降当作训练收益。</p>
    <p>代码：dance_losses.AnchorAlignment:32；imagination.py:184；collector.py:245；rewards.py:91；dance_evaluation.py。建议保留局部H16，同时增加episode初始固定SE2，身体相对姿态与根yaw单独计分，改合同后重新验证。</p>''')
    content=[]
    for v in ['0','100','300','600','700']:
        ds=[x['D_music_content'] for x in obj[v]['song_details']]
        avg=lambda key:statistics.mean(x[key] for x in ds)
        part=lambda name,key:statistics.mean(x[name][key] for x in ds)
        content.append([v,f"{avg('generated_reference_joint_rms_rad'):.5f}",f"{avg('executed_joint_rms_rad'):.5f}",f"{part('upper','range_ratio'):.3f}",f"{part('lower','range_ratio'):.3f}",f"{part('upper','velocity_phase_cosine'):.3f}",f"{part('lower','velocity_phase_cosine'):.3f}"])
    add('2. 不是静止坍缩，而是动作内容、转向和相位偏离',table(['更新','生成参考关节偏差rad','真实执行关节偏差rad','上肢幅度比','下肢幅度比','上肢速度相位cos','下肢速度相位cos'],content)+'''
    <p>此表保留原轨迹内容计算口径；offline_v3/trajectory_objectives.json还保存直接使用真实50Hz原始qd的21关节误差和相位，避免只用位置差分代替速度。幅度和速度总体维持，但关节内容偏差、转向及速度相位变差。第700次新增损失约62.4%来自根部/身体姿态，约17.3%来自根位置；不能只看活动度0.65门槛判断舞蹈保真。</p>''')
    terms=['dance','bc','local','global','track','joint_limit']
    add('3. 梯度与真实AdamW方向：主任务没有主导全部更新',table(['分量','平均加权梯度范数','实际更新方向下降次数','上升次数','平均加权梯度·实际位移'],[
        [k,f"{grad['mean_weighted_gradient_norm'][k]:.6g}",log['actual_weighted_gradient_dot_adamw_delta'][k]['descent'],log['actual_weighted_gradient_dot_adamw_delta'][k]['ascent'],f"{log['actual_weighted_gradient_dot_adamw_delta'][k]['mean']:.6g}"] for k in terms])+'''
    <p>BC范数约44倍dance，local约15倍。BC与dance平均余弦约0.00015，主要接近正交，不能说BC总与dance反向；local与dance有327/700负余弦，local与global有373/700负余弦。原H16执行积分质量为0.297084，BC和prior按均值计算；从H32缩短到H16会改变相对尺度，不能只比较配置权重。</p>
    <p>Actor实际AdamW位移才是结论依据；BC沿700/700实际更新下降，但dance有138/700上升。action_rate_weight=0、terminal_value_weight=0，当前退化不能归因于已关闭的Action Rate或Critic。硬Joint Limit仍为2，不通过投影删除。</p>'''+table(['参数模块','dance加权范数','BC加权范数','local加权范数'],[[m]+[f"{log['module_gradients'][m+'/'+k]['weighted_norm']['mean']:.6g}" for k in ['dance','bc','local']] for m in ['stage1_backbone','history_prefix','future_encoder','diffusion_output']])+'''
    <p>独立候选分解已分别计算六项梯度、四个参数模块及真实AdamW位移。各版本同一anchor比较Dance、Dance+Local/Global、Dance+BC、全部Loss，再分别只更新FutureEncoder/全量GENMO。参数、AdamW一二阶矩/step和RNG全部回滚；详见candidates_*完整JSON/PT与physical_candidate_details.json。建议在不删除BC的前提下，明确执行积分尺度、辅助梯度预算和实际候选主任务检查；不能仅凭梯度余弦证明更新有效。</p>''')
    add('4. 实际物理反事实：先验组合确实能把改善改成恶化',table(['候选/可训练参数','真实物理改善','模型/物理可判方向一致','输出信任域合格'],[[k,str(v['physical_improved'])+'/'+str(v['cases']),str(v['direction_matches'])+'/'+str(v['resolved']),str(v['output_trust_valid'])+'/'+str(v['cases'])] for k,v in phys['variants'].items()])+'''
    <p>共133个实际候选；相同参考逐字节去重后116个，115个方向可判，113一致、2相反。24个请求anchor中只有17个在既有支持域合法；3个候选另有超域拒绝。覆盖3首歌、2个时间段，不能外推为所有舞蹈可靠。</p>
    <p>5个配对案例中，仅Dance改善，全部Loss反而恶化。例如700/F9u/1140：Dance的WM/真实变化为−3.749e−5/−3.651e−5；全部Loss为+1.893e−6/+4.729e−6。这里WM也预测了恶化，直接指向目标组合问题。</p>
    <p>明确反向案例：100/F9u/3540，右脚落地/单支撑/高速，Future-only的WM预测−9.852e−6，Newton实际+2.628e−5，接触改变且真实末端yaw变化0.00386rad。另一个300/Fine194案例方向相反但量级仅约1e−7。所有真实初态恢复误差为0；不是有限差分替代物理执行。</p>
    <p>对16个全量候选继续执行两次真实在线生成，13个世界目标改善、3个恶化。它们是独立反事实子集，不能声称已恢复138个历史更新各自的完整下一轮反事实；旧日志没有保存全部必要求解器/旧优化器状态。</p>''')
    add('5. 信任域限制单次当前请求，未限制整曲策略漂移','''<p>1225候选中700接受、525拒绝；138个接受候选的短时Dance上涨，平均0.0447%、最大0.2791%。实际容差是1e−5+0.5%×loss；在这138例中等效容差平均0.7803%、最大1.2772%，不是纯0.5%。</p>
    <p>接受样本单次关节RMS变化均值0.001614、最大0.009766rad，确实满足单步限制；但累计global关节偏差最大0.32147rad。经过局部对齐的global根位置偏差最大0.05694m，未对齐最大0.48663m；SO(3)坐标RMS最大1.67450rad，注意这不是yaw标量。</p>
    <p>接受样本幅度比最低0.66327、速度比最低0.65078、相位余弦最低−0.00994，均能通过0.65/−0.2门槛。global判断只覆盖当前少数支持点，局部教师条件还随学生历史改变。结论：门槛足以防明显静止，却不能保证固定音乐动作身份、累计转向或未见请求保持。</p>
    <p>建议使用跨音乐/跨时段持久固定请求库检查参数更新，并分开检查未对齐episode偏差；不能仅靠提高阈值或无限缩学习率解决。代码：output_trust_region.py:39；teacher_training.py:237–254、309–339。</p>''')
    add('6. H16真正看到的新动作很少','''<p>5140个真实anchor中，H16均有16步受新未来goal影响，但10步仅由未来goal影响，1步只有当前速度halo支持，真正消费新姿态中心只有5步，对应3个不同30Hz源点（含速度支持共4点）。不能将16步当作16帧完整新动作。</p>
    <p>整曲1037个计划中，每计划平均实际取得41.34个新源点，最终以当前姿态消费10.41点，30.02点在消费前被后续计划覆盖；音乐范围内覆盖比例72.99%。H16只直接约束最终约10个姿态源点中的3个。当前rollout_anchor后续参考接管使用记录的行为计划，不重新可微生成后续计划，不能称为完整多决策闭环BPTT。</p>
    <p>建议同请求增加多个经过真实方向验证的消费时间窗口，已消费参考detach；按真实版本/时间追踪覆盖。不要为了扩大时间覆盖直接把H改32，更不能把被覆盖源点当实际执行。</p>''')
    add('7. 教师动作身份和在线反馈','''<p>冻结参数不等于固定生成条件。40724真实训练请求仅11441（28.1%）与对应教师存在相同request_tick，只有23个commit_tick相同。初始48个请求中23个request_tick相同、0个commit_tick相同。噪声种子规则42+(request_tick//60)×1009，所以不同调度会产生不同动作身份；同tick仍可能有不同物理历史/前缀。</p>
    <p>这说明同请求局部教师和独立固定在线音乐教师可能不是同一条动作，不是缓存身份碰撞，也不意味着固定延迟配对评价无效。应先对齐动作身份和外生调度，再判断local/global冲突。</p>'''+table(['版本','固定条件完整新参数关节RMS','仅新FutureEncoder','仅新Stage1参数','真实在线条件额外差异RMS'],[[v]+[f"{cond[v]['fixed_conditions'][m]['joint_rms_rad']['mean']:.6f}" for m in ['full','future_only','stage1_only']]+[f"{cond[v]['online_feedback_joint_rms']['mean']:.6f}"] for v in ['100','300','700','new10']])+'''
    <p>每版本55个原Stage1真实请求、同音乐/历史/前缀/预测/噪声。原生成重现最大绝对差1.97e−6。主要直接变化来自学生沿用的Stage1主干，冻结原Stage1教师没有更新；真实在线条件变化带来更大的偏移。非线性相互作用另存，不能把这些RMS相加或解释成精确贡献百分比。new10属于本轮多环境主实验，已出现同样反馈问题。</p>''')
    add('8. 训练分布：不是全部低速，而是大幅腿部困难动作被筛掉',table(['阶段','去重真实请求数'],[[k,dist[k]['count']] for k in ['A_requests','B_formed','B_legal','C_outside','D_drawn','E_updated']])+'''
    <p>14首训练音乐、8首固定评价音乐，音乐ID无重叠，种子42。有效更新1400个真实请求，每次2个；68.14%高速、8.50%低速、21.86%单支撑、48.71%包含接触切换，不能称“训练全是静止”。但高幅度只有7/1400（0.5%），形成H-anchor池高幅度8.35%；被检查的高幅度请求98.9%超域，高速请求82.37%超域。合法样本腿速度均值0.5898rad/s，拒绝样本1.1370rad/s。</p>
    <p>全部形成39953，已做域分类29402，其中7118合法、22284超域；其余10551未进入分类，不应当作合法或拒绝。D为1637独立请求，重复抽取记录1678。评价基线高速71.2%、低速2.33%、高幅1.75%。歌曲等权，不能把最长FineDance提供的大量anchor数量当作评分权重。</p>
    <p>A请求的活动统计采用瞬时速度/较长历史窗口，而B/E用H16区间，类别优先级也不同，跨A/B直接比较幅度和动作类别会误导；本结论使用同口径B形成→合法→E。状态分量和首次超域预测步已保存在sampling_unique_summary.json。</p>
    <p>真实音乐时段覆盖：</p>'''+('<pre>'+html.escape(json.dumps(timecover.get('all',{}),ensure_ascii=False,indent=2))+'</pre>')+'''
    <p>建议在既有可信支持域内增加多音乐、多时段、腿部/接触样本，不放宽未经验证的状态界限。14曲单种子是否构成主要过拟合原因仍需扩充独立验证，不能靠分布统计直接断言因果。</p>''')
    add('9. 本轮实际修改与独立0/10/25/50结果',table(['新实验有效更新','八曲等权音乐Loss','完整歌曲','来源'],[[e['update'],'未完成' if e['loss'] is None else f"{e['loss']:.8f}",'待核验' if e['completed'] is None else f"{e['completed']}/{e['episodes']}",e['origin']] for e in evals])+'''
    <p>实际接受候选的当前固定微批模型Loss变化：</p>'''+table(['分量','已评价版本以内的接受更新','模型Loss上涨次数','平均变化'],[[k,v['updates'],v['up'],f"{v['mean_delta']:.8g}"] for k,v in local_checks.items()])+'''
    <p>输入、音乐、初态、种子、请求时刻、实际交付事件和完整时长的逐曲对照保存在paired_input_audit.json；这不是只检查配置里写了fixed_delay。即使训练微批损失下降，完整真实曲线仍可恶化。</p>
    <p>新实验实际完整轨迹同口径重算：</p>'''+table(['新更新','每anchor重对齐H16均值','episode整曲Loss','世界yaw RMS/rad'],[[v,f"{d['A_anchor_H16_mean']:.6f}",f"{d['B_episode_once_loss']:.6f}",f"{d['C_world_yaw_rms_rad']:.6f}"] for v,d in new_audit.items()])+'''
    <p>新10→25的局部真实H16由0.02241降为0.01818，但世界yaw由0.20313涨为0.39983rad，整曲Loss进一步上升。这是实际轨迹支持的“局部部分恢复、累计转向更差”；两版局部指标仍劣于Stage1的0.01067。</p>
    <p>逐曲而非只看总均值：</p>'''+table(['音乐','Stage1']+[str(e['update']) for e in evals[1:] if e['loss'] is not None],[[task,f"{row['physical_metrics_mean']['dance_loss']:.6f}"]+[f"{next(r['loss'] for r in e['per_song'] if r['task']==task):.6f}" for e in evals[1:] if e['loss'] is not None] for task,row in baseline_rows.items()])+'''
    <p>新25八曲均劣于Stage1。FineDance 139（159.2秒）四段yaw RMS为0.5498/1.1127/1.2595/2.4318rad，对应四段音乐Loss为0.1188/0.3059/0.3602/0.7276，后段累计转向显著放大。F9u新25比新10恢复，但仍差于本曲Stage1，不能只选恢复歌曲宣布成功。</p>
    <p>从原Stage1逐张量初始化，新的AdamW，冻结Stage1教师/SM Actor/WM及统计不变。新增episode_objective.py和独立SE2状态字段；collector在reset建立初态目标，imagination保留局部H16并同时计算episode累计目标。teacher_training归一化执行积分，保留BC并限制软辅助梯度预算，冲突软梯度投影；硬限位不截断。</p>
    <p>独立配置：episode权重1，4个真实anchor/4个环境；BC/local/global/track辅助梯度预算相对组合主目标为0.25/0.5/0.25/0.25。候选模型与AdamW/RNG完全回滚；组合主任务容差改为rtol=0、atol=1e−7，其余输出与支持域不放宽。Critic仍0，Action Rate仍0。</p>
    <p>0为精确权重核对后复用完整原八曲基线。0→10在episode_multianchor；10→50从完整10检查点恢复Actor/AdamW/RNG/监督与任务游标，真实求解器明确重新开始episode，不伪装连续物理恢复。25/50评价异步独立进程，使用冻结检查点、相同音乐/初态/噪声种子/固定延迟事件。单环境辅助实验另有10次完整8曲0.07159399，按预定计算预算结束，负结果保留。</p>
    <p>当前已出结果没有确立正向趋势，故不扩大正式训练。新增episode目标、辅助预算和多anchor单独组合尚不能解决整曲反馈漂移；下一优先项是跨请求固定输出约束、教师动作身份一致，以及实际消费窗口覆盖，而不是继续盲目拉长本轮训练。</p>
    <p>测试：服务器2实际25项合同/信任域回滚/教师/采集/episode测试通过（0.82秒）。新目标真实首步12候选方向一致，但6个全量初始步超过输出限制，不能当作接受收益；候选学习率缩至1/16后3个真实Newton小步均改善。它们没有预测出完整新10的退化，进一步证明单anchor方向通过不足以授权整曲长期收益。</p>''')
    add('10. 证据分级、失败与尚未完成事项','''<ul>
    <li><b>确认的代码/目标不一致：</b>局部反复SE2对齐与整曲世界目标不同；身体姿态并非根部相对姿态；执行积分与均值损失尺度不同；局部与固定教师调度/噪声身份通常不同；当前H16并非后续再生成的可微闭环。</li>
    <li><b>实测支持的主要算法缺陷：</b>实际AdamW更稳定降低辅助项而容许138次Dance上升；先验组合有真实物理反转案例；当前微批信任域限制不了其他请求和在线历史反馈；高幅度覆盖严重流失；新增累计目标后的新10仍恶化，不能宣布修复成功。</li>
    <li><b>仍需实验：</b>更广泛腿部/接触WM方向、14曲单种子的泛化、多anchor消费时段对因果结果的影响、跨请求持续信任域以及新合同Critic。没有机器人硬件结论。</li>
    <li>138次历史更新的损失/真实位移/输出变化均已审计，但每个更新之后的完整物理反事实无法从现有日志全部恢复；另做了16个真实候选两次滚动。旧600仅完成整曲重算，没有单独Newton参数方向实验（按要求方向版本为0/100/300/700）。</li>
    <li>本轮独立配对曲线为固定延迟回放；真实墙钟用于训练采集，未完成新0/10/25/50的第二套墙钟八曲评价，二者不混称。旧正式评价和所有失败证据保留。</li>
    <li>新episode Critic MC与双目标条件尚未实现和独立验证，V保持0，不能导入旧奖励Critic冒充兼容。身体相对姿态修正仅做离线诊断，未在本曲线中途改奖励。</li>
    <li>过程失败：临时预算监控shell引号错误已留日志并修正，错误命令未发信号；恢复入口返回值在启动前检查修正。真实超域拒绝不改阈值，未按质量负结果删除证据或停止原正式任务。</li>
    </ul>''')
    add('代码定位、建议与本轮实施状态',table(['问题','代码位置（gem/closedloop/mb_control）','建议与当前状态'],[
        ['局部SE2不能保留累计偏差','dance_losses.py:32；imagination.py:184；collector.py:136；rewards.py:95','已新增episode初始SE2及独立累计执行目标；新10/25仍负，不能宣布修好'],
        ['身体姿态重复计入世界转向','dance_losses.py:70；drift_orientation.py:25','已完成40曲离线相对姿态对照；本曲线没有中途改指标，修正奖励后仍需物理方向复验'],
        ['BC与先验主导部分实际AdamW步','teacher_training.py:200、281、309；drift_physical.py:151','已增加执行积分归一化、软辅助预算/投影与同anchor组合目标复验；硬限位不删除'],
        ['单微批信任域不能控制其他请求','output_trust_region.py:39；teacher_training.py:237','已验证完整回滚和当前候选Loss；跨歌曲、跨时段持久固定请求库尚未实施，是下一优先项'],
        ['教师时间线与真实请求动作身份不同','teacher.py:98、118；collector.py:173；online_teacher.py:103','已逐请求比较时间/噪声身份；应统一外生请求和目标动作身份，不能调宽相位容差'],
        ['H16新姿态直接消费覆盖少','dance_losses.py:131、142；imagination.py:193','已统计当前中心、速度halo、未来goal及整曲覆盖；4环境多anchor已实施，多消费时间窗和后续可微再生成仍未实现'],
        ['局部WM接触方向失真','validation.py:64；drift_physical.py:160','已做实际Newton同候选对照；保留两例相反方向，扩大动作/接触覆盖后才能扩大可信范围'],
        ['困难大幅度样本流失','sample_distribution.py:12、32；formal.py:467；drift_time_coverage.py:18','已区分五级漏斗与真实时间覆盖；不放宽状态支持域，困难动作采集比例仍需单独实验'],
        ['评价与模型选择','evaluation.py:62；dance_evaluation.py:17；drift_document.py:34','已做完整固定8曲、实际交付事件配对和逐曲报告；新墙钟整曲对照与多种子仍未完成'],
        ['新合同Critic','teacher_training.py:190；critic.py:13；short_audit.teacher_target_contract','lambdaV继续0；新MC/双目标条件Critic未重训，不能复用旧奖励价值模型']]))
    evidence=['objective_summary.json','offline_v3/trajectory_objectives.json','offline_v3/accepted_dance_worsening.json','offline_v3/candidate_history.jsonl','physical_direction_summary.json','physical_candidate_details.json','trust_drift_summary.json','gradient_conflict_summary.json','sampling_unique_summary.json','music_time_coverage.json','training_teacher_timing_comparison.json','reference_lifetimes/summary.json','conditional_drift/summary.json','body_orientation_comparison.json','paired_input_audit.json','decisive_report.json']
    if new_audit:evidence.append('new_trajectory_audit/trajectory_objectives.json')
    add('证据文件','<ul>'+''.join('<li><a href="'+p+'">'+p+'</a></li>' for p in evidence)+'</ul>')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(12,4.2))
    axes[0].plot([0,100,300,600,700],[obj[str(v)]['B_episode'] for v in [0,100,300,600,700]],'o-',label='Original run: full music')
    axes[0].plot([0,100,300,600,700],[obj[str(v)]['A_anchor_H16'] for v in [0,100,300,600,700]],'s-',label='Real local H16 mean')
    ready=[e for e in evals if e['loss'] is not None]
    axes[1].plot([e['update'] for e in ready],[e['loss'] for e in ready],'o-',label='Independent episode objective')
    axes[1].axhline(base,linestyle='--',color='gray',label='Frozen Stage1')
    for ax in axes:ax.set_xlabel('Accepted actor updates');ax.set_ylabel('Real music loss (lower is better)');ax.grid(alpha=.2);ax.legend(fontsize=8)
    fig.tight_layout();fig.savefig(root/'real_music_curves.png',dpi=160);plt.close(fig)
    style='body{font:16px/1.8 system-ui,sans-serif;max-width:1180px;margin:32px auto;padding:0 24px;color:#172338}h1{font-size:30px}h2{font-size:22px;margin-top:34px}table{border-collapse:collapse;width:100%;font-size:13px;margin:18px 0}td,th{border:1px solid #ccd4df;padding:8px;text-align:left;overflow-wrap:anywhere}th{background:#eaf0f8}pre{white-space:pre-wrap;background:#f1f4f8;padding:14px}a{color:#125ca0}img{max-width:100%}.note{background:#fff1d9;padding:16px;border-left:4px solid #e5a62c}section{page-break-inside:auto}'
    document='<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>GENMO持续退化：真实诊断与独立实验</title><style>'+style+'</style><body><h1>GENMO–SuperMimic 持续退化诊断</h1><p>'+state['generated_utc']+' · Server-2 · 原正式HEAD 3e1907c</p><p class="note">结论以完整真实Newton轨迹为准。此报告没有把单步梯度、模型回报、检查点或较低的重定义指标当作训练收益。未完成项明确保留。</p><img src="real_music_curves.png" alt="原正式与新独立实验真实音乐误差曲线">'+''.join(sections)+'</body></html>'
    (root/'GENMO_持续退化_决定性诊断报告.html').write_text(document)
    print(json.dumps(state,ensure_ascii=False))


if __name__=='__main__':main()
