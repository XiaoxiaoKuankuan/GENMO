"""汇集独立运动类别的真实方向证书，不扩大任何类别的数值范围。

输入为同一冻结资产、执行目标与数值验收标准的多个Newton报告，输出仍逐H、逐完整
运动/接触组合保存原证书。只允许互不重叠的组合；同一组合的正负结果不能通过挑选一个
报告覆盖另一个。所有原始行、未通过组合、文件SHA及选择配置一并保留，不把合并当成
新增仿真证据。全局范围只是原证据的索引，训练仍必须使用对应组合自己的范围。
这一步不加载GENMO、不产生梯度；WM实际动作输入补测仍由独立Newton入口完成。
"""

import copy


CRITERIA = ('epsilon', 'fd_atol', 'fd_rtol', 'direction_min_abs', 'state_rmse_limits',
            'same_start_atol', 'minimum_class_sources', 'minimum_class_anchors',
            'minimum_direction_fraction')


def merge_reports(reports, identities):
    if not reports or len(reports) != len(identities):
        raise ValueError('报告及身份清单必须非空且一一对应')
    first = reports[0]
    profiles, rows, forecasts = {}, [], []
    for report in reports:
        if report.get('schema') != 'genmo.joint_physics.validation.v2' or report.get('certification_scope') != 'exact_motion_and_contact_profile':
            raise ValueError('只能合并逐运动接触组合的本方法真实报告')
        if not report.get('frozen_unchanged') or not report.get('forecast_passed'):
            raise ValueError('冻结资产与请求前瞻必须已经通过')
        for key in ('assets', 'objective', 'classification', 'prediction_origin', 'checkpoint', 'student_physics_updates'):
            if report.get(key) != first.get(key):
                raise ValueError('报告合同不一致: '+key)
        if any(report['criteria'].get(k) != first['criteria'].get(k) for k in CRITERIA):
            raise ValueError('真实验证阈值不一致')
        if report.get('action_input_rows') is not None:
            raise ValueError('先合并方向证书，再统一执行真实WM动作输入补测，不能丢弃已有输入负结果')
        for horizon, certificates in report['profile_certificates'].items():
            target = profiles.setdefault(horizon, {})
            overlap = set(target) & set(certificates)
            if overlap:
                raise ValueError('同一组合的证据重叠，禁止覆盖或挑选: '+str(sorted(overlap)))
            target.update(copy.deepcopy(certificates))
        rows.extend(copy.deepcopy(report['rows']))
        forecasts.extend(copy.deepcopy(report.get('forecast_rows', [])))
    result = copy.deepcopy(first)
    result.update(profile_certificates=profiles, certificates={}, rows=rows, forecast_rows=forecasts,
                  source_ids=sorted({s for r in reports for s in r['source_ids']}),
                  supported_horizons=sorted(int(h) for h, cs in profiles.items() if any(c['passed'] for c in cs.values())),
                  audited_timesteps=sorted({k for r in reports for k in r.get('audited_timesteps', [])}),
                  component_reports=identities,
                  component_selection=[r['criteria'] for r in reports],
                  evidence='union_of_disjoint_motion_profiles_with_original_newton_rows')
    # 训练for_tags从对应证书取边界，不使用此全局索引代替类别门禁。
    from gem.closedloop.mb_control.validation import combine_envelopes
    result['state_envelope'] = combine_envelopes([r['state_envelope'] for r in reports])
    result['action_min'] = [min(v) for v in zip(*(r['action_min'] for r in reports))]
    result['action_max'] = [max(v) for v in zip(*(r['action_max'] for r in reports))]
    return result
