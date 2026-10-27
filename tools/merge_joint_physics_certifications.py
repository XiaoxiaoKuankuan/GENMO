"""将互不重叠的H/运动接触组合真实方向报告汇集到独立新目录。

所有报告必须使用相同资产、目标、阈值与学生身份，每个组合保持自己的已审计噪声等级。
完整保留正负探针，不覆盖已有
报告、不改变分层状态/动作范围；组合重叠时直接报错，防止择优挑选证书。输出可继续交给
validate_joint_physics_action_domain.py做真实WM动作输入补测，本命令本身不算物理验收。
"""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report', action='append', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    from gem.closedloop.joint_physics.certificates import merge_reports
    from gem.closedloop.joint_physics.contracts import digest
    result = merge_reports([json.loads(Path(p).read_text()) for p in args.report],
                           [{'path': str(Path(p).resolve()), 'sha256': digest(p)} for p in args.report])
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    (output/'validation.json').write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
