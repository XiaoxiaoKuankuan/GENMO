"""使用同一份已保存的独立真实窗口比较多个WM，避免随机验证批次误导结论。

输入为VectorReplay验证过连续性的真实validation窗口及明确命名的WM checkpoint。
复用原SuperMimic状态定义、固定normalizer和FK，报告H1/4/8/16/32/64状态与速度误差。
不创建仿真、不训练模型、不修改checkpoint；结果只说明预测误差，不能作为Adapter
方向通过或真实整歌改善。窗口SHA、源码、模型参数SHA及各模型来源全部保留。
"""

from pathlib import Path
import argparse
import json
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    import torch
    from omegaconf import OmegaConf
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--windows', required=True)
    parser.add_argument('--model', action='append', default=[], help='清晰名称=checkpoint路径，可重复')
    parser.add_argument('--output', required=True)
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()
    output = Path(args.output).resolve()
    if output.exists():
        raise FileExistsError('固定标签比较报告不得覆盖旧证据')
    cfg = OmegaConf.to_container(OmegaConf.load(args.config), resolve=True)
    sys.path.insert(0, cfg['supermimic']['repo'])
    from gem.closedloop.reference_adapter.application import build_engine
    from gem.closedloop.reference_adapter.update_diagnostics import parameter_identity
    from supermimic.bridge.bundle import file_digest
    torch.set_num_threads(1)
    windows = torch.load(args.windows, weights_only=False, map_location=args.device)
    if windows.get('origin') != 'real' or windows['action'].shape[1] != 64 or not bool(windows['mask'].all()):
        raise ValueError('比较必须使用同一完整、连续且有效的真实64步标签')
    engine = build_engine(cfg, args.device)
    models = [('original_frozen_SM_WM', None)]
    for argument in args.model:
        name, path = argument.split('=', 1)
        if name in [n for n, _ in models]:
            raise ValueError('模型名称必须唯一')
        models.append((name, path))
    reports = {}
    for name, path in models:
        if path:
            ck = torch.load(path, weights_only=False, map_location='cpu', mmap=True)
            engine.dynamics.load_state_dict(ck['dynamics'])
        reports[name] = dict(source=path or cfg['supermimic']['checkpoint'],
            wm_sha256=parameter_identity(engine.dynamics),
            errors=engine.dynamics.validate_real(windows, kin=engine.step.runtime.kin))
    result = dict(models=reports, windows=args.windows, windows_sha256=file_digest(args.windows),
        samples=len(windows['state']), horizon=64, accepted_adapter_updates=0,
        source=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
        metrics='root_position/rotation=max over batch; q/qd/root_w/body_w=RMSE',
        boundary='same_independent_real_labels_prediction_only_not_physical_policy_gain')
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps(dict(output=str(output), models=list(reports)), ensure_ascii=False))


if __name__ == '__main__':
    main()
