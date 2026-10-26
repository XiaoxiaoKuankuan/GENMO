"""Adapter合同与真实权重数值测试的显式CPU资产夹具。

加载固定SM checkpoint和实际RobotSpec/MJCF，合成名义动作仅用于数值合同，不作为
真实物理样本或训练收益证据。所有仿真测试必须另走显式audit --physical入口；导入
测试不启动Isaac、不连接服务器、不访问或覆盖训练输出。资产缺失时明确skip。
"""

from pathlib import Path
import os
import sys

import pytest
import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, os.getenv("SUPERMIMIC_REPO", "/home/weili/bumi-closedloop-worktrees/supermimic/SuperMimic"))
torch.set_num_threads(1)


@pytest.fixture(scope="session")
def assets():
    from supermimic.bridge.bundle import load_bundle
    from supermimic.bridge.runtime import SuperMimicRuntime
    checkpoint = os.getenv("SM_BRIDGE_CHECKPOINT", "/home/weili/supermimic_vibed/logs/bumi3_seven_filtered_v1/ckpt_024000.pt")
    asset_dir = os.getenv("SM_ROBOT_ASSET_DIR", "/home/weili/supermimic_vibed/data/bumi3_4340/robot")
    if not Path(checkpoint).is_file() or not (Path(asset_dir)/"bumi3_4340.xml").is_file():
        pytest.skip("需要真实SM编号checkpoint和MJCF资产")
    bundle = load_bundle(checkpoint, robot_asset_dir=asset_dir)
    return bundle, SuperMimicRuntime(bundle)


@pytest.fixture
def system(assets):
    from supermimic.mdp import flatten_state
    from supermimic.rotations import so3_exp, mat_to_quat
    from gem.closedloop.reference_adapter.reference import make_nominal, nominal_window
    from gem.closedloop.reference_adapter.music_reward import AudioTrack, MusicReward
    from gem.closedloop.reference_adapter.policy import ReferenceAdapter
    from gem.closedloop.reference_adapter.rewards import Rewards
    from gem.closedloop.reference_adapter.collector import ControlStep, initial_state
    from gem.closedloop.reference_adapter.dynamics import DynamicsModel
    from gem.closedloop.reference_adapter.critic import AdapterCritic
    from gem.closedloop.reference_adapter.features import critic_dimension
    from gem.closedloop.reference_adapter.imagination import Imagination
    torch.manual_seed(123)
    bundle, runtime = assets
    n = 100
    t = torch.arange(n)/30
    root = torch.zeros(1, n, 3)
    root[..., 2] = bundle.spec.spawn_height
    root[..., 0] = .005 * torch.sin(t)
    rot = torch.zeros(1, n, 3)
    rot[..., 2] = .2+.01*torch.sin(t)
    q = torch.tensor(bundle.spec.default_joint_pos)[None, None] + .015*torch.sin(t[None, :, None]*4+torch.arange(21)[None, None]*.1)
    world = torch.cat((root, mat_to_quat(so3_exp(rot)), q.flip(-1)), -1)
    nominal = make_nominal(world, tuple(reversed(bundle.spec.joint_names)), bundle.spec)
    music = torch.zeros(1, n, 35)
    music[:, 6::15, 34] = 1
    audio = AudioTrack.from_features(music)
    adapter = ReferenceAdapter(bundle.spec)
    reward = Rewards(runtime, MusicReward(torch.ones(21)))
    step = ControlStep(adapter, bundle, runtime, reward)
    current, _, _ = nominal_window(nominal, 0, runtime.kin)
    state = flatten_state((current.root_p, current.root_R, current.root_v, current.root_w, current.q, current.qd))
    xi = torch.zeros(1, 25)
    xi[:, 0] = .75
    z = initial_state(dict(state=state, xi=xi, previous_action=torch.zeros(1, 21)), nominal, audio, step,
                      episode_id="tensor_fixture", song_id="fixture_song")
    dynamics = DynamicsModel(bundle)
    critic = AdapterCritic(critic_dimension(adapter))
    return Imagination(step, dynamics, critic), z
