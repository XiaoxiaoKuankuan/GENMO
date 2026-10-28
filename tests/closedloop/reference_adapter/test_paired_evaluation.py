"""真实评价报告的配对身份回归，不伪造物理改善。

只验证报告必须共享初态、物理参数和预算，以及失败前缀使用完整预算分数的差值。
实际物理执行另由Server-1已有checkpoint配对完成，此处不生成训练样本。
"""
import copy
import pytest
from gem.closedloop.reference_adapter.paired_evaluation import paired_comparison


def report():
    row=dict(task='song',initial_state_sha256='state',physical_xi_sha256='xi',evaluation_budget_steps=100,
        reason='music_end',steps=100,full_horizon_tracking=.09,root_position_rmse_m=.1,joint_q_rmse_rad=.1,
        joint_qd_rmse_rad_s=.5,body_orientation_rmse_rad=.2,jerk_rms=20.,contact_foot_slip_mps=.01,
        sm_terms={k:.01 for k in ('body_pos','body_ori','body_linvel','body_angvel','root_pos','root_ori')})
    return dict(episodes=[row],completion_rate=1.,mean_full_horizon_tracking=.09)


def test_paired_evaluation_rejects_different_initial_physics_or_budget():
    zero=report()
    for field in ('initial_state_sha256','physical_xi_sha256','evaluation_budget_steps'):
        candidate=copy.deepcopy(zero);candidate['episodes'][0][field]='changed'
        with pytest.raises(ValueError,match='配对条件'):
            paired_comparison(zero,candidate)


def test_failed_prefix_cannot_claim_full_song_improvement():
    zero=report();candidate=copy.deepcopy(zero)
    candidate.update(completion_rate=0.,mean_full_horizon_tracking=.05)
    candidate['episodes'][0].update(reason='physical_failure',steps=50,full_horizon_tracking=.05)
    difference=paired_comparison(zero,candidate)
    assert difference['completion_delta']==-1
    assert difference['tracking_delta']<0
    assert difference['cases'][0]['current_steps']==50


def test_projection_exception_rejects_only_bad_evaluation_environment():
    import torch
    from gem.closedloop.reference_adapter.paired_evaluation import _feasible_evaluation_proposal
    class Control:
        def propose(self, z, zero=False):
            bad=z['frame']==3
            if bad.any():
                error=RuntimeError('float32不可执行');error.bad_mask=bad
                raise error
            return {'action':z['frame'].float()[:,None].expand(-1,21)}
        def constraint_certificate(self,z,proposal):
            return {'feasible':torch.ones(len(z['frame']),dtype=torch.bool)}
    ids,z,proposal,failures=_feasible_evaluation_proposal(Control(),{'frame':torch.tensor([1,3,5])},
        torch.arange(3),zero=False)
    assert ids.tolist()==[0,2]
    assert proposal['action'].shape==(2,21)
    assert failures[0]['env']==1 and failures[0]['frame']==3
