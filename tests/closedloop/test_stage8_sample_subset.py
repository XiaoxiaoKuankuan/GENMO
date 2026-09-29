"""验证Stage8确定性音乐清单的精确补测接口，不启动模型或物理仿真。

补测必须在既有完整清单内选择，拒绝未知、重复和空白名单；保留父清单顺序、
跨数据集同名样本身份及可复核SHA。测试确保重跑一个样本不会改变选样规则，
也不能借补测接口混入任意曲目后冒充200首固定评估。
"""
import copy

import pytest

from tools.eval.run_closedloop_baseline import select_evaluation_subset


def samples():
    return [{"dataset": "FineDance", "row": {"sample_id": "shared", "split": "train"}},
            {"dataset": "Mine", "row": {"sample_id": "shared", "split": "train"}},
            {"dataset": "Mine", "row": {"sample_id": "中文/曲目", "split": "train"}}]


def test_exact_subset_keeps_parent_order_and_namespace_without_mutation():
    parent = samples()
    before = copy.deepcopy(parent)
    selected, scope = select_evaluation_subset(parent, ["Mine/中文/曲目", "FineDance/shared"])
    assert selected == [parent[0], parent[2]]
    assert parent == before
    assert scope["parent_music_count"] == 3 and scope["selected_music_count"] == 2
    assert scope["parent_order_preserved"]
    assert scope["parent_selection_sha256"] == select_evaluation_subset(parent)[1]["parent_selection_sha256"]
    assert scope["parent_selection_sha256"] != select_evaluation_subset(parent[:2])[1]["parent_selection_sha256"]


@pytest.mark.parametrize("keys", [[], ["Mine/shared", "Mine/shared"], ["Mine/missing"], ["shared"]])
def test_subset_rejects_invalid_requests(keys):
    with pytest.raises(ValueError):
        select_evaluation_subset(samples(), keys)


def test_subset_rejects_duplicate_parent_keys():
    with pytest.raises(ValueError, match="duplicate"):
        select_evaluation_subset(samples() + samples()[:1])
