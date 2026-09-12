import numpy as np

from wormchess import MoveFeatures, WormChessPolicy


MOVES = [
    ("a1a2", MoveFeatures(0, 0, 0, .14, .8)),
    ("b1c3", MoveFeatures(.14, 0, .29, .29, .6, center=.8, development=1)),
]


def test_deterministic_choice_and_sorted_scores():
    a, b = WormChessPolicy(), WormChessPolicy()
    assert a.choose(MOVES) == b.choose(MOVES)


def test_connectome_is_frozen_and_complete():
    policy = WormChessPolicy()
    assert len(policy.graph.names) == 302
    assert not policy.graph.matrix.flags.writeable
    assert not policy.projection.flags.writeable


def test_no_edge_control_changes_candidate_state():
    policy = WormChessPolicy()
    _, intact = policy.evaluate(MOVES[1][1])
    _, control = policy.evaluate(MOVES[1][1], no_edges=True)
    assert not np.allclose(intact, control)


def test_policy_refuses_empty_legal_list():
    try:
        WormChessPolicy().choose([])
    except ValueError as exc:
        assert "no legal" in str(exc)
    else:
        raise AssertionError("empty legal list should fail")
