import itertools

import numpy as np
import pytest

from synthetic_traffic_validation.deficit_matching import (
    DeficitMatchingController, exact_cardinality_matching,
)


def brute(scores, k, feasible):
    candidates = []
    for rows in itertools.combinations(range(scores.shape[0]), k):
        for cols in itertools.permutations(range(scores.shape[1]), k):
            if all(feasible[r, c] for r, c in zip(rows, cols)):
                candidates.append(sum(scores[r, c] for r, c in zip(rows, cols)))
    return max(candidates)


@pytest.mark.parametrize("shape,k", [((2, 3), 1), ((2, 3), 2), ((3, 2), 1), ((3, 2), 2), ((3, 4), 2)])
def test_cardinality_matches_enumeration(shape, k):
    rng = np.random.default_rng(41)
    for _ in range(8):
        scores = rng.uniform(0, 10, shape)
        feasible = rng.random(shape) > .3
        feasible[np.arange(k), np.arange(k)] = True
        selected = exact_cardinality_matching(scores, k, feasible)
        assert selected.sum() == k
        assert np.all(selected.sum(0) <= 1) and np.all(selected.sum(1) <= 1)
        assert not np.any(selected.astype(bool) & ~feasible)
        assert np.sum(selected * scores) == pytest.approx(brute(scores, k, feasible))


def test_greedy_trap_is_solved_exactly():
    scores = np.array([[9., 8., 0.], [8., 0., 0.]])
    assert (exact_cardinality_matching(scores, 2) * scores).sum() == 16


def test_zero_score_still_respects_forbidden_edges_and_infeasibility():
    feasible = np.array([[False, True], [True, False]])
    np.testing.assert_array_equal(exact_cardinality_matching(np.zeros((2, 2)), 2, feasible), feasible)
    with pytest.raises(ValueError, match="no matching"):
        exact_cardinality_matching(np.ones((2, 2)), 2, np.array([[True, False], [True, False]]))


def test_frozen_deficit_cycle_score_matches_brute_force():
    z = np.array([2., 5., 1.])
    nominal = np.array([[[9., 8., 0.], [8., 0., 1.]], [[2., 4., 3.], [3., 1., 2.]]])
    controller = DeficitMatchingController(3, z)
    plan = controller.plan(nominal, 2)
    optimal = sum(brute(w * z, 2, np.ones((2, 3), bool)) for w in nominal)
    assert plan.score_bits_squared == pytest.approx(optimal)
    np.testing.assert_array_equal(controller.deficit_bits, z)
    np.testing.assert_array_equal(plan.frozen_deficit_bits, z)
    np.testing.assert_allclose(plan.nominal_service_bits, (nominal * plan.selected).sum((0, 1)))


def test_feedback_uses_real_capacity_not_nominal_or_current_demand_cap():
    controller = DeficitMatchingController(2, [10., 0.])
    controller.plan(np.array([[[100., 1.]]]), 1)
    controller.observe([1., 2.], [8., 0.])
    np.testing.assert_array_equal(controller.deficit_bits, [3., 2.])


def test_invalid_feedback_is_atomic_and_duplicate_feedback_is_rejected():
    controller = DeficitMatchingController(2)
    with pytest.raises(RuntimeError):
        controller.observe([1., 1.], [0., 0.])
    controller.plan(np.ones((1, 1, 2)), 1)
    with pytest.raises(RuntimeError):
        controller.plan(np.ones((1, 1, 2)), 1)
    with pytest.raises(ValueError):
        controller.observe([np.nan, 1.], [0., 0.])
    np.testing.assert_array_equal(controller.deficit_bits, [0., 0.])
    controller.observe([1., 1.], [1., 0.])
    with pytest.raises(RuntimeError):
        controller.observe([1., 1.], [1., 0.])


def test_checkpoint_preserves_contiguous_cycle_state():
    controller = DeficitMatchingController(2)
    controller.plan(np.ones((1, 1, 2)), 1)
    with pytest.raises(RuntimeError):
        controller.state_dict()
    controller.observe([2., 1.], [1., 0.])
    restored = DeficitMatchingController.from_state_dict(controller.state_dict())
    np.testing.assert_array_equal(restored.deficit_bits, [1., 1.])
    assert restored.plan(np.ones((1, 1, 2)), 1).cycle == 2


def test_archived_counterexample_has_bounded_debt_but_frequent_violations():
    controller = DeficitMatchingController(2)
    violations = np.zeros(2)
    for _ in range(40):
        plan = controller.plan(np.full((1, 1, 2), 3.), 1)
        service = plan.nominal_service_bits
        violations += service < 1.
        controller.observe([1., 1.], service)
        assert controller.deficit_bits.max() <= 1.
    np.testing.assert_array_equal(violations, [20., 20.])


@pytest.mark.parametrize("bad", [-1., np.nan, np.inf])
def test_invalid_nominal_service_and_deficit_are_rejected(bad):
    with pytest.raises(ValueError):
        DeficitMatchingController(1, [bad])
    with pytest.raises(ValueError):
        DeficitMatchingController(1).plan(np.array([[[bad]]]), 1)


def test_matching_objective_agrees_with_mosek_lp():
    cp = pytest.importorskip("cvxpy")
    if "MOSEK" not in cp.installed_solvers():
        pytest.skip("optional MOSEK validation requires its installed solver and license")
    scores = np.array([[2., 8., 1.], [7., 3., 2.], [1., 4., 6.]])
    x = cp.Variable(scores.shape)
    problem = cp.Problem(cp.Maximize(cp.sum(cp.multiply(scores, x))),
                         [x >= 0, x <= 1, cp.sum(x, 0) <= 1,
                          cp.sum(x, 1) <= 1, cp.sum(x) == 2])
    problem.solve(solver="MOSEK")
    assert problem.status == "optimal"
    assert (exact_cardinality_matching(scores, 2) * scores).sum() == pytest.approx(problem.value, abs=1e-6)
