import numpy as np

from synthetic_traffic_validation.deficit_demo import run


def test_demo_exposes_periodic_violation_despite_bounded_deficit():
    result = run(cycles=6)
    assert len(result["cycles"]) == 6
    assert result["checkpoint"]["next_cycle"] == 7
    assert max(result["checkpoint"]["deficit_bits"]) <= 1


def test_existing_traffic_adapter_uses_bits_and_contiguous_cycles(tmp_path):
    path = tmp_path / "traffic.npz"
    np.savez(path, traffic_bits=np.full((8, 36), 2e5))
    result = run(cycles=3, traffic_path=path, origin=0, horizon=12)
    assert [r["origin"] for r in result["cycles"]] == [0, 12, 24]
    for row in result["cycles"]:
        np.testing.assert_allclose(row["demand_bits"], np.full(8, 2.4e6))


def test_future_traffic_cannot_change_first_cycle_plan_or_feedback(tmp_path):
    first, second = tmp_path / "a.npz", tmp_path / "b.npz"
    a = np.full((8, 24), 2e5)
    b = a.copy()
    b[:, 12:] *= 10
    np.savez(first, traffic_bits=a)
    np.savez(second, traffic_bits=b)
    left = run(2, first, 0, 12, 29)["cycles"][0]
    right = run(2, second, 0, 12, 29)["cycles"][0]
    assert left == right
