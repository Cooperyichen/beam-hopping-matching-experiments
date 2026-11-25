import math

import numpy as np

from beam_hopping.matching.single_slot import build_topology, brute_force_optimum, generate_instance


def test_edge_weight_is_capacity_not_demand_bound():
    inst = generate_instance(
        instance_id="unit",
        M=4,
        N=5,
        K=3,
        topology="complete",
        edge_density=1.0,
        demand_profile="hotspot",
        seed=7,
    )
    expected = inst.T_slot * inst.B0 * np.log2(1.0 + inst.P0 * inst.gain / inst.sigma2.reshape(1, inst.N))
    assert np.allclose(inst.weight, expected)
    assert not np.allclose(inst.weight[0], inst.T_slot * inst.demand_rate)


def test_topology_has_no_empty_beam_or_cell():
    rng = np.random.default_rng(1)
    mat = build_topology(6, 9, "sparse", 0.15, rng)
    assert mat.shape == (6, 9)
    assert mat.any(axis=0).all()
    assert mat.any(axis=1).all()


def test_bruteforce_optimum_is_finite_for_small_complete_graph():
    inst = generate_instance(
        instance_id="small",
        M=3,
        N=4,
        K=2,
        topology="complete",
        edge_density=1.0,
        demand_profile="uniform",
        seed=11,
    )
    value = brute_force_optimum(inst)
    assert math.isfinite(value)
    assert value > 0
