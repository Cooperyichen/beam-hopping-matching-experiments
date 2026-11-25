import random

from beam_hopping.matching.heuristics import (
    demand_priority_greedy,
    genetic_matching,
    greedy_matching,
    random_matching,
)
from beam_hopping.matching.single_slot import generate_instance


def _instance():
    return generate_instance(
        instance_id="baseline",
        M=5,
        N=7,
        K=4,
        topology="sparse",
        edge_density=0.5,
        demand_profile="hotspot",
        seed=31,
    )


def _assert_valid(inst, edges):
    assert len(edges) == inst.K
    assert len({b for b, _ in edges}) == inst.K
    assert len({i for _, i in edges}) == inst.K
    assert all(inst.feasible_edges[b, i] for b, i in edges)


def test_greedy_matching_is_feasible():
    inst = _instance()
    _assert_valid(inst, greedy_matching(inst))


def test_demand_priority_greedy_is_feasible():
    inst = _instance()
    _assert_valid(inst, demand_priority_greedy(inst))


def test_random_matching_is_feasible():
    inst = _instance()
    _assert_valid(inst, random_matching(inst, random.Random(3)))


def test_genetic_matching_is_feasible():
    inst = _instance()
    _assert_valid(inst, genetic_matching(inst, seed=5, population_size=10, generations=5))
