"""Deterministic feedback example, not a workload stability experiment."""
import numpy as np

from synthetic_traffic_validation.deficit_matching import DeficitMatchingController

controller = DeficitMatchingController(cells=2)
edge_service_bits = np.array([[[3., 3.]]])  # T=1, M=1, N=2
for cycle in range(1, 5):
    plan = controller.plan(edge_service_bits, cardinality=1)
    # In this toy example only, realized service equals planned service.
    receipt = controller.observe(np.ones(2), plan.nominal_service_bits)
    print('Cycle', cycle, 'deficit:', receipt['deficit_after_bits'], 'bits')
