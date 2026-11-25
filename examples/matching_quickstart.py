"""Deterministic matching example; all edge services are in bits."""
import numpy as np

from synthetic_traffic_validation.deficit_matching import exact_cardinality_matching

weights = np.array([[100., 99.], [99., 0.]])
selected = exact_cardinality_matching(weights, cardinality=2)
print('Selected beam and cell pairs:', np.argwhere(selected).tolist())
print('Planned capacity:', float(np.sum(weights * selected)), 'bits')
