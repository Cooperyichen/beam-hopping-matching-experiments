"""Exact fixed-resource matching with frozen cycle deficits, in bits.

Implements the policy in the archived conditional theorem. It does not check
its statistical assumptions, establish a chance guarantee, or optimize resources.
"""

from dataclasses import dataclass
from numbers import Integral

import numpy as np
from scipy.optimize import linear_sum_assignment


def _nonnegative(values, name, shape=None):
    array = np.array(values, dtype=float, copy=True)
    if shape is not None and array.shape != shape:
        raise ValueError(f"{name} must have shape {shape}")
    if not np.all(np.isfinite(array)) or np.any(array < 0):
        raise ValueError(f"{name} must be finite and nonnegative")
    return array


def _readonly(array):
    result = np.array(array, copy=True)
    result.setflags(write=False)
    return result


def exact_cardinality_matching(scores, cardinality, feasible_edges=None):
    """Return an integral optimal M by N matching with exactly K edges.

    Dummy vertices force K real edges. Forbidden real edges remain forbidden,
    including when every score is zero. No numerical penalty or greedy repair
    is used. SciPy's exact combinatorial assignment is the computation backend.
    """
    scores = _nonnegative(scores, "scores")
    if scores.ndim != 2 or min(scores.shape) < 1:
        raise ValueError("scores must be a nonempty matrix")
    beams, cells = scores.shape
    if isinstance(cardinality, bool) or not isinstance(cardinality, Integral):
        raise ValueError("cardinality must be an integer")
    if not 1 <= cardinality <= min(beams, cells):
        raise ValueError("cardinality must lie between 1 and min(M, N)")
    if feasible_edges is None:
        feasible = np.ones(scores.shape, dtype=bool)
    else:
        feasible = np.asarray(feasible_edges)
        if feasible.shape != scores.shape or feasible.dtype != np.dtype(bool):
            raise ValueError("feasible_edges must be a Boolean matrix matching scores")
    size = beams + cells - cardinality
    augmented = np.full((size, size), -np.inf)
    augmented[:beams, :cells] = np.where(feasible, scores, -np.inf)
    augmented[:beams, cells:] = 0.0
    augmented[beams:, :cells] = 0.0
    try:
        rows, cols = linear_sum_assignment(augmented, maximize=True)
    except ValueError as exc:
        raise ValueError("feasible graph has no matching of the requested cardinality") from exc
    selected = np.zeros(scores.shape, dtype=np.int8)
    real = (rows < beams) & (cols < cells)
    selected[rows[real], cols[real]] = 1
    if int(selected.sum()) != cardinality or np.any(selected.astype(bool) & ~feasible):
        raise RuntimeError("assignment violated the matching contract")
    return selected


@dataclass(frozen=True)
class CyclePlan:
    """Plan arrays use (slot, beam, cell); service and deficit arrays use cell."""

    cycle: int
    selected: np.ndarray
    nominal_service_bits: np.ndarray
    frozen_deficit_bits: np.ndarray
    score_bits_squared: float


class DeficitMatchingController:
    """One state stream for consecutive cycles and one realized service stream.

    plan() accepts decision-time nominal edge capacity only. observe() accepts
    demand and realized capacity after execution, exactly once per plan. Scores
    are Z*w, not (1+Z)*w and not normalized urgency. Zero scores are left tied;
    the exact assignment implementation resolves ties without a new objective.
    """

    def __init__(self, cells, initial_deficit_bits=None, first_cycle=1):
        if isinstance(cells, bool) or not isinstance(cells, Integral) or cells < 1:
            raise ValueError("cells must be a positive integer")
        if isinstance(first_cycle, bool) or not isinstance(first_cycle, Integral) or first_cycle < 1:
            raise ValueError("first_cycle must be a positive integer")
        self.cells = int(cells)
        initial = np.zeros(cells) if initial_deficit_bits is None else initial_deficit_bits
        self._deficit = _nonnegative(initial, "initial_deficit_bits", (cells,))
        self._cycle = int(first_cycle)
        self._pending = None

    @property
    def deficit_bits(self):
        return _readonly(self._deficit)

    def plan(self, nominal_edge_service_bits, cardinality, feasible_edges=None):
        if self._pending is not None:
            raise RuntimeError("observe the outstanding cycle before planning another")
        nominal = _nonnegative(nominal_edge_service_bits, "nominal_edge_service_bits")
        if nominal.ndim != 3 or nominal.shape[0] < 1 or nominal.shape[2] != self.cells:
            raise ValueError("nominal service must have shape (T, M, cells)")
        if feasible_edges is None:
            feasible = np.ones(nominal.shape, dtype=bool)
        else:
            feasible = np.asarray(feasible_edges)
            if feasible.shape != nominal.shape or feasible.dtype != np.dtype(bool):
                raise ValueError("feasible_edges must be Boolean with shape (T, M, cells)")
        with np.errstate(over="ignore"):
            scores = nominal * self._deficit[None, None, :]
        if not np.all(np.isfinite(scores)):
            raise ValueError("deficit weighted scores overflow; rescale units consistently")
        selected = np.stack([
            exact_cardinality_matching(scores[t], cardinality, feasible[t])
            for t in range(nominal.shape[0])
        ])
        service = np.sum(nominal * selected, axis=(0, 1))
        with np.errstate(over="ignore"):
            score = float(self._deficit @ service)
        if not np.all(np.isfinite(service)) or not np.isfinite(score):
            raise ValueError("cycle nominal service or score overflow")
        result = CyclePlan(self._cycle, _readonly(selected), _readonly(service),
                           _readonly(self._deficit), score)
        self._pending = result
        return result

    def observe(self, demand_bits, service_capacity_bits):
        """Apply [Z + true demand - realized allocated capacity]^+ in bits.

        Capacity is not capped at current demand. This is virtual accounting,
        not a physical queue departure model. Forecasts must not replace either
        observed quantity. Feedback validation failures leave the cycle pending.
        """
        if self._pending is None:
            raise RuntimeError("plan a cycle before providing feedback")
        demand = _nonnegative(demand_bits, "demand_bits", (self.cells,))
        service = _nonnegative(service_capacity_bits, "service_capacity_bits", (self.cells,))
        with np.errstate(over="ignore"):
            updated = np.maximum(self._deficit + demand - service, 0.0)
        if not np.all(np.isfinite(updated)):
            raise ValueError("deficit update overflow")
        receipt = {
            "cycle": self._cycle,
            "deficit_before_bits": self._deficit.tolist(),
            "demand_bits": demand.tolist(),
            "realized_service_capacity_bits": service.tolist(),
            "deficit_after_bits": updated.tolist(),
            "nominal_service_bits": self._pending.nominal_service_bits.tolist(),
            "score_bits_squared": self._pending.score_bits_squared,
        }
        self._deficit = updated
        self._cycle += 1
        self._pending = None
        return receipt

    def state_dict(self):
        """Checkpoint at a cycle boundary, after feedback and before a new plan."""
        if self._pending is not None:
            raise RuntimeError("checkpoint only after feedback at a cycle boundary")
        return {"schema_version": 1, "cells": self.cells, "next_cycle": self._cycle,
                "deficit_bits": self._deficit.tolist()}

    @classmethod
    def from_state_dict(cls, state):
        if state.get("schema_version") != 1:
            raise ValueError("unsupported deficit checkpoint schema")
        return cls(state["cells"], state["deficit_bits"], state["next_cycle"])
