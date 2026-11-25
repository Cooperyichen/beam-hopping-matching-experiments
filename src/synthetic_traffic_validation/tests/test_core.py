import itertools
import numpy as np
import pytest

from synthetic_traffic_validation.traffic import bin_intervals, simulate_cell, make_config
from synthetic_traffic_validation.forecast import split_bounds, fractional_weights, fit_arfima, prediction_weights, predict_at
from synthetic_traffic_validation.allocation import exact_matching, solve_allocation, service_margin, trust_target


def test_interval_conservation():
    import torch
    starts = torch.tensor([0.0, .1, .37, .9], dtype=torch.float64)
    ends = torch.tensor([.05, .4, .81, 1.2], dtype=torch.float64)
    occ = bin_intervals(starts, ends, 10, .1)
    assert float(occ.sum()) == pytest.approx(.05 + .3 + .44 + .1)
    assert np.all(occ.numpy() >= -1e-12)


def test_stationary_mean_and_seed():
    cfg = make_config(steps=10000, alpha=1.9)
    a = simulate_cell(64, 10000, .002, 1.9, .01, 1e6, 12, 'cpu', batch=32)
    b = simulate_cell(64, 10000, .002, 1.9, .01, 1e6, 12, 'cpu', batch=32)
    np.testing.assert_array_equal(a, b)
    assert a.mean() == pytest.approx(64000., rel=.13)
    assert a.min() >= -1e-6 and a.max() <= 128000. + 1e-6
    assert cfg['steps'] == 10000


def test_split_and_filter():
    assert split_bounds(1000) == (600, 700, 800)
    np.testing.assert_allclose(fractional_weights(0, 5), [1, 0, 0, 0, 0])
    rng = np.random.default_rng(2)
    x = rng.normal(size=30000)
    y = np.empty_like(x); y[0] = 0
    for t in range(1, len(y)):
        y[t] = .7*y[t-1] + x[t]
    fit = fit_arfima(y[:20000], 1, d_fixed=0)
    assert fit['phi'][0] == pytest.approx(.7, abs=.05)
    W = prediction_weights(fit, 128, 12)
    # Known AR(1) direct predictors; row 0 multiplies latest observation.
    assert W[0] == pytest.approx(sum(.7**j for j in range(1, 13)), rel=.15)
    a = predict_at(y, np.array([21000, 22000]), W, fit['mean'], 12)
    changed = y.copy(); changed[22000:] += 10000
    b = predict_at(changed, np.array([21000, 22000]), W, fit['mean'], 12)
    np.testing.assert_allclose(a, b)


def test_exact_matching_and_mosek_lp():
    import cvxpy as cp
    w = np.array([[9., 8., 0.], [8., 0., 0.]])
    x = exact_matching(w)
    brute = max(sum(w[j, i] for j, i in enumerate(p)) for p in itertools.permutations(range(3), 2))
    assert (x*w).sum() == brute == 16
    z = cp.Variable(w.shape)
    p = cp.Problem(cp.Maximize(cp.sum(cp.multiply(w,z))), [z>=0,z<=1,cp.sum(z,axis=0)<=1,cp.sum(z,axis=1)==1])
    p.solve(solver='MOSEK')
    assert p.value == pytest.approx(brute, abs=1e-6)


def test_trust_endpoints_and_conic_repair():
    pred = np.array([1.,2.]); reserve = np.array([.1,.2])
    np.testing.assert_allclose(trust_target(pred,reserve,np.ones(2)), pred)
    np.testing.assert_allclose(trust_target(pred,reserve,np.zeros(2)), pred+reserve)
    mu = np.array([[4.,4.,4.,4.], [3.,3.,3.,3.]])
    sd = .2*mu; target = np.array([1.,1.])
    sol = solve_allocation(mu,sd,target,k=1,chance=True,bmax=.6)
    assert sol['bandwidth'] is not None
    assert np.min(service_margin(sol['bandwidth'],mu,sd,target)) >= -1e-5
    bad = solve_allocation(mu,sd,np.array([1000.,1000.]),k=1,chance=True)
    assert bad['bandwidth'] is None
