from beam_hopping.scheduling.online import generate_horizon_instance, run_policy


def test_cross_time_policy_outputs_valid_metrics():
    horizon = generate_horizon_instance(
        instance_id="unit_horizon",
        T=4,
        M=4,
        N=8,
        K=4,
        topology="sparse",
        demand_profile="uniform",
        prediction_mode="noisy",
        edge_density=0.5,
        seed=19,
    )
    row = run_policy(horizon, "urgency_primal_dual")
    assert row["total_throughput"] > 0
    assert row["runtime"] >= 0
    assert 0 <= row["mean_demand_satisfaction_ratio"] <= 1
    assert 0 <= row["min_demand_satisfaction_ratio"] <= 1


def test_prediction_aware_policy_runs_for_all_prediction_modes():
    for mode in ["accurate", "noisy", "adversarial"]:
        horizon = generate_horizon_instance(
            instance_id=f"unit_{mode}",
            T=3,
            M=3,
            N=6,
            K=3,
            topology="complete",
            demand_profile="hotspot",
            prediction_mode=mode,
            edge_density=1.0,
            seed=23,
        )
        row = run_policy(horizon, "prediction_aware")
        assert row["policy"] == "prediction_aware"
        assert row["prediction_error"] >= 0
