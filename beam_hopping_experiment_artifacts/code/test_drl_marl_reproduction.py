from drl_marl_reproduction import (
    BeamHoppingRLEnv,
    RLConfig,
    all_cell_actions,
    evaluate_policy,
    train_cooperative_iql,
    train_single_agent_dqn,
)


def test_single_agent_action_space_matches_cell_selection_count():
    config = RLConfig(N=7, K=2, M=2, T_ttl=3)

    actions = all_cell_actions(config.N, config.K)

    assert len(actions) == 21
    assert all(len(action) == config.K for action in actions)
    assert all(len(set(action)) == config.K for action in actions)


def test_environment_step_serves_unique_cells_and_returns_metrics():
    config = RLConfig(N=7, K=2, M=2, T_ttl=3, seed=11)
    env = BeamHoppingRLEnv(config, seed=11)
    env.reset()

    _, reward, info = env.step([1, 1, 2])

    assert isinstance(reward, float)
    assert len(info["selected_cells"]) == config.K
    assert len(set(info["selected_cells"])) == config.K
    assert info["throughput_mbit"] >= 0.0
    assert 0.0 <= info["mean_satisfaction"] <= 1.0


def test_training_smoke_for_single_agent_and_cooperative_iql():
    config = RLConfig(N=7, K=2, M=2, T_ttl=3, seed=13)

    q_net, actions, rewards = train_single_agent_dqn(config, episodes=2, horizon=6, seed=13)
    agents, marl_rewards = train_cooperative_iql(config, episodes=2, horizon=6, seed=17)

    assert len(rewards) == 2
    assert len(marl_rewards) == 2
    assert len(actions) == 21
    assert len(agents) == config.K

    row = evaluate_policy(
        config,
        method="single_agent_dqn",
        horizon=4,
        repetitions=1,
        seed=19,
        q_net=q_net,
        actions=actions,
    )
    assert row["method"] == "single_agent_dqn"
    assert row["mean_inference_time_s"] >= 0.0
