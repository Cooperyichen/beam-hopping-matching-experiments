from drl_marl_reproduction import (
    BeamHoppingRLEnv,
    RLConfig,
    all_cell_actions,
    evaluate_policy,
    train_cooperative_iql,
    train_single_agent_dqn,
    write_training_log,
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
    config = RLConfig(
        N=7,
        K=2,
        M=2,
        T_ttl=3,
        seed=13,
        hidden_dim=32,
        replay_capacity=200,
        batch_size=8,
        epsilon_decay=0.90,
    )

    q_net, actions, rewards, losses = train_single_agent_dqn(config, episodes=2, horizon=6, seed=13)
    agents, marl_rewards, marl_losses = train_cooperative_iql(config, episodes=2, horizon=6, seed=17)

    assert len(rewards) == 2
    assert len(losses) == 2
    assert len(marl_rewards) == 2
    assert len(marl_losses) == 2
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


def test_rl_config_exposes_training_hyperparameters():
    config = RLConfig(hidden_dim=256, replay_capacity=50000, batch_size=64, gamma=0.97)

    assert config.hidden_dim == 256
    assert config.replay_capacity == 50000
    assert config.batch_size == 64
    assert config.gamma == 0.97


def test_write_training_log(tmp_path):
    output = tmp_path / "training_log.csv"

    write_training_log(
        output,
        single_rewards=[1.0, 2.0],
        single_losses=[0.5, 0.25],
        cmadrl_rewards=[3.0],
        cmadrl_losses=[0.75],
    )

    text = output.read_text()
    assert "single_agent_dqn" in text
    assert "cmadrl_iql" in text
    assert "mean_loss" in text
