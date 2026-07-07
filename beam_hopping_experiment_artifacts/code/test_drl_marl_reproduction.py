import numpy as np
import torch

from drl_marl_reproduction import (
    BeamHoppingRLEnv,
    DuelingQNetwork,
    PrioritizedReplayBuffer,
    RLConfig,
    all_cell_actions,
    evaluate_policy,
    make_sequential_state,
    policy_greedy_throughput,
    policy_prediction_aware_accurate,
    policy_urgency_primal_dual,
    train_improved_cooperative_iql,
    train_sequential_dueling_dqn,
    train_cooperative_iql,
    train_single_agent_dqn,
    train_single_agent_dueling_dqn,
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


def test_online_matching_baselines_return_valid_cell_sets():
    config = RLConfig(N=7, K=2, M=2, T_ttl=3, seed=11)
    env = BeamHoppingRLEnv(config, seed=11)
    env.reset()

    for selected in [
        policy_greedy_throughput(env),
        policy_urgency_primal_dual(env),
        policy_prediction_aware_accurate(env, horizon=5),
    ]:
        assert len(selected) == config.K
        assert len(set(selected)) == config.K
        assert all(0 <= cell < config.N for cell in selected)


def test_evaluate_policy_supports_online_matching_baselines():
    config = RLConfig(N=7, K=2, M=2, T_ttl=3, seed=13)

    for method in ["greedy_throughput", "urgency_primal_dual", "prediction_aware_accurate"]:
        row = evaluate_policy(config, method=method, horizon=4, repetitions=1, seed=19)

        assert row["method"] == method
        assert row["mean_throughput_mbit"] >= 0.0
        assert 0.0 <= row["mean_satisfaction"] <= 1.0


def test_expanded_state_includes_queue_and_cell_features():
    config = RLConfig(N=5, K=2, M=2, T_ttl=3, use_expanded_state=True)
    env = BeamHoppingRLEnv(config, seed=3)
    state = env.reset()

    assert env.state_dim == config.N * config.T_ttl + 6 * config.N + 1
    assert state.shape == (env.state_dim,)


def test_dueling_q_network_returns_action_values():
    net = DuelingQNetwork(state_dim=11, action_dim=7, hidden_dim=16)

    values = net(torch.zeros((2, 11)))

    assert values.shape == (2, 7)


def test_prioritized_replay_updates_priorities():
    buffer = PrioritizedReplayBuffer(capacity=8, seed=5)
    state = np.zeros(3, dtype=np.float32)
    for action in range(4):
        buffer.push((state, action, float(action), state + 1.0))

    sample = buffer.sample(2)
    indices = sample[-1]
    buffer.update_priorities(indices, np.array([0.1, 2.0]))

    assert len(buffer) == 4
    assert max(buffer.priorities) > min(buffer.priorities)


def test_sequential_state_appends_mask_and_progress():
    base_state = np.zeros(10, dtype=np.float32)
    selected = [1, 3]

    seq_state = make_sequential_state(base_state, selected, N=5, K=4)

    assert seq_state.shape == (16,)
    assert seq_state[11] == 1.0
    assert seq_state[13] == 1.0
    assert seq_state[-1] == 0.5


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


def test_training_smoke_for_improved_rl_variants():
    config = RLConfig(
        N=7,
        K=2,
        M=2,
        T_ttl=3,
        seed=23,
        hidden_dim=32,
        replay_capacity=200,
        batch_size=8,
        epsilon_decay=0.90,
        use_expanded_state=True,
        reward_unmet_reduction_weight=0.5,
        reward_unmet_backlog_weight=0.2,
    )

    q_net, actions, rewards, losses = train_single_agent_dueling_dqn(config, episodes=2, horizon=6, seed=23)
    seq_net, seq_rewards, seq_losses = train_sequential_dueling_dqn(config, episodes=2, horizon=6, seed=29)
    agents, marl_rewards, marl_losses = train_improved_cooperative_iql(config, episodes=2, horizon=6, seed=31)

    assert len(rewards) == 2
    assert len(losses) == 2
    assert len(seq_rewards) == 2
    assert len(seq_losses) == 2
    assert len(marl_rewards) == 2
    assert len(marl_losses) == 2
    assert len(actions) == 21
    assert len(agents) == config.K

    row = evaluate_policy(
        config,
        method="sequential_dueling_dqn",
        horizon=4,
        repetitions=1,
        seed=37,
        sequential_q_net=seq_net,
    )
    assert row["method"] == "sequential_dueling_dqn"
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
