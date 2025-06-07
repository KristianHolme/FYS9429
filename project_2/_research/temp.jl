using Random

# Test with multiple shorter episodes to verify episode boundary handling
max_steps = 8
gamma = 1.0f0  # No discounting for simpler verification
gae_lambda = 1.0f0  # Monte Carlo returns
constant_value = 0.0f0  # Zero baseline
n_total_steps = 32  # Should get 4 episodes

# Create environment
env = BroadcastedParallelEnv([CustomEnv(max_steps)])

# Create policy with constant value function
policy = ConstantValuePolicy(DRiL.observation_space(env), DRiL.action_space(env), constant_value)
agent = ActorCriticAgent(policy; n_steps=n_total_steps, batch_size=n_total_steps, epochs=1, verbose=0)
alg = PPO(; gamma=gamma, gae_lambda=gae_lambda)

# Collect rollouts
roll_buffer = RolloutBuffer(DRiL.observation_space(env), DRiL.action_space(env), gae_lambda, gamma, n_total_steps, 1)
DRiL.collect_rollouts!(roll_buffer, agent, env)

rewards = roll_buffer.rewards
advantages = roll_buffer.advantages
returns = roll_buffer.returns

# With gamma=1.0, gae_lambda=1.0, and constant_value=0.0:
# For each episode, only the last step gets reward 1.0
# Monte Carlo return for last step = 1.0
# Monte Carlo return for all other steps = 1.0 (undiscounted future reward)
# Advantages = returns - values = returns - 0.0 = returns

# Check episode boundaries
episode_ends = findall(isapprox.(rewards, 1.0f0, atol=1e-5))

for episode_end in episode_ends
    episode_start = max(1, episode_end - max_steps + 1)

    # For Monte Carlo with gamma=1.0, all steps in episode should have return = 1.0
    for step in episode_start:episode_end
        @test isapprox(returns[step], 1.0f0, atol=1e-4)
        @test isapprox(advantages[step], 1.0f0, atol=1e-4)  # Since values = 0.0
    end
end

trajs = DRiL.collect_trajectories(agent, env, n_total_steps)