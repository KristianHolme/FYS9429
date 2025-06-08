using DrWatson
@quickactivate :project_2
using Lux
using DRiL
using Zygote
using WGLMakie
WGLMakie.activate!()
# using CairoMakie
using Statistics
using LinearAlgebra
using BenchmarkTools
using ClassicControlEnvironments
##
max_steps = 8
gamma = 0.99f0
gae_lambda = 0.95f0
constant_value = 0.5f0

# Create environment that gives reward 1.0 only at final step
env = BroadcastedParallelEnv([CustomEnv(max_steps)])
number_of_envs(env)
# Create policy with constant value function
policy = ConstantValuePolicy(DRiL.observation_space(env), DRiL.action_space(env), constant_value)
agent = ActorCriticAgent(policy; n_steps=max_steps, batch_size=max_steps, epochs=1, verbose=0)
alg = PPO(; gamma=gamma, gae_lambda=gae_lambda)
actions_space = action_space(policy)
rand(actions_space, 2)
# Collect rollouts
roll_buffer = RolloutBuffer(DRiL.observation_space(env), DRiL.action_space(env), gae_lambda, gamma, max_steps, 1)
DRiL.collect_rollouts!(roll_buffer, agent, env)


##
obs_space = Box(Float32[-1.0, -1.0], Float32[1.0, 1.0])
discrete_action_space = Discrete(4, 0)
continuous_action_space = Box(Float32[-1.0], Float32[1.0])

discrete_policy = DiscreteActorCriticPolicy(obs_space, discrete_action_space)
continuous_policy = ContinuousActorCriticPolicy(obs_space, continuous_action_space)

# Test that both policies implement the same interface
rng = Random.MersenneTwister(42)

discrete_params = Lux.initialparameters(rng, discrete_policy)
discrete_states = Lux.initialstates(rng, discrete_policy)

continuous_params = Lux.initialparameters(rng, continuous_policy)
continuous_states = Lux.initialstates(rng, continuous_policy)

obs = Float32[0.5, -0.3]
batched_obs = reduce(hcat, [obs])

# Test that both implement the same methods
discrete_actions, discrete_values, discrete_log_probs, _ = discrete_policy(batched_obs, discrete_params, discrete_states)
continuous_actions, continuous_values, continuous_log_probs, _ = continuous_policy(batched_obs, continuous_params, continuous_states)

# Test predict
discrete_pred, _ = DRiL.predict_actions(discrete_policy, batched_obs, discrete_params, discrete_states)
continuous_pred, _ = DRiL.predict_actions(continuous_policy, batched_obs, continuous_params, continuous_states)

# Test predict_values
discrete_vals, _ = predict_values(discrete_policy, batched_obs, discrete_params, discrete_states)
continuous_vals, _ = predict_values(continuous_policy, batched_obs, continuous_params, continuous_states)

# Test evaluate_actions
batched_discrete_actions = reduce(hcat, discrete_actions)
batched_continuous_actions = reduce(hcat, continuous_actions)

discrete_eval_values, discrete_eval_log_probs, discrete_entropy, _ = DRiL.evaluate_actions(discrete_policy, batched_obs, batched_discrete_actions, discrete_params, discrete_states)
continuous_eval_values, continuous_eval_log_probs, continuous_entropy, _ = DRiL.evaluate_actions(continuous_policy, batched_obs, batched_continuous_actions, continuous_params, continuous_states)


ds = DRiL.get_distributions(continuous_policy, continuous_actions, continuous_params.log_std)

# Test that outputs have expected types and shapes
@test discrete_actions isa AbstractArray{<:Integer}
@test continuous_actions isa AbstractArray{<:Real}
@test discrete_pred isa AbstractArray{<:Integer}  
@test continuous_pred isa AbstractArray{<:Real}
@test discrete_vals isa AbstractArray{<:Real}
@test continuous_vals isa AbstractArray{<:Real}
@test length(discrete_eval_log_probs) == 1
@test length(continuous_eval_log_probs) == 1