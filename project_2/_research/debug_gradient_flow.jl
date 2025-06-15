using DrWatson
@quickactivate :project_2
using Lux
using DRiL
using Zygote
using WGLMakie
WGLMakie.activate!()
using Statistics
using LinearAlgebra
using ClassicControlEnvironments
##

ps = rand(1:4, 2, 2, 3) .|> Float32
log_std = Float32[0.1 0.2 ; 0.3 0.4]

actions = rand(Float32,2,2,3)

function myloss(ps)
    ds = DiagGaussian.(eachslice(ps, dims=ndims(ps)), Ref(log_std))
    return mean(logpdf.(ds, eachslice(actions, dims=ndims(actions))))
end

Zygote.gradient(myloss, ps)


##
stats_window_size = 50
alg = PPO(; ent_coef=0.00999f0, vf_coef=0.480177f0, gamma=0.990886f0, gae_lambda=0.85821f0, clip_range=0.132141f0)
pendenv = BroadcastedParallelEnv([PendulumEnv() |> ScalingWrapperEnv for _ in 1:8])
pendenv = MonitorWrapperEnv(pendenv, stats_window_size)
pendenv = NormalizeWrapperEnv(pendenv, gamma=alg.gamma)

pendpolicy = ActorCriticPolicy(observation_space(pendenv), action_space(pendenv))
pendagent = ActorCriticAgent(pendpolicy; verbose=2, n_steps=128, batch_size=128, learning_rate=1.95409f-4, epochs=20)

##

obs = rand(observation_space(pendenv), 3)
batched_obs = reduce(hcat, obs)

actions = rand(action_space(pendenv), 3)
batched_actions = reduce(hcat, actions)

ps = pendagent.train_state.parameters
st = pendagent.train_state.states
function myloss2(ps)
    feats, _ = DRiL.extract_features(pendpolicy, batched_obs, ps, st)
    new_action_means, _ = DRiL.get_actions_from_features(pendpolicy, feats, ps, st)
    distributions = DRiL.get_distributions(pendpolicy, new_action_means, ps.log_std)
    log_probs = logpdf.(distributions, eachslice(batched_actions, dims=ndims(batched_actions)))
    entropies = entropy.(distributions)
    return mean(log_probs + entropies)
end
myloss2(ps)


##
grads = Zygote.gradient(myloss2, ps)
grads[1].actor_head


##
function calculate_log_probs(action_mean, log_std, action)
    # Calculate log probability for diagonal Gaussian distribution
    # action_mean: mean vector
    # log_std: log standard deviation vector 
    # action: action vector
    
    # Compute difference from mean
    diff = action .- action_mean
    
    # Calculate log probability components
    log_2pi = Float32(log(2π))
    variance_term = sum(2 .* log_std)
    quadratic_term = sum((diff .* diff) ./ (2 .* exp.(2 .* log_std)))
    
    # Sum components for final log probability
    log_prob = -0.5f0 * (log_2pi + variance_term + quadratic_term)
    
    # Sum across action dimensions
    @assert log_prob isa Float32
    return log_prob
end

function calculate_entropy(log_std)
    # Calculate entropy for diagonal Gaussian distribution
    # log_std: log standard deviation vector
    log_2pi = Float32(log(2π))
    return sum(0.5f0 .* (log_2pi .+ 2 .* log_std))
end

actions = rand(action_space(pendenv), 3)
batched_actions = reduce(hcat, actions)
function myloss3(ps)
    log_std = ps.log_std
    feats, _ = DRiL.extract_features(pendpolicy, batched_obs, ps, st)
    new_action_means, _ = DRiL.get_actions_from_features(pendpolicy, feats, ps, st)
    log_probs = calculate_log_probs.(eachslice(new_action_means, dims=ndims(new_action_means)), Ref(log_std), eachslice(batched_actions, dims=ndims(batched_actions)))
    return mean(log_probs)
end

Zygote.gradient(myloss3, ps)

## Full run, this gives zero actor head gradients
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
using ClassicControlEnvironments
using ReverseDiff
using Optimisers
##
stats_window_size = 50
alg = PPO(; ent_coef=0f0, vf_coef=0.480177f0, gamma=0.990886f0, gae_lambda=0.85821f0, clip_range=0.132141f0)
pendenv = BroadcastedParallelEnv([PendulumEnv() |> ScalingWrapperEnv for _ in 1:8])
pendenv = MonitorWrapperEnv(pendenv, stats_window_size)
pendenv = NormalizeWrapperEnv(pendenv, gamma=alg.gamma)

pendpolicy = ActorCriticPolicy(observation_space(pendenv), action_space(pendenv))
pendagent = ActorCriticAgent(pendpolicy; verbose=2, n_steps=128, batch_size=128, learning_rate=1.95409f-4, epochs=2)
learn_stats = learn!(pendagent, pendenv, alg, AutoZygote(); max_steps=1025)
##
single_env = PendulumEnv()
obs, actions, rewards = collect_trajectory(pendagent, single_env; norm_env=pendenv)
actions
sum(rewards)
fig_traj = plot_trajectory(PendulumEnv(), obs, actions, rewards)
plot_trajectory_interactive(PendulumEnv(), obs, actions, rewards)
## Distributions and logprobs, this computes gradients fine

action_shp = (1,)
obs_shp = (6, 2)
batch_size = 8
activation = tanh
log_std = randn(Float32, action_shp...)
model = Chain(FlattenLayer(), Dense(prod(obs_shp), 64, activation), Dense(64, 64, activation), Dense(64, prod(action_shp)), ReshapeLayer(action_shp))
ps, st = Lux.setup(Random.MersenneTwister(42), model)
train_state = Lux.Training.TrainState(model, ps, st, Adam())

function myloss4(model, ps, st, data)
    obs = data[1]
    actions = data[2]
    log_std = data[3]
    outputs, st = model(obs, ps, st)
    distributions = DiagGaussian.(eachslice(outputs, dims=ndims(outputs)), Ref(log_std))
    log_probs = logpdf.(distributions, eachslice(actions, dims=ndims(actions)))
    return mean(log_probs), st, Dict()
end
##
batched_obs = rand(obs_shp..., batch_size) .|> Float32
batched_actions = rand(action_shp..., batch_size) .|> Float32
data = (batched_obs, batched_actions, log_std)
##
grads, loss, stats, training_state = Lux.Training.compute_gradients(AutoZygote(), myloss4, data, train_state)
grads

## Using data from rollout buffer is fine, gives nonzero gradients
using MLUtils
using ComponentArrays
action_shp = size(action_space(pendenv))
obs_shp = size(observation_space(pendenv))
batch_size = pendagent.batch_size
activation = tanh
log_std = randn(Float32, action_shp...)
model = Chain(FlattenLayer(), Dense(prod(obs_shp), 64, activation), Dense(64, 64, activation), Dense(64, prod(action_shp)), ReshapeLayer(action_shp))
ps, st = Lux.setup(Random.MersenneTwister(42), model)
train_state = Lux.Training.TrainState(model, ps, st, Adam())

roll_buffer = RolloutBuffer(observation_space(pendenv), action_space(pendenv), alg.gae_lambda, alg.gamma, pendagent.n_steps, number_of_envs(pendenv))
DRiL.collect_rollouts!(roll_buffer, pendagent, pendenv)
data_loader = DataLoader((roll_buffer.observations, roll_buffer.actions,
                roll_buffer.advantages, roll_buffer.returns,
                roll_buffer.logprobs, roll_buffer.values),
            batchsize=pendagent.batch_size, shuffle=true, parallel=true, rng=pendagent.rng)

function myloss5(model, ps, st, batch_data)
    observations = batch_data[1]
    actions = batch_data[2]
    advantages = batch_data[3]
    returns = batch_data[4]
    old_logprobs = batch_data[5]
    old_values = batch_data[6]
    outputs, st = model(observations, ps, st)
    distributions = DiagGaussian.(eachslice(outputs, dims=ndims(outputs)), Ref(log_std))
    log_probs = logpdf.(distributions, eachslice(actions, dims=ndims(actions)))
    return mean(log_probs), st, Dict()
end

##
grads, loss, stats, training_state = Lux.Training.compute_gradients(AutoZygote(), myloss5, first(data_loader), train_state)
grads
norm(ComponentArray(grads))

##
## This gives zero gradients
using MLUtils
stats_window_size = 50
alg = PPO(; ent_coef=0f0, vf_coef=0.480177f0, gamma=0.990886f0, gae_lambda=0.85821f0, clip_range=0.132141f0)
pendenv = BroadcastedParallelEnv([PendulumEnv() |> ScalingWrapperEnv for _ in 1:8])
pendenv = MonitorWrapperEnv(pendenv, stats_window_size)
pendenv = NormalizeWrapperEnv(pendenv, gamma=alg.gamma)
action_shp = size(action_space(pendenv))
obs_shp = size(observation_space(pendenv))

pendpolicy = ActorCriticPolicy(observation_space(pendenv), action_space(pendenv))
pendagent = ActorCriticAgent(pendpolicy; verbose=2, n_steps=128, batch_size=128, learning_rate=1.95409f-4, epochs=2)
roll_buffer = RolloutBuffer(observation_space(pendenv), action_space(pendenv), alg.gae_lambda, alg.gamma, pendagent.n_steps, number_of_envs(pendenv))
DRiL.collect_rollouts!(roll_buffer, pendagent, pendenv)
data_loader = DataLoader((roll_buffer.observations, roll_buffer.actions,
                roll_buffer.advantages, roll_buffer.returns,
                roll_buffer.logprobs, roll_buffer.values),
            batchsize=pendagent.batch_size, shuffle=true, parallel=true, rng=pendagent.rng)
train_state = pendagent.train_state
function myloss6(policy, ps, st, batch_data)
    observations = batch_data[1]
    actions = batch_data[2]
    advantages = batch_data[3]
    returns = batch_data[4]
    old_logprobs = batch_data[5]
    old_values = batch_data[6]
    values, log_probs, entropy, st = DRiL.evaluate_actions(policy, observations, actions, ps, st)
    return mean(log_probs), st, Dict()
end

##
grads, loss, stats, train_state = Lux.Training.compute_gradients(AutoZygote(), myloss6, first(data_loader), train_state)
grads

## This gives nonzero gradients, problem with dataloader?
stats_window_size = 50
alg = PPO(; ent_coef=0f0, vf_coef=0.480177f0, gamma=0.990886f0, gae_lambda=0.85821f0, clip_range=0.132141f0)
pendenv = BroadcastedParallelEnv([PendulumEnv() |> ScalingWrapperEnv for _ in 1:8])
pendenv = MonitorWrapperEnv(pendenv, stats_window_size)
pendenv = NormalizeWrapperEnv(pendenv, gamma=alg.gamma)
action_shp = size(action_space(pendenv))
obs_shp = size(observation_space(pendenv))

pendpolicy = ActorCriticPolicy(observation_space(pendenv), action_space(pendenv))
pendagent = ActorCriticAgent(pendpolicy; verbose=2, n_steps=128, batch_size=128, learning_rate=1.95409f-4, epochs=2)

train_state = pendagent.train_state
function myloss7(policy, ps, st, batch_data)
    observations = batch_data[1]
    actions = batch_data[2]
    values, log_probs, entropy, st = DRiL.evaluate_actions(policy, observations, actions, ps, st)
    return mean(log_probs), st, Dict()
end

observations = randn(Float32, 3, 64)
actions = randn(Float32, 1, 64)
##
grads, loss, stats, train_state = Lux.Training.compute_gradients(AutoZygote(), myloss7, (observations, actions), train_state)
grads.actor_head
norm(grads.actor_head)

## nonzero  
stats_window_size = 50
alg = PPO(; ent_coef=0f0, vf_coef=0.480177f0, gamma=0.990886f0, gae_lambda=0.85821f0, clip_range=0.132141f0)
pendenv = BroadcastedParallelEnv([PendulumEnv() |> ScalingWrapperEnv for _ in 1:8])
pendenv = MonitorWrapperEnv(pendenv, stats_window_size)
pendenv = NormalizeWrapperEnv(pendenv, gamma=alg.gamma)
action_shp = size(action_space(pendenv))
obs_shp = size(observation_space(pendenv))

pendpolicy = ActorCriticPolicy(observation_space(pendenv), action_space(pendenv))
pendagent = ActorCriticAgent(pendpolicy; verbose=2, n_steps=128, batch_size=128, learning_rate=1.95409f-4, epochs=2)

train_state = pendagent.train_state
function myloss8(policy, ps, st, batch_data)
    observations = batch_data[1]
    actions = batch_data[2]
    feats, st = extract_features(policy, obs, ps, st)
    new_action_means, st = get_actions_from_features(policy, feats, ps, st)
    # @info "new_action_means: $(new_action_means)"
    # @info "actions: $(actions)"
    values, st = get_values_from_features(policy, feats, ps, st)

    loss = mean((new_action_means .- actions) .^ 2)
    return loss, st, Dict()
end

observations = randn(Float32, 3, 64)
actions = randn(Float32, 1, 64)
##
grads, loss, stats, train_state = Lux.Training.compute_gradients(AutoZygote(), myloss7, (observations, actions), train_state)
grads.actor_head
##zero grads

stats_window_size = 50
alg = PPO(; ent_coef=0f0, vf_coef=0.480177f0, gamma=0.990886f0, gae_lambda=0.85821f0, clip_range=0.132141f0)
pendenv = BroadcastedParallelEnv([PendulumEnv() |> ScalingWrapperEnv for _ in 1:8])
pendenv = MonitorWrapperEnv(pendenv, stats_window_size)
pendenv = NormalizeWrapperEnv(pendenv, gamma=alg.gamma)
action_shp = size(action_space(pendenv))
obs_shp = size(observation_space(pendenv))

pendpolicy = ActorCriticPolicy(observation_space(pendenv), action_space(pendenv))
pendagent = ActorCriticAgent(pendpolicy; verbose=2, n_steps=128, batch_size=128, learning_rate=1.95409f-4, epochs=2)
roll_buffer = RolloutBuffer(observation_space(pendenv), action_space(pendenv), alg.gae_lambda, alg.gamma, pendagent.n_steps, number_of_envs(pendenv))
DRiL.collect_rollouts!(roll_buffer, pendagent, pendenv)
data_loader = DataLoader((roll_buffer.observations, roll_buffer.actions,
                roll_buffer.advantages, roll_buffer.returns,
                roll_buffer.logprobs, roll_buffer.values),
            batchsize=pendagent.batch_size, shuffle=true, parallel=true, rng=pendagent.rng)

train_state = pendagent.train_state
function myloss9(policy, ps, st, batch_data)
    observations = batch_data[1]
    actions = batch_data[2]
    feats, st = DRiL.extract_features(policy, observations, ps, st)
    new_action_means, st = DRiL.get_actions_from_features(policy, feats, ps, st)
    # @info "new_action_means: $(new_action_means)"
    # @info "actions: $(actions)"
    values, st = DRiL.get_values_from_features(policy, feats, ps, st)

    loss = mean((new_action_means .- actions) .^ 2)
    return loss, st, Dict()
end

##
grads, loss, stats, train_state = Lux.Training.compute_gradients(AutoZygote(), myloss9, first(data_loader), train_state)
grads.actor_head
##

## nonzero grads

stats_window_size = 50
alg = PPO(; ent_coef=0f0, vf_coef=0.480177f0, gamma=0.990886f0, gae_lambda=0.85821f0, clip_range=0.132141f0)
pendenv = BroadcastedParallelEnv([PendulumEnv() |> ScalingWrapperEnv for _ in 1:8])
pendenv = MonitorWrapperEnv(pendenv, stats_window_size)
pendenv = NormalizeWrapperEnv(pendenv, gamma=alg.gamma)
action_shp = size(action_space(pendenv))
obs_shp = size(observation_space(pendenv))

pendpolicy = ActorCriticPolicy(observation_space(pendenv), action_space(pendenv))
pendagent = ActorCriticAgent(pendpolicy; verbose=2, n_steps=128, batch_size=128, learning_rate=1.95409f-4, epochs=2)

train_state = pendagent.train_state
function myloss10(policy, ps, st, batch_data)
    observations = batch_data[1]
    actions = batch_data[2]
    feats, st = DRiL.extract_features(policy, observations, ps, st)
    new_action_means, st = DRiL.get_actions_from_features(policy, feats, ps, st)
    # @info "new_action_means: $(new_action_means)"
    # @info "actions: $(actions)"
    values, st = DRiL.get_values_from_features(policy, feats, ps, st)

    loss = mean((new_action_means .- actions) .^ 2)
    return loss, st, Dict()
end
observations = randn(Float32, 3, 64)
actions = randn(Float32, 1, 64)
##
grads, loss, stats, train_state = Lux.Training.compute_gradients(AutoZygote(), myloss10, (observations, actions), train_state)
grads.actor_head

## nonzero grads

stats_window_size = 50
alg = PPO(; ent_coef=0f0, vf_coef=0.480177f0, gamma=0.990886f0, gae_lambda=0.85821f0, clip_range=0.132141f0)
pendenv = BroadcastedParallelEnv([PendulumEnv() |> ScalingWrapperEnv for _ in 1:8])
pendenv = MonitorWrapperEnv(pendenv, stats_window_size)
pendenv = NormalizeWrapperEnv(pendenv, gamma=alg.gamma)
action_shp = size(action_space(pendenv))
obs_shp = size(observation_space(pendenv))

pendpolicy = ActorCriticPolicy(observation_space(pendenv), action_space(pendenv))
pendagent = ActorCriticAgent(pendpolicy; verbose=2, n_steps=128, batch_size=128, learning_rate=1.95409f-4, epochs=2)

train_state = pendagent.train_state
function myloss11(policy, ps, st, batch_data)
    observations = batch_data[1]
    actions = batch_data[2]
    @show size(observations), size(actions)
    feats, st = DRiL.extract_features(policy, observations, ps, st)
    new_action_means, st = DRiL.get_actions_from_features(policy, feats, ps, st)
    values, st = DRiL.get_values_from_features(policy, feats, ps, st)

    loss = mean((new_action_means .- actions) .^ 2)
    return loss, st, Dict()
end
observations = randn(Float32, 3, 256)
actions = randn(Float32, 1, 256)
data_loader = DataLoader((observations, actions), batchsize=64, shuffle=true, parallel=true, rng=pendagent.rng)
##
grads, loss, stats, train_state = Lux.Training.compute_gradients(AutoZygote(), myloss11, first(data_loader), train_state)
grads.actor_head

## 

stats_window_size = 50
alg = PPO(; ent_coef=0f0, vf_coef=0.480177f0, gamma=0.990886f0, gae_lambda=0.85821f0, clip_range=0.132141f0)
pendenv = BroadcastedParallelEnv([PendulumEnv() |> ScalingWrapperEnv for _ in 1:8])
pendenv = MonitorWrapperEnv(pendenv, stats_window_size)
pendenv = NormalizeWrapperEnv(pendenv, gamma=alg.gamma)
action_shp = size(action_space(pendenv))
obs_shp = size(observation_space(pendenv))

pendpolicy = ActorCriticPolicy(observation_space(pendenv), action_space(pendenv))
pendagent = ActorCriticAgent(pendpolicy; verbose=2, n_steps=128, batch_size=128, learning_rate=1.95409f-4, epochs=2)

train_state = pendagent.train_state
function myloss12(policy, ps, st, batch_data)
    observations = batch_data[1]
    actions = batch_data[2]
    @show size(observations), size(actions)
    feats, st = DRiL.extract_features(policy, observations, ps, st)
    new_action_means, st = DRiL.get_actions_from_features(policy, feats, ps, st)
    values, st = DRiL.get_values_from_features(policy, feats, ps, st)

    loss = mean((new_action_means .- actions) .^ 2)
    return loss, st, Dict()
end
roll_buffer = RolloutBuffer(observation_space(pendenv), action_space(pendenv), alg.gae_lambda, alg.gamma, pendagent.n_steps, number_of_envs(pendenv))
DRiL.collect_rollouts!(roll_buffer, pendagent, pendenv)
data_loader = DataLoader((roll_buffer.observations, roll_buffer.actions), batchsize=64, shuffle=true, parallel=true, rng=pendagent.rng)
##
grads, loss, stats, train_state = Lux.Training.compute_gradients(AutoZygote(), myloss12, first(data_loader), train_state)
grads.actor_head
##
grads, loss, stats, train_state = Lux.Training.compute_gradients(AutoZygote(), myloss12, (roll_buffer.observations, roll_buffer.actions.*(1.f0 + eps(Float32))), train_state)
grads.actor_head
##
roll_buffer.observations
roll_buffer.actions
##
roll_buffer.observations[1]
roll_buffer.actions[1]
## solution? not random sampling actions in rollouts, resulting in difference in actions being zero, so analytically at the bottom of a parabola, so actually zero gradient was correct