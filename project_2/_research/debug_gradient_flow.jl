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