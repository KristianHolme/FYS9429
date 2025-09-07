using DrWatson
@quickactivate :project_2
using DRiL
# using WGLMakie
using ClassicControlEnvironments
using Zygote
using JET
##
ent_coef = FixedEntropyCoefficient(0.051f0)
# ent_coef = AutoEntropyCoefficient()
alg = SAC(;
    ent_coef, buffer_capacity = 50_000, batch_size = 512, train_freq = 32, gradient_steps = 32,
    gamma = 0.9999f0, tau = 0.01f0, start_steps = 0
)
env = BroadcastedParallelEnv([MountainCarContinuousEnv() for _ in 1:8])
env = MonitorWrapperEnv(env)
env = NormalizeWrapperEnv(env, gamma = alg.gamma)

policy = SACPolicy(
    observation_space(env), action_space(env);
    log_std_init = -0.22f0, hidden_dims = [64, 64]
)
agent = SACAgent(policy, alg; verbose = 2, log_dir = logdir("sac_performance", "mountainccar_cont"))
DRiL.TensorBoardLogger.write_hparams!(agent.logger, DRiL.get_hparams(alg), ["env/ep_rew_mean", "train/actor_loss", "train/critic_loss", "train/entropy_loss"])
##
agent, replay_buffer, training_stats = learn!(agent, env, alg, 1_000)
##
@profview agent, replay_buffer, training_stats = learn!(agent, env, alg, 1_000)
##

const means = randn(Float32, 4, 3, 7)
const logstds = randn(Float32, 4, 3, 7)
const actions = randn(Float32, 4, 3, 7)

function get_ds(means::Array{T}, logstds::Array{T}) where {T <: Real}
    means_vec = collect(eachslice(means, dims = ndims(means)))::Vector{Array{T, ndims(means) - 1}}
    logstds_vec = collect(eachslice(logstds, dims = ndims(logstds)))::Vector{Array{T, ndims(logstds) - 1}}
    ds = SquashedDiagGaussian.(means_vec, logstds_vec)
    return ds
end

function get_logpdfs(means::Array{T}, logstds::Array{T}, actions::Array{T}) where {T <: Real}
    ds = get_ds(means, logstds)
    actions_vec = collect(eachslice(actions, dims = ndims(actions)))::Vector{Array{T, ndims(actions) - 1}}
    return sum(logpdf.(ds, actions_vec))
end


@report_opt get_ds(means, logstds)
@report_opt get_logpdfs(means, logstds, actions)
@report_opt gradient(a -> get_logpdfs(means, logstds, a), actions)
