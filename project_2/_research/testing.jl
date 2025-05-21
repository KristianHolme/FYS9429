using DrWatson
@quickactivate :project_2
using WGLMakie
##
problem = PendulumProblem(dt=0.01f0)
pend_env = PendulumEnv()
θ, τ, fig, update_viz! = live_pendulum_viz(pend_env.problem)
act!(pend_env, 0.0f0)
update_viz!(pend_env.problem)
for i in 1:200
    act!(pend_env, rand(Float32) * 4f0 - 2f0)
    update_viz!(pend_env.problem)
    sleep(pend_env.problem.dt)
end

##
problem = PendulumProblem(dt=0.01f0)
env = PendulumEnv()
θ, τ, dt, fig, sg = interactive_viz(env)


##
pend_env = PendulumEnv()
penv = MultiThreadedParallelEnv([PendulumEnv(), PendulumEnv()])
obs_space = observation_space(penv)
act_space = action_space(penv)
DRiL.reset!(penv, MersenneTwister(1234))
get_info(penv)
rewards, terminateds, truncateds, infos = DRiL.step!(penv, rand(act_space.type, act_space.shape..., 2))

## DataLoader
using MLUtils
n_steps = 12
n_envs = 2
rb = RolloutBuffer(obs_space, act_space, n_steps, n_envs, 0.97, 0.99)
DRiL.reset!(rb)

dl = DataLoader((rb.observations, rb.actions, rb.rewards, rb.advantages, rb.returns, rb.logprobs, rb.values), batchsize=3)

## Lux networks
using Lux

feat_extr = Dense(4, 64, relu)
actor = Chain(Dense(64, 64, relu), Dense(64, 1), name="actor")
critic = Chain(Dense(64, 64, relu), Dense(64, 1), name="critic")
model = Chain(;
    feat_extr=Dense(4, 64, relu),
    actorCritic=Parallel(nothing;
        actor,
        critic
    )
)
model.actorCritic.layer_1



ps, st = Lux.setup(MersenneTwister(1234), model)
x = rand(Float32, 4, 1)
model(x, ps, st)
feats = model.feat_extr(x, ps.feat_extr, st.feat_extr)
actor_out = model.actorCritic.
critic_out = model.actorCritic.critic(feats)

ps
st

##
using Lux, Zygote, Optimisers
abstract type AbstractPolicy <: Lux.AbstractLuxLayer end
struct MyACPolicy <: AbstractPolicy
    feature_extractor::Chain
    actor_head::Dense
    critic_head::Dense
    log_std_init::AbstractFloat
end

function Lux.initialparameters(rng::AbstractRNG, policy::MyACPolicy)
    params = (:feature_extractor => Lux.initialparameters(rng, policy.feature_extractor),
        :actor_head => Lux.initialparameters(rng, policy.actor_head),
        :log_std_init => policy.log_std_init * ones(typeof(policy.log_std_init), policy.actor_head.out_dim),
        :critic_head => Lux.initialparameters(rng, policy.critic_head))
    return params
end

function Lux.initialstates(rng::AbstractRNG, policy::MyACPolicy)
    states = (:feature_extractor => Lux.initialstates(rng, policy.feature_extractor),
        :actor_head => Lux.initialstates(rng, policy.actor_head),
        :critic_head => Lux.initialstates(rng, policy.critic_head))
    return states
end

Lux.parameterlength(policy::MyACPolicy) = Lux.parameterlength(policy.feature_extractor) + Lux.parameterlength(policy.actor_head) + Lux.parameterlength(policy.critic_head)

Lux.statelength(policy::MyACPolicy) = Lux.statelength(policy.feature_extractor) + Lux.statelength(policy.actor_head) + Lux.statelength(policy.critic_head)

function MyACPolicy(in_dim, out_dim; log_std_init=0.0f0)
    return MyACPolicy(Chain(Dense(in_dim, 64, relu), Dense(64, 64, relu)), Dense(64, out_dim), Dense(64, 1), log_std_init)
end

function (policy::MyACPolicy)(x, ps, st)
    feats = policy.feature_extractor(x, ps.feature_extractor, st.feature_extractor)
    action_mean = policy.actor_head(feats, ps.actor_head, st.actor_head)
    value = policy.critic_head(feats, ps.critic_head, st.critic_head)
    return action_mean, value
end

function predict(policy::MyACPolicy, x, ps, st)
    feats = policy.feature_extractor(x, ps.feature_extractor, st.feature_extractor)
    action_mean = policy.actor_head(feats, ps.actor_head, st.actor_head)
    return action_mean
end

function evaluate_actions(policy::MyACPolicy, x, actions, ps, st) end


policy = MyACPolicy(4, 1)
ps, st = Lux.setup(MersenneTwister(1234), policy)
ps
###
using Distributions
using BenchmarkTools
means = rand(Float32, 2, 3, 5)
stds = rand(Float32, 2, 3, 5)

function get_entropy(means, stds)
    distris = MvNormal.(vec.(eachslice(means, dims=3)), vec.(eachslice(stds, dims=3)))
    entropy.(distris)
end
get_entropy(means, stds)
@benchmark get_entropy(means, stds) setup = (means=rand(Float32, 2, 3, 5), stds=rand(Float32, 2, 3, 5))


distris = MvNormal.(vec.(eachslice(means, dims=3)), vec.(eachslice(stds, dims=3)))



loglikelihood.(distris, vec.(eachslice(rand(Float32, 2, 3, 5), dims=3)))
entropy.(distris)
##

##
using DrWatson
@quickactivate :project_2
using Lux
using DRiL
using Debugger
using Distributions
using Enzyme
##
pend_env = PendulumEnv()
pend_policy = ActorCriticPolicy(pend_env.observation_space, pend_env.action_space)
rng = MersenneTwister(1234)
ps, st = Lux.setup(rng, pend_policy)
ps
obs = observe(pend_env)
feats, st = DRiL.extract_features(pend_policy, obs, ps, st)
feats

obs = rand(Float32, 3, 4)

actions, st = DRiL.predict(pend_policy, obs, ps, st)

means = rand(1, 1)
std = rand(1, 1)
Normal.(means, std)
pend_policy.action_space.shape
vec_actions = eachslice(means, dims=2)
typeof(vec_actions)
vec_actions[1]
Normal.(means, Ref(0.1))

## Testing functionality
env = MultiThreadedParallelEnv([PendulumEnv(), PendulumEnv()])
policy = ActorCriticPolicy(observation_space(env), action_space(env))
agent = ActorCriticAgent(policy; verbose=2, n_steps=64)
##

n_steps = 64
roll_buffer = RolloutBuffer(observation_space(env), action_space(env), 0.97, 0.99, n_steps, env.n_envs)
fps = DRiL.collect_rollouts!(roll_buffer, agent, env)
DRiL.reset!(roll_buffer)
trajectories = DRiL.collect_trajectories(agent, env, n_steps)

## copmonentarray
using ComponentArrays
ca = ComponentArray(a=[1 2 3; 4 5 6], b=[7 8 9 10 11 12])
ca[1:12]
ca = ca ./ 12
norm(ca)
norm(1:12)


## does everything work?
using DrWatson
@quickactivate :project_2
using Lux
using DRiL
using Zygote
# using WGLMakie
using CairoMakie
using Statistics
using LinearAlgebra
using Pendulum
##
env = MultiThreadedParallelEnv([PendulumEnv(; gravity=10f0, max_steps=500) |> ScalingWrapperEnv for _ in 1:16])
policy = ActorCriticPolicy(observation_space(env), action_space(env))
agent = ActorCriticAgent(policy; verbose=2, n_steps=256, learning_rate=1f-3, epochs=10,
    log_dir="logs/hyper_search/run", batch_size=64)
alg = PPO(; ent_coef=0.01f0, vf_coef=0.5f0, gamma=0.9f0, gae_lambda=0.95f0)
metrics = ["env/avg_ep_rew", "train/loss"]
# hparams = merge(DRiL.get_hparams(alg), DRiL.get_hparams(agent))
# DRiL.TensorBoardLogger.write_hparams!(agent.logger, hparams, metrics)
# DRiL.TensorBoardLogger.write_hparams!(agent.logger, agent, metrics)
# DRiL.TensorBoardLogger.write_hparams!(agent.logger, alg, metrics)
DRiL.TensorBoardLogger.write_hparams!(agent.logger, alg, agent, metrics)

learn!(agent, env, alg; max_steps=300_000)
## viz
single_env = PendulumEnv(; gravity=10f0, max_steps=500) |> ScalingWrapperEnv
observations, actions, rewards = collect_trajectory(agent, single_env)
actions = first.(actions) .* 2
plot_trajectory(PendulumEnv(), observations, actions, rewards)
sum(rewards)
animate_trajectory_video(PendulumEnv(), observations, actions, "test.mp4")
##
agent.train_state.parameters.log_std
observations, actions, rewards = collect_trajectory(agent, single_env)
WGLMakie.activate!()
plot_trajectory_interactive(PendulumEnv(), observations, actions, rewards)
CairoMakie.activate!()
plot_trajectory(PendulumEnv(), observations, actions, rewards)

mock_obs = rand(Float32, 3, 100) .* 2f0 .- 1f0
mock_actions = predict_actions(agent, mock_obs)
hist(vec(mock_actions))
mock_actions_det = predict_actions(agent, mock_obs; deterministic=true)
hist(vec(mock_actions_det))
mock_actions_mean = Statistics.mean(mock_actions, dims=2)
mock_actions_std = Statistics.std(mock_actions, dims=2)
agent.train_state.parameters.log_std

mock_actions_det_mean = Statistics.mean(mock_actions_det, dims=2)
mock_actions_det_std = Statistics.std(mock_actions_det, dims=2)

mock_values = predict_values(agent, mock_obs)
hist(vec(mock_values))
##
# Survey of actions and values across different velocities
thetas = LinRange(-π, π, 100) .|> Float32
velocities = LinRange(-1f0, 1f0, 23) # 9 different velocity values

fig = Figure(size=(600, 800))
ax_actions = Axis(fig[1, 1], xlabel="θ", ylabel="Actions", title="Actions vs Angle")
ax_values = Axis(fig[2, 1], xlabel="θ", ylabel="Values", title="Values vs Angle")
ax_rewards = Axis(fig[3, 1], xlabel="θ", ylabel="Rewards", title="Rewards vs Angle")

# Remove custom function and use the one from pendulum.jl

for (i, vel) in enumerate(velocities)
    xs = cos.(thetas)'
    ys = sin.(thetas)'
    vels = fill(vel, 1, length(thetas))
    mock_obs = vcat(xs, ys, vels)

    mock_actions_det = predict_actions(agent, mock_obs; deterministic=true)
    mock_values = predict_values(agent, mock_obs)

    # Calculate rewards for each theta and velocity using pendulum_rewards from pendulum.jl
    rewards = [sum(project_2.pendulum_rewards(theta, vel * 8.0f0, action * 2f0)) for (theta, action) in zip(thetas, vec(mock_actions_det))]

    lines!(ax_actions, thetas, vec(mock_actions_det), color=vel, colorrange=(-1f0, 1f0),
        label="v = $(round(vel, digits=1))")
    lines!(ax_values, thetas, vec(mock_values), color=vel, colorrange=(-1f0, 1f0))
    lines!(ax_rewards, thetas, rewards, color=vel, colorrange=(-1f0, 1f0))
end

# Add a colorbar to show velocity mapping
colorbar = Colorbar(fig[1:3, 2], colormap=cgrad(:viridis), limits=extrema(velocities),
    label="Velocity")

fig







