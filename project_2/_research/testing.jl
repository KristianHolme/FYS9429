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
rewards, terminateds, truncateds, infos = DRiL.step!(penv, rand(act_space))

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

function predict_actions(policy::MyACPolicy, x, ps, st)
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

actions, st = DRiL.predict_actions(pend_policy, obs, ps, st)

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
fps = DRiL.collect_rollout!(roll_buffer, agent, env)
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
using WGLMakie
# using CairoMakie
using Statistics
using LinearAlgebra
using ClassicControlEnvironments
##
env = MultiThreadedParallelEnv([PendulumEnv() |> ScalingWrapperEnv for _ in 1:16])
policy = ActorCriticPolicy(observation_space(env), action_space(env))
agent = ActorCriticAgent(policy; verbose=2, n_steps=256, learning_rate=1f-3, epochs=10,
    log_dir="logs/working_tests/run", batch_size=64)
alg = PPO(; ent_coef=0.01f0, vf_coef=0.5f0, gamma=0.9f0, gae_lambda=0.95f0)
DRiL.TensorBoardLogger.write_hparams!(agent.logger, DRiL.get_hparams(alg), ["env/avg_step_rew", "train/loss"])
learn_stats = learn!(agent, env, alg; max_steps=100_000)
##
using JLD2
jldsave("data/saved_agent.jld2"; agent)
agent = load("data/saved_agent.jld2")["agent"]
## viz
single_env = PendulumEnv(; max_steps=500) |> ScalingWrapperEnv
observations, actions, rewards = collect_trajectory(agent, single_env)
actions = first.(actions) .* 2
plot_trajectory_interactive(PendulumEnv(), observations, actions, rewards)
##
Pendulum.interactive_viz(PendulumEnv())
##
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
    rewards = [sum(Pendulum.pendulum_rewards(theta, vel * 8.0f0, action * 2f0)) for (theta, action) in zip(thetas, vec(mock_actions_det))]

    lines!(ax_actions, thetas, vec(mock_actions_det), color=vel, colorrange=(-1f0, 1f0),
        label="v = $(round(vel, digits=1))")
    lines!(ax_values, thetas, vec(mock_values), color=vel, colorrange=(-1f0, 1f0))
    lines!(ax_rewards, thetas, rewards, color=vel, colorrange=(-1f0, 1f0))
end

# Add a colorbar to show velocity mapping
colorbar = Colorbar(fig[1:3, 2], colormap=cgrad(:viridis), limits=extrema(velocities),
    label="Velocity")

fig
save("plots/action_values_rewards.png", fig)

## 3D Surface plots and heatmaps
thetas_3d = LinRange(-π, π, 50) .|> Float32  # Reduced resolution for 3D
velocities_3d = LinRange(-1f0, 1f0, 25) .|> Float32

# Create meshgrids
theta_grid = repeat(thetas_3d', length(velocities_3d), 1)
velocity_grid = repeat(velocities_3d, 1, length(thetas_3d))

# Flatten for batch processing
theta_flat = vec(theta_grid)
velocity_flat = vec(velocity_grid)

# Create observations for all combinations
xs_flat = cos.(theta_flat)'
ys_flat = sin.(theta_flat)'
vels_flat = velocity_flat'
mock_obs_3d = vcat(xs_flat, ys_flat, vels_flat)

# Get predictions for all combinations
actions_3d = predict_actions(agent, mock_obs_3d; deterministic=true)
values_3d = predict_values(agent, mock_obs_3d)

# Calculate rewards
rewards_3d = [sum(Pendulum.pendulum_rewards(theta, vel * 8.0f0, action * 2f0))
              for (theta, vel, action) in zip(theta_flat, velocity_flat, vec(actions_3d))]

# Reshape back to grid format
actions_grid = reshape(vec(actions_3d), length(velocities_3d), length(thetas_3d))
values_grid = reshape(vec(values_3d), length(velocities_3d), length(thetas_3d))
rewards_grid = reshape(rewards_3d, length(velocities_3d), length(thetas_3d))

# Create 3D surface plots
fig_3d = Figure(size=(1200, 400))

ax_actions_3d = Axis3(fig_3d[1, 1], xlabel="θ", ylabel="Velocity", zlabel="Actions",
    title="Actions Surface")
ax_values_3d = Axis3(fig_3d[1, 2], xlabel="θ", ylabel="Velocity", zlabel="Values",
    title="Values Surface")
ax_rewards_3d = Axis3(fig_3d[1, 3], xlabel="θ", ylabel="Velocity", zlabel="Rewards",
    title="Rewards Surface")

surface!(ax_actions_3d, thetas_3d, velocities_3d, actions_3d |> vec, colormap=:viridis)
surface!(ax_values_3d, thetas_3d, velocities_3d, values_3d |> vec, colormap=:plasma)
surface!(ax_rewards_3d, thetas_3d, velocities_3d, rewards_3d |> vec, colormap=:inferno)

display(fig_3d)
save("plots/action_values_rewards_3d_surface.png", fig_3d)
##
# Create heatmaps
fig_heatmap = Figure(size=(1200, 400))

ax_actions_heat = Axis(fig_heatmap[1, 1], xlabel="θ", ylabel="Velocity", title="Actions Heatmap")
ax_values_heat = Axis(fig_heatmap[1, 2], xlabel="θ", ylabel="Velocity", title="Values Heatmap")
ax_rewards_heat = Axis(fig_heatmap[1, 3], xlabel="θ", ylabel="Velocity", title="Rewards Heatmap")

hm1 = heatmap!(ax_actions_heat, thetas_3d, velocities_3d, actions_grid, colormap=:viridis)
hm2 = heatmap!(ax_values_heat, thetas_3d, velocities_3d, values_grid, colormap=:plasma)
hm3 = heatmap!(ax_rewards_heat, thetas_3d, velocities_3d, rewards_grid, colormap=:inferno)

Colorbar(fig_heatmap[1, 4], hm1, label="Actions")
Colorbar(fig_heatmap[1, 5], hm2, label="Values")
Colorbar(fig_heatmap[1, 6], hm3, label="Rewards")

fig_heatmap
save("plots/action_values_rewards_heatmap.png", fig_heatmap)

# Create contour plots as an alternative view
fig_contour = Figure(size=(1200, 400))

ax_actions_cont = Axis(fig_contour[1, 1], xlabel="θ", ylabel="Velocity", title="Actions Contour")
ax_values_cont = Axis(fig_contour[1, 2], xlabel="θ", ylabel="Velocity", title="Values Contour")
ax_rewards_cont = Axis(fig_contour[1, 3], xlabel="θ", ylabel="Velocity", title="Rewards Contour")

contourf!(ax_actions_cont, thetas_3d, velocities_3d, actions_grid, colormap=:viridis, levels=15)
contourf!(ax_values_cont, thetas_3d, velocities_3d, values_grid, colormap=:plasma, levels=15)
contourf!(ax_rewards_cont, thetas_3d, velocities_3d, rewards_grid, colormap=:inferno, levels=15)

# Add contour lines for better readability
contour!(ax_actions_cont, thetas_3d, velocities_3d, actions_grid, color=:black, alpha=0.3, levels=8)
contour!(ax_values_cont, thetas_3d, velocities_3d, values_grid, color=:black, alpha=0.3, levels=8)
contour!(ax_rewards_cont, thetas_3d, velocities_3d, rewards_grid, color=:black, alpha=0.3, levels=8)

fig_contour
save("plots/action_values_rewards_contour.png", fig_contour)