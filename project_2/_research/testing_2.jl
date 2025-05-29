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
##
stats_window_size = 50
alg = PPO(; ent_coef=0.001f0, vf_coef=0.5f0, gamma=0.98f0, gae_lambda=0.95f0)
env = MultiThreadedParallelEnv([PendulumEnv() for _ in 1:16])
env = MonitorWrapperEnv(env, stats_window_size)
env = NormalizeWrapperEnv(env, gamma=alg.gamma)

all_rewards = Float32[]
all_obs = Vector{Float32}[]
for _ in 1:200
    observe(env)
    rewards, term, trunc, info = DRiL.step!(env, rand(action_space(env), 16))
    push!(all_rewards, rewards...)
    push!(all_obs, eachcol(observe(env))...)
end
##
fig = Figure()
ax_x = Axis(fig[1, 1], title="x")
ax_y = Axis(fig[1, 2], title="y")
ax_vel = Axis(fig[2, 1], title="vel")
ax_rew = Axis(fig[2, 2], title="rew")
hist!(ax_x, getindex.(all_obs, 1))
hist!(ax_y, getindex.(all_obs, 2))
hist!(ax_vel, getindex.(all_obs, 3))
hist!(ax_rew, all_rewards)
fig
##

##
policy = ActorCriticPolicy(observation_space(env), action_space(env))
agent = ActorCriticAgent(policy; verbose=2, n_steps=256, learning_rate=3f-4, epochs=20,
    log_dir=logdir("working_tests", "normalized_monitored_run"), batch_size=128)
DRiL.TensorBoardLogger.write_hparams!(agent.logger, alg, agent, ["env/avg_step_rew", "train/loss"])
learn_stats = learn!(agent, env, alg; max_steps=100_000)
##
evaluate_agent(agent, env; show_progress=true)

##
using JLD2
jldsave("data/saved_agent.jld2"; agent)
agent = load("data/saved_agent.jld2")["agent"]
## viz
single_env = PendulumEnv(; gravity=10f0) |> ScalingWrapperEnv
observations, actions, rewards = collect_trajectory(agent, single_env)
actions = first.(actions) .* 2
mean(rewards)
plot_trajectory_interactive(PendulumEnv(), observations, actions, rewards)