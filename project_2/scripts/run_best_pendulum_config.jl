using DrWatson
@quickactivate :project_2
using DRiL
using Zygote
using ClassicControlEnvironments
using ProgressMeter
using Random
using JLD2
using Statistics
using DataFrames
using CSV
using Dates
using Base.Threads
using WGLMakie
##
include("hyperParamSearch.jl")
##
experiment_name = "ppo_search_2025-06-01_21-27"
df, best_config = analyze_results(experiment_name, "Pendulum")
##
params = Dict{String,Any}((String(k) => v for (k, v) in pairs(best_config)))
params["environment"] = "Pendulum"
params["log_std_init"] = 0.0f0
env = create_env("Pendulum", params)
policy = get_policy(params)
alg = get_alg(params)
agent = ActorCriticAgent(
    policy;
    learning_rate=Float32(params["learning_rate"]),
    n_steps=Int(params["n_steps"]),
    batch_size=Int(params["batch_size"]),
    epochs=Int(params["epochs"]),
    verbose=3,
    log_dir=logdir("pendulum", "best_config")
)
learn_stats = learn!(agent, env, alg, max_steps=100_000)
##
single_env = create_single_environment("Pendulum", params)
obs, actions, rewards = collect_trajectory(agent, single_env; norm_env=env)
plot_trajectory_interactive(PendulumEnv(), obs, actions, rewards)
animate_trajectory_video(PendulumEnv(), obs, actions, projectdir("videos", "pendulum_best_config.mp4"))
## plot snapshot of trajectory
using CairoMakie
CairoMakie.activate!()
actions = first.(actions)
prob = PendulumProblem()
for i in [1, 20, 30, 40, 50]
    x = obs[i][1]
    y = obs[i][2]
    prob.theta = atan(y, x)
    prob.velocity = obs[i][3]
    prob.torque = actions[i]
    fig = ClassicControlEnvironments.plot(prob)
    save(plotsdir("pendulum", "pendulum_best_config_$(i).svg"), fig)
    save(plotsdir("pendulum", "pendulum_best_config_$(i).png"), fig)
end
##