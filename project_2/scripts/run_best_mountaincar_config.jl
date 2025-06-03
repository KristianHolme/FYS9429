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
WGLMakie.activate!()
##
include("hyperParamSearch.jl")
##
experiment_name = "ppo_search_2025-06-02_22-11"
df, best_config = analyze_results(experiment_name, "MountainCar")
##
params = Dict{String,Any}((String(k) => v for (k, v) in pairs(best_config)))
params["environment"] = "MountainCar"
env = create_env("MountainCar", params)
policy = get_policy(params)
alg = get_alg(params)
agent = ActorCriticAgent(
    policy;
    learning_rate=Float32(params["learning_rate"]),
    n_steps=Int(params["n_steps"]),
    batch_size=Int(params["batch_size"]),
    epochs=Int(params["epochs"]),
    verbose=3,
    log_dir=logdir("mountaincar", "best_config")
)
learn_stats = learn!(agent, env, alg, max_steps=100_000)
##
mean_return = evaluate_trained_agent(agent, env)
##
single_env = create_single_environment("MountainCar", params)
obs, actions, rewards = collect_trajectory(agent, single_env; norm_env=env)
fig_traj = plot_trajectory(MountainCarEnv(), obs, actions, rewards)
plot_trajectory_interactive(MountainCarEnv(), obs, actions, rewards)
animate_trajectory_video(MountainCarEnv(), obs, actions, projectdir("videos", "mountaincar_best_config.mp4"))
## plot snapshot of trajectory
using CairoMakie
CairoMakie.activate!()
actions = first.(actions)
prob = MountainCarProblem()
for i in [1, 192, 309, 372]
    pos = obs[i][1]
    vel = obs[i][2]
    prob.position = pos
    prob.velocity = vel
    prob.force = actions[i][1]
    fig = ClassicControlEnvironments.plot(prob)
    save(plotsdir("mountaincar", "mountaincar_best_config_$(i).svg"), fig)
    save(plotsdir("mountaincar", "mountaincar_best_config_$(i).png"), fig)
end
##