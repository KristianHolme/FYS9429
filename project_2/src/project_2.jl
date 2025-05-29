module project_2

using Reexport
@reexport using DRiL
using Makie
@reexport using Random
using ClassicControlEnvironments

include("experiments.jl")
export default_PPO, default_agent, default_env

include("custom_envs.jl")
export GoalReachingEnv

logdir(args...) = joinpath(@__DIR__, "..", "logs", args...)
export logdir
end