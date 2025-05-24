module project_2

using Reexport
@reexport using DRiL
using Makie
@reexport using Random
using Pendulum

include("experiments.jl")
export default_PPO, default_agent, default_env

include("custom_envs.jl")
export GoalReachingEnv
end