module project_2

    using Reexport
    @reexport using DRiL
    using Makie
    @reexport using Random

    include("pendulum/pendulum.jl")
    export PendulumEnv, PendulumProblem
    include("pendulum/pendulum_viz.jl")
    export plot_pendulum, live_pendulum_viz, interactive_viz, 
        reward, plot_trajectory, plot_trajectory_interactive
    include("pendulum/utils.jl")
    export angle
end