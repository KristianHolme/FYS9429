module project_2

    using Reexport
    @reexport using DRiL
    using Makie
    @reexport using Random

    export PendulumEnv, PendulumProblem
    export plot_pendulum, live_pendulum_viz, interactive_viz

    include("pendulum/pendulum.jl")
    include("pendulum/pendulum_viz.jl")


end