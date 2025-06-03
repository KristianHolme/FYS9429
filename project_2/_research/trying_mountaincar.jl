using DrWatson
@quickactivate :project_2
using DRiL
using ClassicControlEnvironments
##

# Create environment
env = MountainCarEnv()

# Reset and run
reset!(env)
obs = observe(env)
reward = act!(env, 0.5f0)  # Apply force

# Visualize (requires Makie)
using WGLMakie
ClassicControlEnvironments.plot(env.problem)  # Static plot
ClassicControlEnvironments.interactive_viz(env)  # Interactive visualization