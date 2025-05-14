using DrWatson
@quickactivate :project_2
using WGLMakie
##
problem = PendulumProblem(dt=0.01f0)
pend_env = PendulumEnv()
θ, τ, dt, fig, sg = live_pendulum_viz(pend_env.problem)