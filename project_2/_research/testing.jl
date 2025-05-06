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
    act!(pend_env, rand(Float32)*4f0 - 2f0)
    update_viz!(pend_env.problem)
    sleep(pend_env.problem.dt)
end

##
problem = PendulumProblem(dt=0.01f0)
env = PendulumEnv()
θ, τ, dt, fig, sg = interactive_viz(env)

