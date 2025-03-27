using DrWatson
@quickactivate :RDEML
##
params=RDEParam(tmax=400)
env = RDEEnv(params, dt=1.0f0)
policy = StepwiseRDEPolicy(env, [20.0f0, 100.0f0, 200.0f0, 350.0f0], 
[0.64f0, 0.86f0, 0.64f0, 0.96f0])
sim_data = run_policy(policy, env)
fig = plot_shifted_history(sim_data, env.prob.x, use_rewards=false, plot_shocks=false)
##
path = joinpath(homedir(), "Code", "FYS9429-reports", "reports", "Project1", "figures")
save(joinpath(path, "simulation-control.svg"), fig)
save(joinpath(path, "simulation-control.png"), fig)
