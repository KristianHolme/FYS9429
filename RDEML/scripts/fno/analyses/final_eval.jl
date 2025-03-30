using DrWatson
@quickactivate :RDEML
using CairoMakie
using Statistics
## Test using old
experiment_name = "final_runs_2"
df = collect_results(datadir("fno", experiment_name))
const cdev = cpu_device()
const gdev = CUDADevice(device!(0))
##
fig = Figure()
ax = Axis(fig[1,1], yscale=log10, title="Final Test Loss", xlabel="Run", ylabel="Loss")
barplot!(ax, df.run, df.final_test_loss)
hlines!(ax, mean(df.final_test_loss), color=:red, linestyle=:dash, alpha=0.5)
hlines!(ax, median(df.final_test_loss), color=:green, linestyle=:dash, alpha=0.5)
fig
##
extra_save_dir = joinpath(@__DIR__, "..", "..", "..", "..", 
 "reports", "Project1", "figures", "final_eval")
plot_losses_final_eval(df; save_plot=true, 
    folder=experiment_name,
    extra_save_dir)
## Get best config
fno_config = df[argmin(df.final_test_loss), :].full_config
fno_config.ps = fno_config.ps |> gdev
fno_config.st = fno_config.st |> gdev

## Test the FNO with a known initial condition
RDEParams = RDEParam(tmax = 410.0)
env = RDEEnv(RDEParams,
    dt = 1.0,
    observation_strategy=SectionedStateObservation(minisections=32), 
    reset_strategy=WeightedCombination(Float32[0.15, 0.5, 0.2, 0.15]))
policy = StepwiseRDEPolicy(env, [20.0f0, 100.0f0, 200.0f0, 300.0f0, 400.0f0], 
[0.64f0, 0.96f0, 0.45f0, 0.7f0, 0.5f0])
sim_test_data = run_policy(policy, env)


##plot the initial condition
plot_initial_conditions(env,
     save_dir=extra_save_dir,
     title="Initial condition",
     subtitle="Weighted combination of shockwaves",
     savename="initial_condition_known")
##

hist_fig = plot_shifted_history(sim_test_data,
 env.prob.x, title="FDM simulation",
 use_rewards=false, plot_shocks=false,
 size=(1000, 500)
)
rowsize!(hist_fig.layout, 2, Relative(1/4))
display(hist_fig)
save(joinpath(extra_save_dir, "simulation_known.png"), hist_fig)
##
rec_sim_test_data = replace_sim_with_prediction(;sim_test_data, fno_config, gdev, cdev)
rec_hist_fig = plot_shifted_history(rec_sim_test_data,
 env.prob.x, title="FNO recursive predictions",
 use_rewards=false, plot_shocks=false,
 size=(1000, 500)
)
rowsize!(rec_hist_fig.layout, 2, Relative(1/4))
display(rec_hist_fig)
save(joinpath(extra_save_dir, "recursive_prediction_known.png"), rec_hist_fig)



## plot selected predictions
fig = compare_to_policy(;fnoconfig=fno_config, 
    sim_test_data, env, cdev, gdev, timesteps=[1, 18, 105, 340])
display(fig)
save(joinpath(extra_save_dir, "one_step_known.svg"), fig)
##
fig = compare_to_policy(;fnoconfig=fno_config, 
    sim_test_data, env, cdev, gdev, recursive=true, 
    timesteps=[14, 20, 90, 150, 270, 350], plot_input=false)
display(fig)
save(joinpath(extra_save_dir, "recursive_predictions_known.svg"), fig)





## Test with new policy
env.prob.params.tmax = 410f0
policy = SawtoothPolicy(env, 100f0, 0.95f0, 0.4f0)
reset!(env)
##
sim_test_data = run_policy(policy, env)
##
hist_fig = plot_shifted_history(sim_test_data,
 env.prob.x, title="FDM simulation",
 use_rewards=false, plot_shocks=false,
 size=(1000, 500)
)
rowsize!(hist_fig.layout, 2, Relative(1/4))
display(hist_fig)
save(joinpath(extra_save_dir, "simulation_saw.png"), hist_fig)
##
rec_sim_test_data = replace_sim_with_prediction(;sim_test_data, fno_config, gdev, cdev)
rec_hist_fig = plot_shifted_history(rec_sim_test_data,
 env.prob.x, title="FNO recursive predictions",
 use_rewards=false, plot_shocks=false,
 size=(1000, 500)
)
rowsize!(rec_hist_fig.layout, 2, Relative(1/4))
display(rec_hist_fig)
save(joinpath(extra_save_dir, "fno_solution_saw.png"), rec_hist_fig)



## plot selected predictions
fig = compare_to_policy(;fnoconfig=fno_config, 
    sim_test_data, env, cdev, gdev, timesteps=[1, 102, 190])
display(fig)
save(joinpath(extra_save_dir, "one_step_saw.svg"), fig)
##
fig = compare_to_policy(;fnoconfig=fno_config, 
    sim_test_data, env, cdev, gdev, recursive=true, 
    timesteps=[1,102,190], plot_input=false)
display(fig)
save(joinpath(extra_save_dir, "recursive_predictions_saw.svg"), fig)

## Test with unseen initial condition, but seen policy
using SpecialFunctions
function custom_init(x;c = [1f0,3f0,5f0], m=[10f0,4f0,10f0])
    return 1f0 .* sech.((x .- c[1]).*m[1]) .- 
           0.2f0 .* sech.((x .- c[2]).*m[2]) .+ 
           0.8f0 .* sech.((x .- c[3]).*m[3]) .+ 0.5f0
end
env.prob.reset_strategy = CustomPressureReset(custom_init)
policy = ScaledPolicy(SinusoidalRDEPolicy(env, w_1=0f0, w_2=0.05f0), 0.01f0)
env.prob.params.tmax = 410f0
reset!(env)
##plot the initial condition
plot_initial_conditions(env,
save_dir=extra_save_dir,
title="Initial condition",
subtitle="",
savename="initial_condition_sech")
##
sim_test_data = run_policy(policy, env)
##
hist_fig = plot_shifted_history(sim_test_data,
 env.prob.x, title="FDM simulation",
 use_rewards=false, plot_shocks=false,
 size=(1000, 500)
)
rowsize!(hist_fig.layout, 2, Relative(1/4))
display(hist_fig)
save(joinpath(extra_save_dir, "test_simulation_sech.png"), hist_fig)

## plot selected predictions
fig = compare_to_policy(;fnoconfig=fno_config, 
    sim_test_data, env, cdev, gdev, timesteps=[2,5])
display(fig)
save(joinpath(extra_save_dir, "one_step_sech.svg"), fig)
##
fig = compare_to_policy(;fnoconfig=fno_config, 
    sim_test_data, env, cdev, gdev, recursive=true, 
    timesteps=[1,2,5], plot_input=false)
display(fig)
save(joinpath(extra_save_dir, "recursive_predictions_sech.svg"), fig)