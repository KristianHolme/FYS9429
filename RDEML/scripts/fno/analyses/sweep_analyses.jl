using DrWatson
@quickactivate :RDEML
using CairoMakie
using Statistics
##
extra_save_dir = joinpath(@__DIR__, "..", "..", "..", "..", 
 "reports", "Project1", "figures", "sweeps")
##
experiment_name = "batch_size_2"

df = collect_results(datadir("fno", experiment_name))
plot_parameter_analysis(df, "batch_size"; 
    title="Batch Size Analysis",
    xlabel="Batch Size",
    save_plots=true,
    experiment_name=experiment_name,
    extra_save_dir) |> display
plot_training_time_analysis(df, "batch_size"; 
    title="Batch Size Analysis",
    xlabel="Batch Size",
    save_plots=true,
    experiment_name=experiment_name,
    extra_save_dir) |> display

##
experiment_name = "mode_comparison_2"

df = collect_results!(datadir("fno", experiment_name))
# df.final_train_loss = df.final_loss
fig = plot_parameter_analysis(df, "modes"; 
    title="Mode Comparison Analysis",
    xlabel="Modes",
    save_plots=true,
    experiment_name=experiment_name,
    extra_save_dir) |> display
plot_training_time_analysis(df, "modes"; 
    title="Mode Comparison Analysis",
    xlabel="Modes",
    save_plots=true,
    experiment_name=experiment_name,
    extra_save_dir) |> display
##
experiment_name = "depth_comparison_2"

df = collect_results(datadir("fno", experiment_name))
fig = plot_parameter_analysis(df, "depth"; 
    title="Depth Analysis",
    xlabel="Depth",
    save_plots=true,
    experiment_name=experiment_name,
    extra_save_dir) |> display
plot_training_time_analysis(df, "depth"; 
    title="Depth Analysis", 
    xlabel="Depth",
    save_plots=true,
    experiment_name=experiment_name,
    extra_save_dir) |> display
##
experiment_name = "width_comparison_2"

df = collect_results(datadir("fno", experiment_name))
fig = plot_parameter_analysis(df, "width"; 
    title="Width Analysis",
    xlabel="Width",
    save_plots=true,
    experiment_name=experiment_name,
    extra_save_dir) |> display
plot_training_time_analysis(df, "width"; 
    title="Width Analysis",
    xlabel="Width",
    save_plots=true,
    experiment_name=experiment_name,
    extra_save_dir) |> display

##
experiment_name = "init_lr_2"

df = collect_results(datadir("fno", experiment_name))
fig = plot_parameter_analysis(df, "init_lr"; 
    title="Initial Learning Rate Analysis",
    xlabel="Learning Rate",
    save_plots=true,
    experiment_name=experiment_name,
    extra_save_dir) |> display
plot_training_time_analysis(df, "init_lr"; 
    title="Initial Learning Rate Analysis",
    xlabel="Initial Learning Rate",
    save_plots=true,
    experiment_name=experiment_name,
    extra_save_dir) |> display

##
experiment_name = "secondary_lr_2"

df = collect_results(datadir("fno", experiment_name))
df = df[df.secondary_lr .!= 0.1f0, :]
fig = plot_parameter_analysis(df, "secondary_lr"; 
    title="Secondary Learning Rate Analysis",
    xlabel="End Learning Rate",
    save_plots=true,
    experiment_name=experiment_name,
    extra_save_dir) |> display
plot_training_time_analysis(df, "secondary_lr"; 
    title="Secondary Learning Rate Analysis",
    xlabel="End Learning Rate",
    save_plots=true,
    experiment_name=experiment_name,
    extra_save_dir) |> display
