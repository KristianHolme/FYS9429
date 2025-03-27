using DrWatson
@quickactivate :RDEML
using CairoMakie
using Statistics
##
experiment_name = "batch_size"

df = collect_results(datadir("fno", experiment_name))
plot_parameter_analysis(df, "batch_size", 
    title="Batch Size Analysis",
    xlabel="Batch Size",
    save_plots=true,
    experiment_name=experiment_name) |> display
plot_training_time_analysis(df, "batch_size", 
    title="Batch Size Analysis",
    xlabel="Batch Size",
    save_plots=true,
    experiment_name=experiment_name) |> display

##
experiment_name = "mode_comparison"

df = collect_results(datadir("fno", experiment_name))
df.final_train_loss = df.final_loss
fig = plot_parameter_analysis(df, "modes", 
    title="Mode Comparison Analysis",
    xlabel="Modes",
    save_plots=true,
    experiment_name=experiment_name) |> display
plot_training_time_analysis(df, "modes", 
    title="Mode Comparison Analysis",
    xlabel="Modes",
    save_plots=true,
    experiment_name=experiment_name) |> display
##
experiment_name = "depth_comparison"

df = collect_results(datadir("fno", experiment_name))
fig = plot_parameter_analysis(df, "depth", 
    title="Depth Analysis",
    xlabel="Depth",
    save_plots=true,
    experiment_name=experiment_name) |> display
plot_training_time_analysis(df, "depth", 
    title="Depth Analysis",
    xlabel="Depth",
    save_plots=true,
    experiment_name=experiment_name) |> display
##
experiment_name = "width_comparison"

df = collect_results(datadir("fno", experiment_name))
fig = plot_parameter_analysis(df, "width", 
    title="Width Analysis",
    xlabel="Width",
    save_plots=true,
    experiment_name=experiment_name) |> display
plot_training_time_analysis(df, "width", 
    title="Width Analysis",
    xlabel="Width",
    save_plots=true,
    experiment_name=experiment_name) |> display
