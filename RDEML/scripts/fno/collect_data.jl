using DrWatson
@quickactivate :RDEML
##
## Get Policies and evironments for collecting data
policies = get_data_policies()
reset_strategies = get_data_reset_strategies()
data_gatherer = DataGatherer(policies[1:3], reset_strategies[1:3], 1)
## Collect data
generate_data(data_gatherer; dataset_name="test")
## Visualize data
visualize_data(run_data, policies, envs, 
    reset_strategies, n_runs_per_reset_strategy; 
    save_plots=true)