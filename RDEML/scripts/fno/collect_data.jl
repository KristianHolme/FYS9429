using DrWatson
@quickactivate :RDEML
##
## Get Policies and evironments for collecting data
policies = get_data_policies()
reset_strategies = get_data_reset_strategies()
data_gatherer = DataGatherer(policies, reset_strategies, 10)
## Collect data
generate_data(data_gatherer;)
## Visualize data
df = collect_results(datadir("datasets"))
visualize_data(df[df.run .== 1, :])
