using DrWatson
@quickactivate :project_2
using DataFrames
using WGLMakie
using AlgebraOfGraphics
##
df = collect_results(datadir("experiments", "ppo_search_2025-05-24_17-12"))
columns = names(df)

fig_opts = (size = (600, 600),)
##
x_data = Symbol(columns[5])
data(df) * mapping(x_data, :eval_return, color = :trial_id,
    marker = :seed_idx => nonnumeric) * 
visual(Scatter) |> draw(figure = fig_opts, axis=())

##
x_data = :learning_rate
data(df) * mapping(x_data, :eval_return,
    row = :batch_size => nonnumeric,
    col = :epochs => nonnumeric,
    color = :gamma,
    marker = :n_steps => nonnumeric,
    ) * visual(Scatter) |> draw(figure = fig_opts)