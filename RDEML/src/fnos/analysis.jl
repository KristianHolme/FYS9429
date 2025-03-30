function compare_to_policy(;fnoconfig::FNOConfig, sim_test_data, env, cdev, gdev, recursive=false, kwargs...)
    test_states = sim_test_data.states
    N = env.prob.params.N
    test_data = zeros(Float32, N, 3, length(test_states))
    for i in eachindex(test_states)
        obs = test_states[i]
        test_data[:, 1, i] = obs[1:N]
        test_data[:, 2, i] = obs[N+1:2N]
        test_data[:, 3, i] .= sim_test_data.u_ps[min(i+1, length(sim_test_data.u_ps))] # shift u_p for correct causality, last is repeated but unused
    end
    fno = FNO(fnoconfig)
    ps, st = fnoconfig.ps, fnoconfig.st
    if recursive
        title="Recursive prediction"
        data = recursive_prediction(;fno, test_data, ps, st,
         steps=length(test_states), gdev, cdev)
        predicted_states = data[:,1:2,:] #remove u_p
    else
        title="One-step-ahead prediction"
        predictions, st = Lux.apply(fno, test_data[:,:,1:end-1] |> gdev, ps, st) |> cdev
        predicted_states = zeros(Float32, N, 2, length(test_states))
        predicted_states[:,:,1] = test_data[:,1:2,1] #set initial state
        predicted_states[:,:,2:end] = predictions
    end
    plot_test_comparison(;n_t=length(test_states), test_data, predicted_states, title, x=env.prob.x, kwargs...)
end

function recursive_prediction(;fno, test_data, ps, st, steps, gdev, cdev)
    N = size(test_data, 1)
    data = zeros(Float32, N, 3, steps)
    data[:,3,:] = test_data[:,3,:] #use the injection pressure for all steps
    data[:,1:2,1] = test_data[:,1:2,1] #set the initial state
    for i in 2:steps
        prediction, st = Lux.apply(fno, data[:,:,i-1:i-1] |> gdev, ps, st) |> cdev
        data[:,1:2,i] = prediction
    end
    return data
end

function replace_sim_with_prediction(;sim_test_data, fno_config, gdev, cdev)
    pred_data = deepcopy(sim_test_data)
    fno = FNO(fno_config)
    ps = fno_config.ps |> gdev
    st = fno_config.st |> gdev
    n_sim = length(pred_data.states[1]) ÷ 2
    test_data = sim_data_to_data_set(pred_data)
    data = recursive_prediction(;fno, test_data, ps, st, steps=length(pred_data.states), gdev, cdev)
    @assert size(data, 3) == length(pred_data.states)
    @assert size(data, 1) == n_sim "n_sim: $n_sim, size(data, 1): $(size(data, 1))"
    for i in eachindex(pred_data.states)
        pred_data.states[i][1:n_sim] .= data[:,1,i]
        pred_data.states[i][n_sim+1:end] .= data[:,2,i]
    end
    return pred_data
end

function plot_initial_conditions(env; save_dir="", savename="", kwargs...)
    x = env.prob.x
    fig = Figure()
    ax = Axis(fig[1,1], xlabel="x", ylabel="u", ylabelrotation=0; kwargs...)
    lines!(ax, x, env.prob.u0)
    hidexdecorations!(ax, ticks=true, ticklabels=true, grid=false)
    ax2 = Axis(fig[2,1], xlabel="x", ylabel="λ", ylabelrotation=0)
    lines!(ax2, x, env.prob.λ0)
    ylims!(ax2, 0.0, 1.1)
    xlims!(ax2, 0.0, 2π)
    linkxaxes!(ax, ax2)

    if !isempty(save_dir) && !isempty(savename)
        isdir(save_dir) || mkdir(save_dir)
        save(joinpath(save_dir, "$savename.svg"), fig)
    end
    return fig
end