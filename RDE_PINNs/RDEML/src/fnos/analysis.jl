function compare_to_policy(;fnoconfig::FNOConfig, policy, env, cdev, gdev, recursive=false, kwargs...)
    sim_test_data = run_policy(policy, env)
    plot_shifted_history(sim_test_data, env.prob.x, title="Test simulation", use_rewards=false)
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