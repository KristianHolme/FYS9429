function compare_to_policy(;fno, ps, st, policy, env, cdev, gdev)
    sim_test_data = run_policy(policy, env)
    plot_shifted_history(sim_test_data, env.prob.x, title="Test simulation", use_rewards=false)
    test_states = sim_test_data.states
    N = env.prob.params.N
    test_data = zeros(Float32, N, 3, length(test_states))
    for i in eachindex(test_states)
        obs = test_states[i]
        test_data[:, 1, i] = obs[1:N]
        test_data[:, 2, i] = obs[N+1:2N]
        test_data[:, 3, i] .= sim_test_data.u_ps[i]
    end

    output_data, st = Lux.apply(fno, test_data[:,:,1:end-1] |> gdev, ps, st) |> cdev
    plot_test_comparison(;n_t=length(test_states), test_data, output_data)
end