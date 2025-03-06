"""
    plot_comparison(results::Vector{TrainingResult}, labels::Vector{String}; 
                   size=(1200, 800), time_indices=[1, -1])

Compare multiple training results.
"""
function plot_comparison(predictions::Dict; size=(900, 1200), title="")
    
    ts = predictions[:ts]
    xs = predictions[:xs]
    us = predictions[:us]
    λs = predictions[:λs]
    us_sim = predictions[:us_sim]
    λs_sim = predictions[:λs_sim]
    
    u_min, u_max = minimum(minimum.(us)), maximum(maximum.(us))
    u_min  = min(u_min, minimum(minimum.(us_sim)))
    u_max = max(u_max, maximum(maximum.(us_sim)))
    λ_min, λ_max = minimum(minimum.(λs)), maximum(maximum.(λs))
    λ_min  = min(λ_min, minimum(minimum.(λs_sim)))
    λ_max = max(λ_max, maximum(maximum.(λs_sim)))

    fig = Figure(size = size)
    ax_u = Axis(fig[1, 1], title = "PINN u(t, x)", xlabel = "t", ylabel = "x")
    hm_u = heatmap!(ax_u, ts, xs, stack(us)', colorrange = (u_min, u_max))
    ax_λ = Axis(fig[1, 2], title = "PINN λ(t, x)", xlabel = "t", ylabel = "x")
    hm_λ = heatmap!(ax_λ, ts, xs, stack(λs)', colorrange = (0, 1.1))
    ax_u_sim = Axis(fig[2, 1], title = "Sim u(t, x)", xlabel = "t", ylabel = "x")
    heatmap!(ax_u_sim, ts, xs, stack(us_sim)', colorrange = (u_min, u_max))
    ax_λ_sim = Axis(fig[2, 2], title = "Sim λ(t, x)", xlabel = "t", ylabel = "x")
    heatmap!(ax_λ_sim, ts, xs, stack(λs_sim)', colorrange = (0, 1.1))
    Colorbar(fig[3, 1], hm_u, vertical = false)
    Colorbar(fig[3, 2], hm_λ, vertical = false)
    ax_u_t = Axis(fig[4, 1], title = "PINN u(t, x)", xlabel = "x", ylabel = "u")
    N = length(ts)
    l_start = lines!(ax_u_t, xs, us[1], label="t=0")
    l_mid = lines!(ax_u_t, xs, us[Int(round(N/2))], label="t=$(ts[Int(round(N/2))])")
    l_end = lines!(ax_u_t, xs, us[end], label="t=$(ts[end])")
    ax_λ_t = Axis(fig[4, 2], title = "PINN λ(t, x)", xlabel = "x", ylabel = "λ")
    lines!(ax_λ_t, xs, λs[1])
    lines!(ax_λ_t, xs, λs[Int(round(N/2))])
    lines!(ax_λ_t, xs, λs[end])
    ylims!(ax_λ_t, 0, 1.1)
    N_sim = length(ts)
    ax_u_sim_t = Axis(fig[5, 1], title = "Sim u(t, x)", xlabel = "x", ylabel = "u")
    linkaxes!(ax_u_sim_t, ax_u_t)
    lines!(ax_u_sim_t, xs, us_sim[1])
    lines!(ax_u_sim_t, xs, us_sim[Int(round(N_sim/2))])
    lines!(ax_u_sim_t, xs, us_sim[end])
    ax_λ_sim_t = Axis(fig[5, 2], title = "Sim λ(t, x)", xlabel = "x", ylabel = "λ")
    linkaxes!(ax_λ_sim_t, ax_λ_t)
    lines!(ax_λ_sim_t, xs, λs_sim[1])
    lines!(ax_λ_sim_t, xs, λs_sim[Int(round(N_sim/2))])
    lines!(ax_λ_sim_t, xs, λs_sim[end])
    ylims!(ax_λ_sim_t, 0, 1.1)
    Legend(fig[end+1, :], [l_start, l_mid, l_end], 
        ["t=0", "t=$(round(ts[Int(round(N/2))], digits=2))", "t=$(round(ts[end], digits=2))"],
        orientation=:horizontal, tellwidth=false, tellheight=true)
    Label(fig[0, :], title, fontsize=20)
    return fig
end

"""
    plot_solution(ts, xs, us, λs; title="Rotating Detonation Engine Solution")

Plot the solution of a Rotating Detonation Engine simulation.
"""
function plot_solution(ts, xs, us, λs; title="Rotating Detonation Engine Solution")
    fig = Figure(size=(600, 1200))
    
    # Plot u
    ax1 = Axis(fig[1, 1], title="u(t,x)", xlabel="t", ylabel="x")
    hm1 = heatmap!(ax1, ts, xs, stack(us)')
    Colorbar(fig[1, 2], hm1)
    
    # Plot λ
    ax2 = Axis(fig[1, 3], title="λ(t,x)", xlabel="t", ylabel="x")
    hm2 = heatmap!(ax2, ts, xs, stack(λs)')
    Colorbar(fig[1, 4], hm2)
    
    Label(fig[0, 1:4], title, fontsize=20)
    
    return fig
end

"""
    plot_error(ts, xs, us, λs, us_sim, λs_sim; title="RDE PINN Error Analysis")

Plot the error between PINN solution and simulation for Rotating Detonation Engine.
"""
function plot_error(ts, xs, us, λs, us_sim, λs_sim; title="RDE PINN Error Analysis")
    # Calculate errors
    u_error = abs.(stack(us) - stack(us_sim))
    λ_error = abs.(stack(λs) - stack(λs_sim))
    
    fig = Figure(size=(1000, 500))
    
    # Plot u error
    ax1 = Axis(fig[1, 1], title="u(t,x) Error", xlabel="t", ylabel="x")
    hm1 = heatmap!(ax1, ts, xs, u_error', colormap=:thermal)
    Colorbar(fig[2, 1], hm1, vertical=false)
    
    # Plot λ error
    ax2 = Axis(fig[1, 2], title="λ(t,x) Error", xlabel="t", ylabel="x")
    hm2 = heatmap!(ax2, ts, xs, λ_error', colormap=:thermal)
    Colorbar(fig[2, 2], hm2, vertical=false)
    
    Label(fig[0, 1:2], title, fontsize=20)
    
    return fig
end


"""
    plot_experiment_comparison(result_dicts; metric="final_loss", title="RDE Experiment Comparison")

Compare multiple Rotating Detonation Engine experiments based on a specific metric.
"""
function plot_experiment_comparison(result_dicts; metric="final_loss", title="RDE Experiment Comparison")
    fig = Figure(size=(800, 500))
    
    ax = Axis(fig[1, 1], title=title, xlabel="Experiment", ylabel=metric)
    
    # Extract experiment names and metric values
    names = [dict["name"] for dict in result_dicts]
    values = [getfield(dict["metrics"], Symbol(metric)) for dict in result_dicts]
    
    # Create bar plot
    barplot!(ax, 1:length(names), values)
    ax.xticks = (1:length(names), names)
    ax.xticklabelrotation = π/4
    
    return fig
end