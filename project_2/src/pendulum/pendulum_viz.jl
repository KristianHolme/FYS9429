function _pendulum_coords(L, θ)
    return Point2f(-L * sin(θ), L * cos(θ))
end

function _torque_arrow_coords(L, θ, τ)
    # Arrow is centered at the midpoint of the pendulum
    mid_x, mid_y = _pendulum_coords(L/2, θ)
    # Arrow direction: perpendicular to pendulum
    perp_angle = θ + π/2 * sign(τ)
    # Arrow length scales with torque, now up to 0.8*L
    arrow_length = 0.8 * L * clamp(abs(τ) / 2, 0, 1)
    dx = arrow_length * -sin(perp_angle)
    dy = arrow_length * cos(perp_angle)
    color = τ > 0 ? :green : :orange
    (; mid_x, mid_y, dx, dy, color)
end

function plot_pendulum(problem::PendulumProblem)
    L = problem.length
    θ = problem.theta
    τ = problem.torque
    fig = Figure(size = (400, 400))
    ax = Axis(fig[1, 1], aspect = 1)
    # Pendulum
    pt = _pendulum_coords(L, θ)
    lines!(ax, [Point2f(0.0, 0.0), pt], linewidth=4, color=:black)
    scatter!(ax, [0.0], [0.0], color=:red, markersize=15)
    scatter!(ax, pt, color=:blue, markersize=20)
    # Torque arrow
    if abs(τ) > 1e-4
        arr = _torque_arrow_coords(L, θ, τ)
        arrows!(ax, [Point2f(arr.mid_x, arr.mid_y)], [Vec2f(arr.dx, arr.dy)], color=arr.color, arrowsize=0.2)
    end
    xlims!(ax, -L-0.2, L+0.2)
    ylims!(ax, -L-0.2, L+0.2)
    fig
end

function live_pendulum_viz(problem::PendulumProblem)
    θ = Observable(problem.theta)
    τ = Observable(problem.torque)
    L = problem.length
    fig = Figure(size = (400, 400))
    ax = Axis(fig[1, 1], aspect = 1)
    pendulum_line = lines!(ax, @lift([Point2f(0.0, 0.0), _pendulum_coords(L, $θ)]), linewidth=4, color=:black)
    scatter!(ax, [0.0], [0.0], color=:red, markersize=15)
    mass_scatter = scatter!(ax, @lift(_pendulum_coords(L, $θ)), color=:blue, markersize=20)
    # Torque arrow
    torque_arrow = arrows!(ax,
        lift((θ, τ) -> [Point2f(_torque_arrow_coords(L, θ, τ).mid_x, _torque_arrow_coords(L, θ, τ).mid_y)], θ, τ),
        lift((θ, τ) -> [Vec2f(_torque_arrow_coords(L, θ, τ).dx, _torque_arrow_coords(L, θ, τ).dy)], θ, τ),
        color=lift((θ, τ) -> _torque_arrow_coords(L, θ, τ).color, θ, τ), arrowsize=0.2)
    xlims!(ax, -L-0.2, L+0.2)
    ylims!(ax, -L-0.2, L+0.2)
    display(fig)
    update_viz! = (problem) -> begin
        θ[] = problem.theta
        τ[] = problem.torque
    end
    return θ, τ, fig, update_viz!
end

function interactive_viz(env::PendulumEnv)
    θ = Observable(env.problem.theta)
    τ = Observable(env.problem.torque)
    dt = Observable(env.problem.dt)
    live = Observable(true)
    L = env.problem.length

    fig = Figure(size = (400, 500))
    ax = Axis(fig[1, 1], aspect = 1)
    pendulum_line = lines!(ax, @lift([Point2f(0.0, 0.0), _pendulum_coords(L, $θ)]), linewidth=4, color=:black)
    scatter!(ax, [0.0], [0.0], color=:red, markersize=15)
    mass_scatter = scatter!(ax, @lift(_pendulum_coords(L, $θ)), color=:blue, markersize=20)
    torque_arrow = arrows!(ax,
        lift((θ, τ) -> [Point2f(_torque_arrow_coords(L, θ, τ).mid_x, _torque_arrow_coords(L, θ, τ).mid_y)], θ, τ),
        lift((θ, τ) -> [Vec2f(_torque_arrow_coords(L, θ, τ).dx, _torque_arrow_coords(L, θ, τ).dy)], θ, τ),
        color=lift((θ, τ) -> _torque_arrow_coords(L, θ, τ).color, θ, τ), arrowsize=0.2)
    xlims!(ax, -L-0.2, L+0.2)
    ylims!(ax, -L-0.2, L+0.2)

    # SliderGrid for torque and dt
    sg = SliderGrid(fig[2, 1],
        (label = "Torque", range = -2.0:0.01:2.0, startvalue = env.problem.torque),
        (label = "dt", range = 0.0001:0.0001:0.01, startvalue = env.problem.dt),
        width = Relative(0.9)
    )
    live_button = Button(fig[3, 1], label = "Stop", tellwidth = false)
    on(live_button.clicks) do n
        live[] = !live[]
    end
    torque_slider = sg.sliders[1]
    dt_slider = sg.sliders[2]
    on(torque_slider.value) do val
        τ[] = val
    end
    on(dt_slider.value) do val
        dt[] = val
        env.problem.dt = val
    end

    display(fig)

    @async begin
        while live[]
            sleep(dt[])
            act!(env, τ[])
            θ[] = env.problem.theta
        end
    end

    return θ, τ, dt, fig, sg
end

