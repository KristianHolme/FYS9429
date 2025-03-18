"""
    PDESystemConfig

Configuration for Rotating Detonation Engine (RDE) PDE system.
"""
struct PDESystemConfig
    rde_params::RDEParam
    u_scale::Float64
end

"""
    default_pde_config(; tmax=1.0, u_scale=1.5, periodic_boundary=true)

Create a default PDE system configuration for Rotating Detonation Engine modeling.
"""
function default_pde_config(; tmax=1.0, u_scale=1.5)
    return PDESystemConfig(
        RDEParam{Float64}(tmax=tmax),
        u_scale)
end

"""
    Base.show(io::IO, config::PDESystemConfig)

Custom display for PDESystemConfig objects.
"""
function Base.show(io::IO, config::PDESystemConfig)
    println(io, "PDESystemConfig:")
    println(io, "├─ u_scale: $(config.u_scale)")
    println(io, "└─ tmax: $(config.rde_params.tmax)")
end

"""
    Base.show(io::IO, ::MIME"text/plain", config::PDESystemConfig)

Detailed display for PDESystemConfig objects.
"""
function Base.show(io::IO, ::MIME"text/plain", config::PDESystemConfig)
    println(io, "PDESystemConfig:")
    println(io, "├─ u_scale: $(config.u_scale)")
    println(io, "├─ tmax: $(config.rde_params.tmax)")
    println(io, "└─ RDE Parameters:")
    println(io, "   ├─ q_0: $(config.rde_params.q_0)")
    println(io, "   ├─ ν_1: $(config.rde_params.ν_1)")
    println(io, "   ├─ ν_2: $(config.rde_params.ν_2)")
    println(io, "   ├─ L: $(config.rde_params.L)")
    println(io, "   ├─ u_c: $(config.rde_params.u_c)")
    println(io, "   ├─ α: $(config.rde_params.α)")
    println(io, "   ├─ u_0: $(config.rde_params.u_0)")
    println(io, "   ├─ n: $(config.rde_params.n)")
    println(io, "   ├─ s: $(config.rde_params.s)")
    println(io, "   ├─ u_p: $(config.rde_params.u_p)")
    println(io, "   ├─ k_param: $(config.rde_params.k_param)")
    println(io, "   └─ ϵ: $(config.rde_params.ϵ)")
end

"""
    create_pde_system(config::PDESystemConfig)

Create a PDE system for Rotating Detonation Engine based on the configuration.
"""
function create_pde_system(config::PDESystemConfig)
    # Parameters and variables
    @parameters t, x
    @variables u(..), λ(..)
    Dt = Differential(t)
    Dx = Differential(x)
    Dxx = Differential(x)^2
    
    # Extract parameters from RDEParam
    params = config.rde_params
    q_0 = params.q_0
    ν_1 = params.ν_1
    ν_2 = params.ν_2
    L = params.L
    u_c = params.u_c
    α = params.α
    u_0 = params.u_0
    n = params.n
    s = params.s
    u_p = params.u_p
    k = params.k_param
    ϵ = params.ϵ
    tmax = params.tmax
    
    # Define helper functions
    ω(u) = RDE.ω(u, u_c, α)
    ξ(u) = RDE.ξ(u, u_0, n)
    β(u) = RDE.β(u, s, u_p, k)
    
    # Define equations
    eqs = [
        Dt(u(t, x)) + u(t, x) * Dx(u(t, x)) ~ (1 - λ(t, x)) * ω(u(t, x)) * q_0 + ν_1 * Dxx(u(t, x)) + ϵ * ξ(u(t, x)),
        Dt(λ(t, x)) ~ (1 - λ(t, x)) * ω(u(t, x)) - β(u(t, x)) * λ(t, x) + ν_2 * Dxx(λ(t, x))
    ]
    
    # Define boundary conditions
    bcs = [
        u(0, x) ~ RDE.default_u(x) * config.u_scale,
        λ(0, x) ~ RDE.default_λ(x),
        u(t, 0) ~ u(t, L),
        λ(t, 0) ~ λ(t, L)
    ]
    

    # Define domains
    domains = [t ∈ Interval(0.0, tmax), x ∈ Interval(0.0, L)]
    
    # Create and return PDESystem
    return @named pdesystem = PDESystem(eqs, bcs, domains, [t, x], [u(t, x), λ(t, x)])
end

"""
    run_simulation(rde_params; dt=0.01)

Run a Rotating Detonation Engine simulation for comparison with PINN results.
"""
function run_simulation(rde_params; dt=0.01)
    env = RDEEnv(rde_params, dt=dt)
    sim_data = run_policy(ConstantRDEPolicy(env), env)
    ts_sim = sim_data.state_ts
    xs_sim = env.prob.x
    us_sim, λs_sim = RDE.split_sol(sim_data.states)
    
    return ts_sim, xs_sim, us_sim, λs_sim
end 