using NeuralPDE, Lux, ModelingToolkit, Optimization, OptimizationOptimisers
import ModelingToolkit: Interval, infimum, supremum

@parameters x y
@variables u(..)
Dxx = Differential(x)^2
Dyy = Differential(y)^2

# 2D PDE
eq = Dxx(u(x, y)) + Dyy(u(x, y)) ~ -sin(pi * x) * sin(pi * y)

# Boundary conditions
bcs = [u(0, y) ~ 0.0, u(1, y) ~ 0,
    u(x, 0) ~ 0.0, u(x, 1) ~ 0]
# Space and time domains
domains = [x ∈ Interval(0.0, 1.0),
    y ∈ Interval(0.0, 1.0)]
# Discretization
dx = 0.1

# Neural network
dim = 2 # number of dimensions
chain = Lux.Chain(Dense(dim, 16, Lux.σ), Dense(16, 16, Lux.σ), Dense(16, 1))

discretization = PhysicsInformedNN(chain, QuadratureTraining())
discretization = PhysicsInformedNN(chain, QuasiRandomTraining(128))


@named pde_system = PDESystem(eq, bcs, domains, [x, y], [u(x, y)])
prob = discretize(pde_system, discretization)

prog = ProgressUnknown("Training...", showspeed = true)
callback = function (p, l)
    next!(prog, showvalues = [(:loss, l)])
    return false
end
## Training
res = Optimization.solve(prob, ADAM(0.1); callback = callback, maxiters = 4000)
prob = remake(prob, u0 = res.minimizer)
res = Optimization.solve(prob, ADAM(0.01); callback = callback, maxiters = 2000)
phi = discretization.phi

## Exact solution 
exact_sol(x,y) = (sin(pi*x)*sin(pi*y))/(2*pi^2)

# Create grid for visualization
xs = 0:0.01:1
ys = 0:0.01:1

# Get predictions
u_predict = reshape([first(phi([x,y], res.minimizer)) for x in xs for y in ys], (length(xs), length(ys)))
u_exact = [exact_sol(x,y) for x in xs, y in ys]

# Create visualization
fig = Figure(size=(900,300))

ax1 = Axis(fig[1,1], title="PINN Solution")
hm1 = heatmap!(ax1, xs, ys, u_predict)
Colorbar(fig[1,2], hm1)

ax2 = Axis(fig[1,3], title="Exact Solution") 
hm2 = heatmap!(ax2, xs, ys, u_exact)
Colorbar(fig[1,4], hm2)

# Calculate and display error metrics
mse = mean((u_predict .- u_exact).^2)
mae = mean(abs.(u_predict .- u_exact))

@info "Error Metrics" MSE=mse MAE=mae

display(fig)
