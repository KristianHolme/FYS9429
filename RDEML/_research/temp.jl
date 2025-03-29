function number_of_params(hidden_sizes::Vector{Int}; input_size::Int=2, output_size::Int=1)
    p = 0
    push!(hidden_sizes, output_size)
    prepend!(hidden_sizes, input_size)
    for i in 1:length(hidden_sizes)-1
        p += hidden_sizes[i] * hidden_sizes[i+1] + hidden_sizes[i+1]
    end
    return p
end

number_of_params([64, 64])
number_of_params([16, 16, 16, 16, 16].*2)
number_of_params([64, 64, 64, 64])
number_of_params(ones(Int64, 16)*16)
##
include("scripts/fno/experiments/mode_comparison.jl")
include("scripts/fno/experiments/batch_size.jl")
include("scripts/fno/experiments/width.jl")

include("scripts/fno/experiments/depth.jl")

include("scripts/fno/experiments/init_lr.jl")

include("scripts/fno/experiments/secondary_lr.jl")

##
using DrWatson
@quickactivate :RDEML
const gdev = CUDADevice(device!(dev_id-1))

## Test regular MLP to learn a simple function
using Lux, Random, LuxCUDA, MLUtils, OptimizationOptimisers
using Statistics
using ProgressMeter

const gdev = CUDADevice(device!(0))

# Create the target function
f(x) = sin(3x) + sin(5x)/3



# Generate training data
rng = Random.default_rng()
Random.seed!(rng, 123)
n_train = 100_000
x_train = range(-2π, 2π, length=n_train) |> collect
y_train = f.(x_train)
x_train = reshape(x_train, 1, :)  # Reshape for Lux input
y_train = reshape(y_train, 1, :)

# Create the model (using [128, 128, 128] which gives ~100k parameters)

width=256
model = Chain(
    Dense(1, width, tanh),
    Dense(width, width, tanh),
    Dense(width, width, tanh),
    Dense(width, 1)
)

# Initialize parameters and states
ps, st = Lux.setup(rng, model)
ps = ps |> gdev
st = st |> gdev


# Training parameters
n_epochs = 10
batch_size = 512
learning_rate = 3f-4
AD = AutoZygote()
losses = Float32[]

dataloader = DataLoader((x_train, y_train), batchsize=batch_size, shuffle=true)
tstate = Training.TrainState(model, ps, st, OptimizationOptimisers.Adam(learning_rate))
# Training loop
p = Progress(n_epochs*length(dataloader), showspeed=true)

for epoch in 1:n_epochs
    for (x, y) in gdev(dataloader)
        _, loss, _, tstate = Training.single_train_step!(AD, MSELoss(), (x, y), tstate)
        push!(losses, loss)
        next!(p, showvalues=[("Loss", loss)])
    end    
end
##
