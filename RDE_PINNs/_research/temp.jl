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
function supersin(x)
    modes = 2:9
    shifts = rand(1..2*pi, length(modes))
    M = stack([sin.(i .* x .+ shifts[ix])./(3*i) for (ix, i) in enumerate(modes)])
    return 1.0 .+ max.(0, sum(M, dims=2))
end
x = LinRange(0, 2π, 512) |> collect
y = supersin(x)
lines(x, vec(y))
##
fc = FNOConfig(modes=32)
savename(fc)
RDEML.fnoconfig_to_dict(fc)
safesave(datadir("test", savename(fc)*".jld2"), fc)
res = collect_results(datadir("test"))
fc = res[1, :full_config]
res[1, "chs"]