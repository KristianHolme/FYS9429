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