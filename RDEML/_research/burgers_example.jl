x_data, y_data = data[1]

deeponet = DeepONet(;
    branch=(size(x_data, 1), ntuple(Returns(32), 5)...),
    trunk=(size(grid, 1), ntuple(Returns(32), 5)...),
    branch_activation=tanh,
    trunk_activation=tanh
)