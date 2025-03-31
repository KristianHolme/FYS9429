using Test
using RDEML
using Random
using Statistics
using Lux
using MLUtils
using NeuralOperators

# Test FNOConfig
@testset "FNOConfig" begin
    # Test default parameters
    config = FNOConfig()
    @test config.chs == (3, 64, 64, 64, 2)
    @test config.modes == 16
    @test config.activation == gelu
    
    # Test custom parameters
    custom_config = FNOConfig(chs=(4, 32, 32, 32, 4), modes=8, activation=relu)
    @test custom_config.chs == (4, 32, 32, 32, 4)
    @test custom_config.modes == 8
    @test custom_config.activation == relu
    
    # Test RNG seeding
    rng1 = Random.default_rng()
    rng2 = Random.default_rng()
    config1 = FNOConfig(rng=rng1)
    config2 = FNOConfig(rng=rng2)
    @test config1.ps != config2.ps  # Different random initialization
end

# Test FNO Constructor
@testset "FNO Constructor" begin
    # Test creation from FNOConfig
    config = FNOConfig()
    fno = FNO(config)
    @test fno isa FourierNeuralOperator
    
    # Test creation with direct parameters
    fno_direct = FNO(chs=(3, 64, 64, 64, 2), modes=16, activation=gelu)
    @test fno_direct isa FourierNeuralOperator
    
    # Test parameter validation
    @test_throws ArgumentError FNO(chs=(2,), modes=16)  # Invalid channel configuration
end

# Test FNODataset
@testset "FNODataset" begin
    # Test constructor with valid data
    x_data = rand(Float32, 32, 3, 100)
    y_data = rand(Float32, 32, 2, 100)
    dataset = FNODataset(x_data, y_data)
    @test dataset.raw_x_data == x_data
    @test dataset.raw_y_data == y_data
    
    # Test type constraints
    @test_throws MethodError FNODataset(rand(Int, 32, 3, 100), y_data)  # Wrong type
    @test_throws MethodError FNODataset(x_data, rand(Int, 32, 2, 100))  # Wrong type
end

# Test prepare_dataset
# @testset "prepare_dataset" begin
#     # Test train/test split
#     train_data, test_data = prepare_dataset(create_loader=false, test_split=0.2)
#     @test train_data isa FNODataset
#     @test test_data isa FNODataset
#     @test number_of_samples(train_data) + number_of_samples(test_data) == number_of_samples(train_data)
    
#     # Test DataLoader creation
#     train_loader, test_loader = prepare_dataset(create_loader=true, test_split=0.2)
#     @test train_loader isa DataLoader
#     @test test_loader isa DataLoader
# end

# Test train! function
@testset "train!" begin
    # Setup test data
    x_data = rand(Float32, 32, 3, 100)
    y_data = rand(Float32, 32, 2, 100)
    dataset = FNODataset(x_data, y_data)
    train_loader = create_dataloader(dataset, batch_size=32)
    
    # Test basic training
    config = FNOConfig()
    train!(config, train_loader, 0.001, 1)
    @test !isempty(config.history.losses)
    @test !isempty(config.history.epochs)
    
    # Test with test data
    test_loader = create_dataloader(dataset, batch_size=32)
    train!(config, train_loader, 0.001, 1, test_data=test_loader)
    @test !isempty(config.history.test_losses)
end

# Test evaluate_test_loss
@testset "evaluate_test_loss" begin
    # Setup test data
    x_data = rand(Float32, 32, 3, 100)
    y_data = rand(Float32, 32, 2, 100)
    dataset = FNODataset(x_data, y_data)
    test_loader = create_dataloader(dataset, batch_size=32)
    
    # Test loss calculation
    config = FNOConfig()
    model = FNO(config)
    test_loss = evaluate_test_loss(model, config.ps, config.st, test_loader, cpu_device())
    @test test_loss isa Float32
    @test test_loss >= 0  # Loss should be non-negative
end

# Test TrainingHistory
@testset "TrainingHistory" begin
    history = TrainingHistory()
    @test isempty(history.losses)
    @test isempty(history.test_losses)
    @test isempty(history.epochs)
    @test isempty(history.learning_rates)
    @test isempty(history.training_time)
end

# Test create_dataloader
@testset "create_dataloader" begin
    x_data = rand(Float32, 32, 3, 100)
    y_data = rand(Float32, 32, 2, 100)
    dataset = FNODataset(x_data, y_data)
    
    # Test batch size handling
    loader = create_dataloader(dataset, batch_size=32)
    @test loader isa DataLoader
    
    # Test shuffling
    loader_shuffled = create_dataloader(dataset, batch_size=32, shuffle=true)
    @test loader_shuffled isa DataLoader
end

# Test number_of_batches
@testset "number_of_batches" begin
    x_data = rand(Float32, 32, 3, 100)
    y_data = rand(Float32, 32, 2, 100)
    dataset = FNODataset(x_data, y_data)
    
    # Test different batch sizes
    @test number_of_batches(dataset, 32) == 4  # 100/32 rounded up
    @test number_of_batches(dataset, 100) == 1  # Exact fit
    @test number_of_batches(dataset, 200) == 1  # Larger than dataset
end

# Test sim_data_to_data_set
@testset "sim_data_to_data_set" begin
    # Create mock simulation data
    n_steps = 100
    n_sections = 32
    states = [rand(Float32, 2*n_sections) for _ in 1:n_steps]
    observations = [rand(Float32, n_sections) for _ in 1:n_steps]
    u_ps = rand(Float32, n_steps)
    
    sim_data = PolicyRunData(
        Float32[], #action_ts
        Float32[], #ss
        u_ps, #u_ps
        Float32[], #rewards
        Float32[], #energy_bal
        Float32[], #chamber_p
        Float32[], #state_ts
        states, #states
        observations) #observations
    
    # Test conversion
    raw_data = sim_data_to_data_set(sim_data)
    @test size(raw_data) == (n_sections, 3, n_steps)
    @test eltype(raw_data) == Float32
end 