# Demonstrate custom savename functionality
function demonstrate_custom_savename()
    config = create_and_train_model()
    
    # Get the savename directly
    custom_name = DrWatson.savename(config)
    println("\nCustom savename generated: $custom_name")
    
    # Get savename with a custom identifier
    custom_name_with_id = DrWatson.savename(config, "test_run")
    println("Custom savename with ID: $custom_name_with_id")
    
    # Save using the standard save function which now uses the custom savename
    filename = save_fno_config(config, "custom_example")
    println("Saved with custom savename to: $filename")
    
    return filename
end

# Run the examples
println("\n=== Testing save_fno_config ===")
filename1 = save_model_example()
config1 = load_model_example(filename1)

println("\n=== Testing direct saving with wsave/safesave/tagsave ===")
filename2 = save_model_directly()
config2 = load_model_directly(filename2)

println("\n=== Testing full serialization ===")
filename3 = save_model_fully_serialized()
config3 = load_model_fully_serialized(filename3)

println("\n=== Testing custom savename functionality ===")
filename4 = demonstrate_custom_savename()

println("\n=== Testing produce_or_load_fno ===")
demonstrate_produce_or_load()

println("\n=== Plotting losses ===")
plot_losses_example(filename1) 