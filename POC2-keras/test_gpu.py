import tensorflow as tf

# Print TensorFlow version
print("TensorFlow version:", tf.version.VERSION)

# Check if CUDA is available
print("\nCUDA Configuration:")
print("CUDA Available:", tf.test.is_built_with_cuda())
print("GPU Available:", tf.config.list_physical_devices('GPU'))

# Get detailed GPU information
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.list_logical_devices('GPU')
        print("\nPhysical GPUs:", len(gpus))
        print("Logical GPUs:", len(logical_gpus))
        
        # Print GPU device details
        print("\nGPU Device Details:")
        for gpu in gpus:
            print(f"Device: {gpu}")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)
else:
    print("\nNo GPU devices found. Please check your CUDA installation.") 