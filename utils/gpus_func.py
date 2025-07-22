import tensorflow as tf
import os

def get_strategy():

    # Define the strategy
    gpus = tf.config.list_logical_devices('GPU')
    strategy = tf.distribute.MirroredStrategy(gpus)
    #strategy = tf.distribute.MultiWorkerMirroredStrategy(gpus)

    print(f"Using {strategy.num_replicas_in_sync} GPUs for training.")

    return strategy


def get_num_gpus():

    #num_gpus = os.getenv("gpus", 1)  # Fetch from SLURM
    slurm_gpus = os.getenv("SLURM_JOB_GPUS")  # Fetch from SLURM
    num_gpus = len(slurm_gpus.split(",")) if slurm_gpus else 0  # Count GPUs
    print('num_gpus', num_gpus)
    tf.compat.v1.ConfigProto( device_count = {'GPU': 1 , 'CPU': 10} )

    return num_gpus


def set_gpus(num_gpus):

    gpus = tf.config.list_physical_devices('GPU')
    #if num_gpus > 0:
    #    # does not work, nan value in loss:  gen_loss: nan - disc_loss: nan - val_gen_loss: nan - val_disc_loss
    #    tf.keras.mixed_precision.set_global_policy("mixed_float16")

    # Add print statements to VERIFY the policy immediately
    print(f"TensorFlow global compute_dtype: {tf.keras.mixed_precision.global_policy().compute_dtype}")
    print(f"TensorFlow global variable_dtype: {tf.keras.mixed_precision.global_policy().variable_dtype}")

    print('gpus:', gpus)

    # Limit GPU memory usage (optional)
    if num_gpus <= 1:

        if gpus:
            try:
                tf.config.set_visible_devices(gpus[0], 'GPU')
                # Currently, memory growth needs to be the same across GPUs
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                    #tf.config.experimental.set_virtual_device_configuration(gpus[0],
                    #    [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=4096)]  # Set memory limit in MB
                    #    )
                logical_gpus = tf.config.list_logical_devices('GPU')
                print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")

                #if multiple_GPUs_with_virtual_devices:
                #    tf.config.set_logical_device_configuration(
                #        gpus[0],
                #        [tf.config.LogicalDeviceConfiguration(memory_limit=1024),
                #         tf.config.LogicalDeviceConfiguration(memory_limit=1024)])

                #tf.config.set_logical_device_configuration(
                #        gpus[0],
                #        [tf.config.LogicalDeviceConfiguration(memory_limit=40960)],
                #        )
            except RuntimeError as e:
                # Memory growth must be set before GPUs have been initialized
                print('No GPU Error!')
                print(e)

    elif num_gpus > 1:

        if gpus:
            try:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                    #tf.config.experimental.set_virtual_device_configuration(gpu,
                    #    [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=40960)]  # Set memory limit in MB
                    #    )
                    #tf.config.gpu.set_per_process_memory_fraction(0.4)
                    #tf.config.set_logical_device_configuration(gpu,
                    #    [tf.config.LogicalDeviceConfiguration(memory_limit=500)]  # limit in MB, adjust as needed
                    #    )
                print("Enabled GPU memory growth")
            except RuntimeError as e:
                print(e)

    return  
