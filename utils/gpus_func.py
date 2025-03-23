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

    return num_gpus


def set_gpus(num_gpus):

    gpus = tf.config.list_physical_devices('GPU')
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
            except RuntimeError as e:
                # Memory growth must be set before GPUs have been initialized
                print('No GPU Error!')
                print(e)

    elif num_gpus > 1:

        if gpus:
            try:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                    #tf.config.experimental.set_virtual_device_configuration(gpus[0],
                    #    [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=40960)]  # Set memory limit in MB
                    #    )

                print("Enabled GPU memory growth")
            except RuntimeError as e:
                print(e)

    return  
