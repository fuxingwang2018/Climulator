import tensorflow as tf
import os

def get_strategy():

    # Define the strategy
    gpus = tf.config.list_logical_devices('GPU')
    strategy = tf.distribute.MirroredStrategy(gpus)

    print(f"Using {strategy.num_replicas_in_sync} GPUs for training.")

    return strategy


def get_num_gpus():

    #num_gpus = os.getenv("gpus", 1)  # Fetch from SLURM
    slurm_gpus = os.getenv("SLURM_JOB_GPUS")  # Fetch from SLURM
    num_gpus = len(slurm_gpus.split(",")) if slurm_gpus else 0  # Count GPUs
    print('num_gpus', num_gpus)

    return num_gpus
