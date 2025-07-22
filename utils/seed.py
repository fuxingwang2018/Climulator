import os
import random
import numpy as np
import tensorflow as tf

def set_seed(seed=42, disable_parallel = True, enforce_determinism = True):

    # Set random seeds for reproducibility

    #os.environ['TF_DETERMINISTIC_OPS'] = '1'  # Ensure TF uses deterministic algorithms when possible

    # Set seeds across Python, NumPy, and TensorFlow:
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

    #latent_dim = 100  # Change this based on your model
    #noise = tf.random.normal([1, latent_dim], seed=SEED)

    if enforce_determinism:
        # Some TensorFlow ops can be non-deterministic (e.g., certain convs or upsampling). Enforce determinism (Available in TensorFlow ≥ 2.9)
        tf.config.experimental.enable_op_determinism()

    if disable_parallel:
        # Disabling some parallel execution to make results consistent (this will make script very slow, e.g., 90 minutes for EPOCH=2 over Test_Domain on Freja)
        os.environ["OMP_NUM_THREADS"] = "1"
        os.environ["TF_NUM_INTRAOP_THREADS"] = "1"
        os.environ["TF_NUM_INTEROP_THREADS"] = "1"
