import os
import random
import numpy as np
import tensorflow as tf

# Set random seeds for reproducibility
def set_seed(seed=42):
    os.environ['PYTHONHASHSEED'] = str(seed)
    #os.environ['TF_DETERMINISTIC_OPS'] = '1'  # Ensure TF uses deterministic algorithms when possible
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    #latent_dim = 100  # Change this based on your model
    #noise = tf.random.normal([1, latent_dim], seed=SEED)
