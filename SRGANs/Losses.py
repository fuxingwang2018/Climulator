import tensorflow as tf
from tensorflow.keras import losses

# https://github.com/weiji14/deepbedmap/blob/master/srgan_train.ipynb

# https://pyimagesearch.com/2022/06/06/super-resolution-generative-adversarial-networks-srgan/

# Understanding GANs — Deriving the Adversarial loss from scratch | by Hrithick Sen | Analytics Vidhya | Medium
# https://medium.com/analytics-vidhya/understanding-gans-deriving-the-adversarial-loss-from-scratch-ccd8b683d7e2

# Understanding binary cross-entropy / log loss: a visual explanation | by Daniel Godoy | Towards Data Science
# https://towardsdatascience.com/understanding-binary-cross-entropy-log-loss-a-visual-explanation-a3ac6025181a

# Binary Cross-Entropy – Emma Benjaminson – Data Scientist: https://sassafras13.github.io/BiCE/

weight_content_adversarial = 1e-5 #1e-3 # in original version: 1e-5
weight_cross_entropy = 0.2 #1.0 # in original verison 0.2
"""
def discriminator_loss(real_Y, fake_Y):

    cross_entropy = losses.BinaryCrossentropy()
    #real_loss = cross_entropy(tf.ones(real_Y.shape), real_Y)
    #fake_loss = cross_entropy(tf.zeros(fake_Y.shape), fake_Y)
    real_loss = cross_entropy(tf.random.uniform(real_Y.shape), real_Y)
    fake_loss = cross_entropy(tf.ones(real_Y.shape) - tf.random.uniform(real_Y.shape), fake_Y)
    total_loss = 0.5 * (real_loss + fake_loss)

    return total_loss


def generator_loss(fake_Y, hr_predic, hr_target):

    cross_entropy = losses.BinaryCrossentropy()
    
    #adversarial_loss = cross_entropy(tf.ones(fake_Y.shape), fake_Y)
    adversarial_loss = cross_entropy(tf.random.uniform(fake_Y.shape), fake_Y)
    content_loss = losses.MSE(hr_target, hr_predic)

    return content_loss + weight_content_adversarial*adversarial_loss
"""

def discriminator_loss(real_Y, fake_Y):

    cross_entropy = losses.BinaryCrossentropy()
    real_loss = cross_entropy(
        tf.ones(real_Y.shape) - tf.random.uniform(real_Y.shape) * weight_cross_entropy, real_Y)
    fake_loss = cross_entropy(
        tf.random.uniform(fake_Y.shape) * weight_cross_entropy, fake_Y)
    total_loss = 0.5 * (real_loss + fake_loss)

    return total_loss


def generator_loss(fake_Y, hr_predic, hr_target):

    cross_entropy = losses.BinaryCrossentropy()
    
    adversarial_loss = cross_entropy(
        tf.ones(fake_Y.shape) - tf.random.uniform(fake_Y.shape) * weight_cross_entropy, fake_Y)
    content_loss = losses.MSE(hr_target, hr_predic)

    return content_loss + weight_content_adversarial*adversarial_loss
