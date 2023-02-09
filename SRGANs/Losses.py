import tensorflow as tf
from tensorflow.keras import losses

def discriminator_loss(real_Y, fake_Y):

    cross_entropy = losses.BinaryCrossentropy()
    real_loss = cross_entropy(tf.ones(real_Y.shape) - tf.random.uniform(real_Y.shape)*0.2, real_Y)
    fake_loss = cross_entropy(tf.random.uniform(fake_Y.shape)*0.2, fake_Y)
    total_loss = 0.5 * (real_loss + fake_loss)

    return total_loss


def generator_loss(fake_Y, hr_predic, hr_target):

    cross_entropy = losses.BinaryCrossentropy()
    
    adversarial_loss = cross_entropy(
      tf.ones(fake_Y.shape) - tf.random.uniform(fake_Y.shape) * 0.2, 
      fake_Y
    )
    content_loss = losses.MSE(hr_target, hr_predic)

    return content_loss + 1e-5*adversarial_loss
