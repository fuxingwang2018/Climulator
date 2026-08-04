import tensorflow as tf
from tensorflow.keras import losses
from utils import gpus_func 
import get_configuration
import os

# https://github.com/weiji14/deepbedmap/blob/master/srgan_train.ipynb

# https://pyimagesearch.com/2022/06/06/super-resolution-generative-adversarial-networks-srgan/

# Understanding GANs — Deriving the Adversarial loss from scratch | by Hrithick Sen | Analytics Vidhya | Medium
# https://medium.com/analytics-vidhya/understanding-gans-deriving-the-adversarial-loss-from-scratch-ccd8b683d7e2

# Understanding binary cross-entropy / log loss: a visual explanation | by Daniel Godoy | Towards Data Science
# https://towardsdatascience.com/understanding-binary-cross-entropy-log-loss-a-visual-explanation-a3ac6025181a

# Binary Cross-Entropy – Emma Benjaminson – Data Scientist: https://sassafras13.github.io/BiCE/

# Get configuration 
args = get_configuration.get_args()
config_file = args.config
if not os.path.isfile(config_file):
    raise ValueError(f"\nConfig file, '{config_file}', does not exist!")
cdict = get_configuration.get_settings(config_file)

#weight_cross_entropy = 0.2 #0.2 #1.0 # in original verison 0.2
#lambda_corr = 0.0 #default 0.1; tune from 0.05 to 0.5
weight_adversarial = cdict['stats_conf']['LOSS']['WEIGHT_ADVERSARIAL'] #1e-5
weight_content = cdict['stats_conf']['LOSS']['WEIGHT_CONTENT'] #1.0
weight_cross_entropy = cdict['stats_conf']['LOSS']['WEIGHT_CROSS_ENTROPY'] #0.2
lambda_corr = cdict['stats_conf']['LOSS']['LAMBDA_CORR'] #0.0
content_loss_func = cdict['stats_conf']['LOSS']['CONTENT_LOSS_FUNC']
losses_reduction = cdict['stats_conf']['LOSS']['LOSSES_REDUCTION']
label_smoothing_real = cdict['stats_conf']['LOSS']['LABEL_SMOOTHING_REAL']
disc_noise_std = cdict['stats_conf']['LOSS']['DISCRIMINATOR_NOISE_STD'] #0.0 -> 0.03  
disc_label_noise = cdict['stats_conf']['LOSS']['DISCRIMINATOR_LABEL_NOISE'] #1.0 -> 0.1  

print('weight_adversarial, weight_content, weight_cross_entropy, lambda_corr', \
        weight_adversarial, weight_content, weight_cross_entropy, lambda_corr) 

def discriminator_loss_test(real_Y, fake_Y):

    # Define loss function with correct reduction
    num_gpus = gpus_func.get_num_gpus()
    if num_gpus <= 1: 
        cross_entropy = losses.BinaryCrossentropy()
    elif num_gpus > 1: 
        cross_entropy = tf.keras.losses.BinaryCrossentropy(reduction=tf.keras.losses.Reduction.NONE)

    weight = 1.0 #0.5

    #cross_entropy = losses.BinaryCrossentropy()
    real_loss = cross_entropy(tf.ones_like(real_Y), real_Y)
    fake_loss = cross_entropy(tf.zeros_like(fake_Y), fake_Y)
    #real_loss = cross_entropy(tf.random.uniform(real_Y.shape), real_Y)
    #fake_loss = cross_entropy(tf.ones(real_Y.shape) - tf.random.uniform(real_Y.shape), fake_Y)

    if num_gpus > 1: 
        strategy = gpus_func.get_strategy()
        global_batch_size = tf.shape(real_Y)[0] * strategy.num_replicas_in_sync
        real_loss = tf.reduce_sum(real_loss) / tf.cast(global_batch_size, tf.float32)
        fake_loss = tf.reduce_sum(fake_loss) / tf.cast(global_batch_size, tf.float32)

    total_loss = weight * (real_loss + fake_loss)

    return total_loss


def generator_loss_test(fake_Y, hr_predic, hr_target):

    # Define loss function with correct reduction
    num_gpus = gpus_func.get_num_gpus()
    if num_gpus <= 1: 
        cross_entropy = losses.BinaryCrossentropy()
    elif num_gpus > 1: 
        cross_entropy = tf.keras.losses.BinaryCrossentropy(reduction=tf.keras.losses.Reduction.NONE)

    weight = 1e-3 #1e-5

    #cross_entropy = losses.BinaryCrossentropy()
    
    adversarial_loss = cross_entropy(tf.ones_like(fake_Y), fake_Y)
    #adversarial_loss = cross_entropy(tf.random.uniform(fake_Y.shape), fake_Y)
    content_loss = losses.MSE(hr_target, hr_predic)

    if num_gpus > 1: 
        # Compute mean loss and scale by global batch size
        strategy = gpus_func.get_strategy()
        global_batch_size = tf.shape(fake_Y)[0] * strategy.num_replicas_in_sync
        adversarial_loss = tf.reduce_sum(adversarial_loss) / tf.cast(global_batch_size, tf.float32)

    return content_loss + weight*adversarial_loss


def discriminator_loss_origin(real_Y, fake_Y):

    cross_entropy = losses.BinaryCrossentropy()
    real_loss = cross_entropy(
        tf.ones(real_Y.shape) - tf.random.uniform(real_Y.shape) * weight_cross_entropy, real_Y)
    fake_loss = cross_entropy(
        tf.random.uniform(fake_Y.shape) * weight_cross_entropy, fake_Y)
    total_loss = 0.5 * (real_loss + fake_loss)

    return total_loss


def generator_loss_origin(fake_Y, hr_predic, hr_target):

    num_gpus = gpus_func.get_num_gpus()
    cross_entropy = losses.BinaryCrossentropy()
    
    print('fake_Y.shape:', fake_Y.shape)
    adversarial_loss = cross_entropy(
        tf.ones(fake_Y.shape) - tf.random.uniform(fake_Y.shape) * weight_cross_entropy, fake_Y)
    content_loss = losses.MSE(hr_target, hr_predic)
    #content_loss = tf.cast(content_loss, dtype=tf.float32)  # Convert y to float32 before adding

    if num_gpus > 1: 
        # Compute mean loss and scale by global batch size
        strategy = gpus_func.get_strategy()
        global_batch_size = tf.shape(fake_Y)[0] * strategy.num_replicas_in_sync
        adversarial_loss = tf.reduce_sum(adversarial_loss) / tf.cast(global_batch_size, tf.float32)

    print(' content_loss.dtype, adversarial_loss.dtype:', content_loss.dtype, adversarial_loss.dtype) 
    return weight_content*content_loss + weight_adversarial*adversarial_loss


def discriminator_loss(real_Y, fake_Y):

    #real_Y = tf.cast(real_Y, tf.float32)
    #fake_Y = tf.cast(fake_Y, tf.float32)

    # Define loss function with correct reduction
    num_gpus = gpus_func.get_num_gpus()
    #cross_entropy = losses.BinaryCrossentropy()
    #if num_gpus <= 1: 
    #    cross_entropy = losses.BinaryCrossentropy()
    #elif num_gpus > 1: 
    #    if losses_reduction == 'NONE':
    #        cross_entropy = tf.keras.losses.BinaryCrossentropy(
    #            from_logits=False,
    #            reduction=tf.keras.losses.Reduction.NONE
    #            )
    #    elif losses_reduction == 'SUM_OVER_BATCH_SIZE':
    #        cross_entropy = tf.keras.losses.BinaryCrossentropy(
    #            from_logits=False, 
    #            reduction=tf.keras.losses.Reduction.SUM_OVER_BATCH_SIZE
    #        ) 
    #ce_reduction = tf.keras.losses.Reduction.NONE if num_gpus > 1 else tf.keras.losses.Reduction.AUTO
    ce_reduction = "none" if num_gpus > 1 else "sum_over_batch_size"
    cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=False, reduction=ce_reduction)

    # ADD INPUT NOISE
    if disc_noise_std > 0:
        real_Y = real_Y + tf.random.normal(tf.shape(real_Y), stddev=disc_noise_std)
        fake_Y = fake_Y + tf.random.normal(tf.shape(fake_Y), stddev=disc_noise_std)

    real_labels = (tf.ones_like(real_Y) * label_smoothing_real - tf.random.uniform(real_Y.shape) * weight_cross_entropy) * disc_label_noise
    #tf.ones(real_Y.shape) * label_smoothing_real - tf.random.uniform(real_Y.shape) * weight_cross_entropy, real_Y)
    fake_labels = (tf.random.uniform(fake_Y.shape) * weight_cross_entropy) * disc_label_noise

    real_loss = cross_entropy(real_labels, real_Y)
    fake_loss = cross_entropy(fake_labels, fake_Y)

    if len(real_loss.shape) >= 2:
        real_loss = tf.reduce_mean(real_loss, axis=[1, 2])
        fake_loss = tf.reduce_mean(fake_loss, axis=[1, 2])

    if num_gpus > 1: 
        strategy = gpus_func.get_strategy()
        real_loss = tf.nn.compute_average_loss(real_loss)
        fake_loss = tf.nn.compute_average_loss(fake_loss)
        #global_batch_size = tf.shape(real_Y)[0] * strategy.num_replicas_in_sync
        #real_loss = tf.reduce_sum(real_loss) / tf.cast(global_batch_size, tf.float32)
        #fake_loss = tf.reduce_sum(fake_loss) / tf.cast(global_batch_size, tf.float32)

    #tf.print('real_loss:', real_loss, 'fake_loss:', fake_loss)
    #tf.print("Val Real_Y Max:", tf.reduce_max(real_Y), "Min:", tf.reduce_min(real_Y))
    disc_loss = 0.5 * (real_loss + fake_loss)
    #tf.print(">>> DEBUG BATCH LOSS:", disc_loss)

    return disc_loss, real_loss, fake_loss


def generator_loss_default(fake_Y, hr_predic, hr_target):

    # Define loss function with correct reduction
    num_gpus = gpus_func.get_num_gpus()
    if num_gpus <= 1: 
        cross_entropy = losses.BinaryCrossentropy()
    elif num_gpus > 1: 
        cross_entropy = tf.keras.losses.BinaryCrossentropy(reduction=tf.keras.losses.Reduction.NONE)

    #cross_entropy = losses.BinaryCrossentropy(reduction=tf.keras.losses.Reduction.NONE)
    
    #print('fake_Y.shape:', fake_Y.shape)
    adversarial_loss = cross_entropy(
        tf.ones(fake_Y.shape) - tf.random.uniform(fake_Y.shape) * weight_cross_entropy, fake_Y)
    content_loss = losses.MSE(hr_target, hr_predic)
    #content_loss = tf.cast(content_loss, dtype=tf.float32)  # Convert y to float32 before adding

    if num_gpus > 1: 
        # Compute mean loss and scale by global batch size
        strategy = gpus_func.get_strategy()
        global_batch_size = tf.shape(fake_Y)[0] * strategy.num_replicas_in_sync
        content_loss = tf.reduce_sum(content_loss) / tf.cast(global_batch_size, tf.float32)
        adversarial_loss = tf.reduce_sum(adversarial_loss) / tf.cast(global_batch_size, tf.float32)

    #print(' content_loss.dtype, adversarial_loss.dtype:', content_loss.dtype, adversarial_loss.dtype) 
    return weight_content*content_loss + weight_adversarial*adversarial_loss


def generator_loss(fake_Y, hr_predic, hr_target):

    #fake_Y = tf.cast(fake_Y, tf.float32)
    #hr_predic = tf.cast(hr_predic, tf.float32)
    #hr_target = tf.cast(hr_target, tf.float32)

    num_gpus = gpus_func.get_num_gpus()
    #if num_gpus <= 1:
    #    cross_entropy = losses.BinaryCrossentropy()
    #else:
    #    if losses_reduction == 'NONE':
    #        cross_entropy = tf.keras.losses.BinaryCrossentropy(
    #            from_logits=False, 
    #            reduction=tf.keras.losses.Reduction.NONE
    #        )
    #    elif losses_reduction == 'SUM_OVER_BATCH_SIZE':
    #        cross_entropy = losses.BinaryCrossentropy(
    #            from_logits=False, 
    #            reduction=tf.keras.losses.Reduction.SUM_OVER_BATCH_SIZE
    #        )
    #ce_reduction = tf.keras.losses.Reduction.NONE if num_gpus > 1 else tf.keras.losses.Reduction.AUTO
    ce_reduction = "none" if num_gpus > 1 else "sum_over_batch_size"
    cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=False, reduction=ce_reduction)

    adversarial_loss = cross_entropy(
        #tf.ones(fake_Y.shape) - tf.random.uniform(fake_Y.shape) * weight_cross_entropy,
        tf.ones_like(fake_Y) - tf.random.uniform(fake_Y.shape) * weight_cross_entropy,
        fake_Y
    )

    #if content_loss_func == 'MSE':
    content_loss = losses.MSE(hr_target, hr_predic)
    content_loss = tf.reduce_sum(content_loss, axis=[1, 2])
    # get Height (index 1) and Width (index 2)
    target_shape = tf.shape(hr_target)
    height = tf.cast(target_shape[1], tf.float32)
    width = tf.cast(target_shape[2], tf.float32)
    num_pixels = height * width
    #tf.print('num_pixels', num_pixels, 'width', width, 'height', height)
    content_loss = content_loss / num_pixels
    #elif content_loss_func == 'REDUCE_MEAN':
    #content_loss = tf.reduce_mean(tf.square(hr_target - hr_predic), axis=[1, 2])

    # NEW: correlation preservation loss
    corr_loss = correlation_loss(hr_predic, hr_target)

    if num_gpus > 1:
        strategy = gpus_func.get_strategy()
        # This function sums the loss across all GPUs and divides by the GLOBAL batch size automatically.
        content_loss = tf.nn.compute_average_loss(content_loss)
        adversarial_loss = tf.nn.compute_average_loss(adversarial_loss)
        # global_batch_size = tf.shape(fake_Y)[0] * strategy.num_replicas_in_sync
        # content_loss = tf.reduce_sum(content_loss) / tf.cast(global_batch_size, tf.float32)
        # adversarial_loss = tf.reduce_sum(adversarial_loss) / tf.cast(global_batch_size, tf.float32)

    #tf.print('adversarial_loss:', adversarial_loss, 'content_loss:', content_loss)
    #print('weight_adversarial, weight_content', weight_adversarial, weight_content)

    gen_loss = weight_content * content_loss + \
        weight_adversarial * adversarial_loss + \
        lambda_corr * corr_loss

    return gen_loss, content_loss, adversarial_loss


def correlation_loss(hr_predic, hr_target, eps=1e-6):
    """
    Computes correlation loss between the two output channels
    using pure TensorFlow (no tfp).
    """
    # Flatten spatial dims: [batch, H*W, 2]
    pred = tf.reshape(hr_predic, [tf.shape(hr_predic)[0], -1, 2])
    true = tf.reshape(hr_target, [tf.shape(hr_target)[0], -1, 2])

    def corr(a):
        """
        Pearson correlation between channel 0 and channel 1.
        a shape: [N,2]
        """
        x = a[:, 0]
        y = a[:, 1]

        x_mean = tf.reduce_mean(x)
        y_mean = tf.reduce_mean(y)

        xm = x - x_mean
        ym = y - y_mean

        numerator = tf.reduce_sum(xm * ym)
        denominator = tf.sqrt(tf.reduce_sum(tf.square(xm)) * tf.reduce_sum(tf.square(ym)) + eps)

        return numerator / (denominator + eps)

    # Compute correlation difference for each sample
    batch_corr_loss = tf.map_fn(
        lambda xy: tf.abs(corr(xy[0]) - corr(xy[1])),
        (pred, true),
        dtype=tf.float32
    )

    return tf.reduce_mean(batch_corr_loss)

