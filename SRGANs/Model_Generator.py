import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models

# https://github.com/paulaharder/deep-downscaling-overview

# Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network. https://arxiv.org/abs/1609.04802
# https://github.com/manishdhakal/SuperResolution/tree/master
# https://github.com/tensorlayer/SRGAN
# https://github.com/HasnainRaz/Fast-SRGAN
# https://github.com/deepak112/Keras-SRGAN
# https://github.com/carlos-gg/dl4ds/tree/master, 
# https://github.com/fuxingwang2018/mooc-machine-learning-weather-climate/blob/main/tier_3/post_processing/precipitation_downscaling_gan.ipynb
# https://github.com/jackshiwl/srganMedium

# Torch:
# https://github.com/deu30303/ClimateSD/tree/main, Downscaling Earth System Models with Deep Learning
# https://github.com/lzhengchun/DSGAN/tree/main, Fast and accurate learned multiresolution dynamical downscaling for precipitation
# https://github.com/evbecker/climate-spatial-downscaling/tree/main, https://openreview.net/pdf?id=0Z-jS-aMQFv
# https://github.com/jleinonen/downscaling-rnn-gan , Stochastic Super-Resolution for Downscaling Time-Evolving Atmospheric Fields with a Generative Adversarial Network

def model_generator(nx, nz, channels, subsampling, n_res_block, batch_size):

    inputs = layers.Input(shape=(int(nz / subsampling), int(nx / subsampling), channels), batch_size=batch_size, name='low-res-input')

    conv_1 = layers.Conv2D(filters=64, kernel_size=7, strides=1, activation='linear', padding='same')(inputs)

    prelu_1 = layers.PReLU()(conv_1) #layers.PReLU(alpha_initializer='zeros', alpha_regularizer=None, alpha_constraint=None, shared_axes=[2,3])(conv_1)

    res_block = prelu_1

    for index in range(n_res_block):

        res_block = res_block_gen(res_block, 3, 64, 1)


    conv_2 = layers.Conv2D(filters = 64, kernel_size = 3, strides = 1, padding = "same")(res_block)
    batch_1 = layers.BatchNormalization(momentum = 0.5)(conv_2) #axis=1, 
    add_1 = layers.Add()([prelu_1, batch_1])

    up_sampling = add_1

    for index in range(int(np.log2(subsampling))):

        up_sampling = up_sampling_block(up_sampling, 3, 256, 1)

    conv_3 = layers.Conv2D(filters = 1, kernel_size = 3, strides = 1, padding = "same")(up_sampling)
    outputs = conv_3


    model = models.Model(inputs, outputs, name='Generator')


    print(model.summary())

    return model


def res_block_gen(model, kernal_size, filters, strides):

    gen = model
    
    model = layers.Conv2D(filters = filters, kernel_size = kernal_size, strides = strides, padding = "same")(model)
    model = layers.BatchNormalization(momentum = 0.5)(model)
    # Using Parametric ReLU
    model = layers.PReLU()(model) #layers.PReLU(alpha_initializer='zeros', alpha_regularizer=None, alpha_constraint=None, shared_axes=[2,3])(model)
    model = layers.Conv2D(filters = filters, kernel_size = kernal_size, strides = strides, padding = "same")(model)
    model = layers.BatchNormalization(momentum = 0.5)(model)
        
    model = layers.Add()([gen, model])
    
    return model


def up_sampling_block(model, kernal_size, filters, strides):

    # In place of Conv2D and UpSampling2D we can also use Conv2DTranspose (Both are used for Deconvolution)
    # Even we can have our own function for deconvolution (i.e one made in Utils.py)
    #model = Conv2DTranspose(filters = filters, kernel_size = kernal_size, strides = strides, padding = "same")(model)
    model = layers.Conv2D(filters = filters, kernel_size = kernal_size, strides = strides, padding = "same")(model)
    model = layers.UpSampling2D(size = 2)(model)
    # model = SubpixelConv2D(model.shape, scale=2)(model)
    model = layers.LeakyReLU(alpha = 0.2)(model)
    
    return model


def SubpixelConv2D(input_shape, scale=4):
    
    def subpixel_shape(input_shape):
        dims = [input_shape[0],
                int(input_shape[1] / (scale ** 2)),
                input_shape[2] * scale,
                input_shape[3] * scale]
        output_shape = tuple(dims)
        return output_shape

    def subpixel(x):
        return tf.nn.depth_to_space(x, scale, data_format='NCHW')


    return layers.Lambda(subpixel, output_shape=subpixel_shape)
