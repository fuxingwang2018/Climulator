import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.layers import Conv2D, Dense, Flatten, LayerNormalization, Reshape
from SRGANs import ViT

# https://github.com/paulaharder/deep-downscaling-overview

# This code is based on:
# Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network. https://arxiv.org/abs/1609.04802
# Similar to these repos:
# https://github.com/manishdhakal/SuperResolution/tree/master
# https://github.com/tensorlayer/SRGAN
# https://github.com/HasnainRaz/Fast-SRGAN
# https://github.com/deepak112/Keras-SRGAN
# https://github.com/carlos-gg/dl4ds/tree/master, 
# https://github.com/jackshiwl/srganMedium
# Other GAN codes: "Stochastic Super-Resolution for Downscaling Time-Evolving Atmospheric Fields With a Generative Adversarial Network", 
# https://github.com/fuxingwang2018/mooc-machine-learning-weather-climate/blob/main/tier_3/post_processing/precipitation_downscaling_gan.ipynb
# https://pyimagesearch.com/2022/06/06/super-resolution-generative-adversarial-networks-srgan/

# Torch:
# https://github.com/deu30303/ClimateSD/tree/main, Downscaling Earth System Models with Deep Learning
# https://github.com/lzhengchun/DSGAN/tree/main, Fast and accurate learned multiresolution dynamical downscaling for precipitation
# https://github.com/evbecker/climate-spatial-downscaling/tree/main, https://openreview.net/pdf?id=0Z-jS-aMQFv
# https://github.com/jleinonen/downscaling-rnn-gan , Stochastic Super-Resolution for Downscaling Time-Evolving Atmospheric Fields with a Generative Adversarial Network


def model_generator_no_const_input(nx, nz, channels, subsampling, n_res_block, batch_size):

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


def model_generator(nx, nz, channels, subsampling, n_res_block, batch_size):

    inputs_low_res = layers.Input(shape=(int(nz / subsampling), int(nx / subsampling), channels), batch_size=batch_size, name='low-res-input')
    inputs_high_res = layers.Input(shape=(int(nz), int(nx), 1), batch_size=batch_size, name='high-res-input')  

    # Initial convolutional layers
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(inputs_low_res)
    print('x shape 0:', x.shape)
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    print('x shape 1:', x.shape)

    # Vision Transformer Block
    #vit_output = ViT.vit_block_v3(x, patch_size=4, projection_dim=64)
    vit_output = ViT.vit_block_v1(x, num_patches = int(nz / subsampling) * int(nx / subsampling), projection_dim=64)
    print('vit_output shape:', vit_output.shape)

    # Upsampling and concatenation with orography
    #x = layers.Conv2DTranspose(64, (4, 4), strides=(2, 2), padding='same', activation='relu')(vit_output)
    x = layers.Conv2DTranspose(64, (4, 4), strides=(4, 4), padding='same', activation='relu')(vit_output)
    print('x shape 2:', x.shape)
    x = layers.Concatenate()([x, inputs_high_res])
    print('x shape 3:', x.shape)
    #x = layers.Conv2DTranspose(64, (4, 4), strides=(2, 2), padding='same', activation='relu')(x)
    x = layers.Conv2DTranspose(64, (4, 4), strides=(1, 1), padding='same', activation='relu')(x)
    print('x shape 4:', x.shape)

    # Final output layer
    outputs = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)
    print('outputs shape:', outputs.shape)

    model = models.Model([inputs_low_res, inputs_high_res], outputs, name='Generator')

    print(model.summary())

    return model




def model_generator_correct(nx, nz, channels, subsampling, n_res_block, batch_size):

    inputs_low_res = layers.Input(shape=(int(nz / subsampling), int(nx / subsampling), channels), batch_size=batch_size, name='low-res-input')
    inputs_high_res = layers.Input(shape=(int(nz), int(nx), 1), batch_size=batch_size, name='high-res-input')  

    conv_1 = layers.Conv2D(filters=64, kernel_size=7, strides=1, activation='linear', padding='same')(inputs_low_res)

    prelu_1 = layers.PReLU()(conv_1) #layers.PReLU(alpha_initializer='zeros', alpha_regularizer=None, alpha_constraint=None, shared_axes=[2,3])(conv_1)
    print('inputs_low_res shape:', inputs_low_res.shape)
    print('inputs_high_res shape:', inputs_high_res.shape)
    print('conv_1 shape:', conv_1.shape)
    print('prelu_1 shape:', prelu_1.shape)
    
    res_block = prelu_1

    for index in range(n_res_block):

        res_block = res_block_gen(res_block, 3, 64, 1)

    print('res_block shape:', res_block.shape)

    conv_2 = layers.Conv2D(filters = 64, kernel_size = 3, strides = 1, padding = "same")(res_block) #vit_output)
    batch_1 = layers.BatchNormalization(momentum = 0.5)(conv_2) #axis=1, 
    add_1 = layers.Add()([prelu_1, batch_1])
    #add_1 = layers.Add()([vit_output, batch_1])

    print('conv_2 shape:', conv_2.shape)
    print('batch_1 shape:', batch_1.shape)
    print('add_1 shape:', add_1.shape)
    up_sampling = add_1

    for index in range(int(np.log2(subsampling))):

        up_sampling = up_sampling_block(up_sampling, 3, 256, 1)

    combined_input = layers.Concatenate()([up_sampling, inputs_high_res])
    combined_input = layers.Conv2D(filters=64, kernel_size = 3, strides = 1, padding='same', activation='relu')(combined_input)

    print('up_sampling shape:', up_sampling.shape)
    print('combined_input shape:', combined_input.shape)

    # Vision Transformer Block
    #vit_output = ViT.vit_block_v1(prelu_1, num_patches = int(nz / subsampling) * int(nx / subsampling), projection_dim=64, transformer_layers=2)
    vit_output = ViT.vit_block_v3(combined_input, patch_size = subsampling, projection_dim=64)
    print('vit_output shape:', vit_output.shape)

    #conv_3 = layers.Conv2D(filters = 1, kernel_size = 3, strides = 1, padding = "same")(combined_input)
    conv_3 = layers.Conv2D(filters = 1, kernel_size = 3, strides = 1, padding = "same")(vit_output)
    print('conv_3 shape:', conv_3.shape)

    outputs = conv_3

    model = models.Model([inputs_low_res, inputs_high_res], outputs, name='Generator')

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
