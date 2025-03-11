from tensorflow.keras import layers, models
from SRGANs import ViT
from tensorflow.keras.regularizers import l2


# Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network. https://arxiv.org/abs/1609.04802
# https://github.com/manishdhakal/SuperResolution/tree/master
# https://github.com/tensorlayer/SRGAN
# Torch:
# https://github.com/deu30303/ClimateSD/tree/main, Downscaling Earth System Models with Deep Learning

def model_discriminator(nx, nz, channels, batch_size, dropout_rate):

    inputs = layers.Input(shape=(nz, nx, channels), batch_size=batch_size, name='high-res-input')
        
    model = layers.Conv2D(filters = 16, kernel_size = 7, strides = 1, padding = "same")(inputs)

    if dropout_rate >= 0.0 and dropout_rate <= 1.0:
        model = layers.Dropout(dropout_rate)(model)  # Add Dropout, increase 0.3 if overfitting persists.

    model = layers.LeakyReLU(alpha = 0.2)(model)
    
    #model = discriminator_block(model, 16, 3, 2)
    #model = discriminator_block(model, 32, 3, 1)
    #model = discriminator_block(model, 32, 3, 2)
    #model = discriminator_block(model, 64, 3, 1)
    #model = discriminator_block(model, 64, 3, 2)
    #model = discriminator_block(model, 128, 3, 1)
    #model = discriminator_block(model, 128, 3, 2)

    model = discriminator_block(model, 16, 3, 2, dropout_rate)
    model = discriminator_block(model, 32, 3, 1, dropout_rate)
    model = discriminator_block(model, 32, 3, 2, dropout_rate)
    model = discriminator_block(model, 64, 3, 1, dropout_rate)
    model = discriminator_block(model, 64, 3, 2, dropout_rate)
    model = discriminator_block(model, 128, 3, 1, dropout_rate)
    model = discriminator_block(model, 128, 3, 2, dropout_rate)
    
    model = layers.Flatten()(model)
    model = layers.Dense(1024)(model)
    model = layers.LeakyReLU(alpha = 0.2)(model)

    model = layers.Dense(1)(model)
    model = layers.Activation('sigmoid')(model) 
    
    model = models.Model(inputs=inputs, outputs = model, name='Discriminator')


    print(model.summary())
    
    return model


def model_discriminator_vit(nx, nz, channels, batch_size):

    inputs = layers.Input(shape=(nz, nx, channels), batch_size=batch_size, name='high-res-input')
        
    model = layers.Conv2D(filters = 16, kernel_size = 7, strides = 1, padding = "same")(inputs)
    model = layers.LeakyReLU(alpha = 0.2)(model)
    
    model = discriminator_block(model, 16, 3, 2)
    model = discriminator_block(model, 32, 3, 1)
    model = discriminator_block(model, 32, 3, 2)

    # Vision Transformer Block
    print('model shape 0:', model.shape)
    subsampling = 4
    vit_output = ViT.vit_block_v1(model, num_patches = int(nz / subsampling) * int(nx / subsampling), projection_dim=32)
    model = discriminator_block(vit_output, 64, 3, 1)

    #model = discriminator_block(model, 64, 3, 1)
    model = discriminator_block(model, 64, 3, 2)
    model = discriminator_block(model, 128, 3, 1)
    model = discriminator_block(model, 128, 3, 2)

    model = layers.Flatten()(model)
    model = layers.Dense(1024)(model)
    model = layers.LeakyReLU(alpha = 0.2)(model)

    model = layers.Dense(1)(model)
    model = layers.Activation('sigmoid')(model) 
    
    model = models.Model(inputs=inputs, outputs = model, name='Discriminator')


    print(model.summary())
    
    return model



def discriminator_block(model, filters, kernel_size, strides, dropout_rate):
        
    model = layers.Conv2D(filters = filters, kernel_size = kernel_size, strides = strides, padding = "same")(model)
    if dropout_rate >= 0.0 and dropout_rate <= 1.0:
        model = layers.Dropout(dropout_rate)(model)  # Add Dropout, increase 0.3 if overfitting persists.
    model = layers.BatchNormalization(momentum = 0.5)(model)
    model = layers.LeakyReLU(alpha = 0.2)(model)
    
    return model
