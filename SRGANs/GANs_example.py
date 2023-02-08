import numpy as np
from scipy.io import savemat
from matplotlib import pyplot as plt

wdir = './'
data_folder = './'


import tensorflow as tf
from tensorflow.keras import layers, optimizers
from tensorflow.keras.callbacks import ModelCheckpoint
from sklearn.model_selection import train_test_split


from SRGANs.Model_Generator import model_generator
from SRGANs.Model_Discriminator import model_discriminator
from SRGANs.Losses import generator_loss, discriminator_loss

from SRGANs.srgan import SRGAN


data = np.load('pr_data.npz')
nc3 = data['hr']
nc12 = data['lr']

max_pool_2d = layers.MaxPooling2D(pool_size=4, padding='valid')
nc12_gen = max_pool_2d(nc3)
nc12_gen = nc12_gen.numpy()
print(nc12_gen.shape)


X_train, X_test, y_train, y_test = train_test_split(nc12_gen, nc3, test_size = 3000, random_state = 24)

dataset_train = tf.data.Dataset.from_tensor_slices((X_train, y_train))
dataset_valid = tf.data.Dataset.from_tensor_slices((X_test, y_test))

batch_size = 50
dataset_train = dataset_train.batch(batch_size)
dataset_valid = dataset_valid.batch(batch_size)
#%%

"""
    Training
"""

model_name = 'model_1'


generator_optimizer = optimizers.Adam(1e-3)
discriminator_optimizer = optimizers.Adam(1e-3)

subsampling_lr = 4
n_res_block = 4
input_channels = 1
output_channels = 1

nx = 104
nz = 88

generator =  model_generator(nx, nz, input_channels, subsampling_lr, n_res_block, batch_size)
discriminator = model_discriminator(nx, nz, output_channels, batch_size)

model = SRGAN(generator, discriminator)
model.compile(generator_optimizer, discriminator_optimizer, generator_loss, discriminator_loss)

checkpoint_filepath = wdir + 'checkpoint_NN'

checkpoint = ModelCheckpoint(
    filepath=checkpoint_filepath,
    save_weights_only=True,
    monitor='val_gen_loss',
    mode='min',
    save_best_only=True)

callbacks_list = [checkpoint]

hist = model.fit(dataset_train, epochs = 20, callbacks = callbacks_list, validation_data = dataset_valid, verbose = 1)
savemat(wdir + f'loss_{model_name}.mat', hist.history)
model.load_weights(checkpoint_filepath)

generator.save(wdir + f'{model_name}_generator.h5')
discriminator.save(wdir + f'{model_name}_discriminator.h5')

#%%
pr_pred = generator.predict(X_test)
#%%
n = 100

var_p = pr_pred[n, :,:, 0]
var_ref = y_test[n, :,:, 0]
var_in = X_test[n, :,:, 0]

fig, ax = plt.subplots(1, 3, figsize = (10, 4))

ax[0].imshow(var_in)
ax[1].imshow(var_p)
ax[2].imshow(var_ref)
