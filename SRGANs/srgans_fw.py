import numpy as np
import netCDF4
import glob
import time

from scipy.io import loadmat, savemat
from matplotlib import pyplot as plt

import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, metrics, losses
from tensorflow.keras.callbacks import ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import sys

from SRGANs.Model_Generator import model_generator
from SRGANs.Model_Discriminator import model_discriminator
from SRGANs.Losses import generator_loss, discriminator_loss
from SRGANs.srgan import SRGAN

# 3km:  tas, pr 
# 12km: ta500,  ta700,  ta850,  ta950, 
#	hus500, hus700, hus850, hus950, 
#	ua500,  ua700,  ua850,  ua950, 
#	va500,  va700,  va850,  va950, 
#	phi500, phi700, phi850, phi950,
# var = 'ta500' 
# exp_name = '12km' # '3km', '12km'

""" 
inputs: phi500, phi700, phi850, phi950, hus500, hus700, hus850, hus950,  ta500, ta700, ta850, ta950, ua500, ua700, ua850, ua950, va500, va700, va850, va950
outputs: pr 
"""


wdir = '/nobackup/rossby26/users/sm_fuxwa/AI/SRGAN_OUT/'
data_folder = '/nobackup/rossby26/users/sm_fuxwa/AI/'
path_figure = '/nobackup/rossby26/users/sm_fuxwa/AI/Figure_Epoch100/'

dir_lr = data_folder + '12km/6hr/' 
dir_hr = data_folder + '3km/6hr/'

"""
    Read data from NetCDF
"""

lr_files = glob.glob(dir_lr + '*')
lr_files.sort()
print(lr_files)
lr_files = lr_files[16:17]
# lr_files.pop(12)

hr_files = glob.glob(dir_hr + '*')
hr_files.sort()
print(hr_files)
# hr_files.pop(1)
hr_files = hr_files[0:1]

nc12 = []
var_list12 = []
for file in lr_files:
  data = netCDF4.Dataset(file)
  var = list(data.variables.keys())[-1]
  nc12.append(data.variables[var])
  var_list12.append(var)

nc3 = []
var_list3 = []
for file in hr_files:
  data = netCDF4.Dataset(file)
  var = list(data.variables.keys())[-1]
  nc3.append(data.variables[var])
  var_list3.append(var)

print(var_list12)
print(var_list3)

nc12 = np.array(nc12)
nc3 = np.array(nc3)

"""
    Scale data to same magnitude
"""

def scale(scaler, var):
  shape = var.shape
  var = var.reshape((shape[0], -1))
  var = scaler.fit_transform(var)
  var = var.reshape(shape)
  return scaler, var

scalers_list12 = []
for i in range(len(nc12)):
  var = nc12[i]
  scaler = MinMaxScaler()
  scaler, var = scale(scaler, var)
  scalers_list12.append(scaler)
  nc12[i] = var

scalers_list3 = []
for i in range(len(nc3)):
  var = nc3[i]
  scaler = MinMaxScaler()
  scaler, var = scale(scaler, var)
  scalers_list3.append(scaler)
  nc3[i] = var

nc12 = nc12.transpose((1, 2, 3, 0))[:-12]
nc3 = nc3.transpose((1, 2, 3, 0))[:-12]
nc3 = nc3[:, :, 1:-1]

print(nc12.shape)
print(nc3.shape)

"""
    Filter 3km data to 12km
"""
max_pool_2d = layers.MaxPooling2D(pool_size=4, padding='valid')
nc12_gen = max_pool_2d(nc3)
nc12_gen = nc12_gen.numpy()
print(nc12_gen.shape)

n = 2000

var12 = nc12_gen[n, :, :, 0]
var3 = nc3[n, :, :, 0]

fig, ax = plt.subplots(1, 2, figsize = (10, 4))

ax[0].imshow(var12)
ax[1].imshow(var3)
fig.savefig(path_figure + "SRGAN_input.png")

"""
    Split dataset into subsets 
"""

X_train, X_test, y_train, y_test = train_test_split(nc12_gen, nc3, test_size = 3000, random_state = 24)

dataset_train = tf.data.Dataset.from_tensor_slices((X_train, y_train))
dataset_valid = tf.data.Dataset.from_tensor_slices((X_test, y_test))

batch_size = 50
dataset_train = dataset_train.batch(batch_size)
dataset_valid = dataset_valid.batch(batch_size)

"""
    Training
"""

model_name = f"model_1"

generator_optimizer = optimizers.Adam(1e-4)
discriminator_optimizer = optimizers.Adam(1e-4)

subsampling_lr = 4
n_res_block = 8
input_channels = 1
output_channels = 1

nx = 104
nz = 88

generator =  model_generator(nx, nz, input_channels, subsampling_lr, n_res_block, batch_size)
discriminator = model_discriminator(nx, nz, output_channels, batch_size)

model = SRGAN(generator, discriminator)
model.compile(generator_optimizer, discriminator_optimizer, generator_loss, discriminator_loss)

checkpoint_filepath = wdir + 'checkpoint_NN'

# Create a callback that saves the model's weights
checkpoint = ModelCheckpoint(
    filepath=checkpoint_filepath,
    save_weights_only=True,
    monitor='val_gen_loss',
    mode='min',
    save_best_only=True)

callbacks_list = [checkpoint]

# Train the model with the new callback
# Model weights are saved at the end of every epoch, if it's the best seen so far.
EPOCHS = 100 #100
hist = model.fit(dataset_train, epochs = EPOCHS, callbacks = callbacks_list, validation_data = dataset_valid, verbose = 1)
savemat(wdir + f'loss_{model_name}.mat', hist.history)

# The model weights (that are considered the best) are loaded into the model.
model.load_weights(checkpoint_filepath)

# Save the model to a HDF5 file.
generator.save(wdir + f'{model_name}_generator.h5')
discriminator.save(wdir + f'{model_name}_discriminator.h5')

# Prediction
pr_pred = generator.predict(X_test)
np.savez_compressed(wdir + 'preds', hr = y_test, hr_p = pr_pred)

"""
    Plot final result
"""

n = 50

var_p = pr_pred[n, :,:, 0]
var_ref = y_test[n, :,:, 0]
var_in = X_test[n, :,:, 0]

fig, ax = plt.subplots(1, 3, figsize = (10, 4))

ax[0].imshow(var_in)
ax[1].imshow(var_p)
ax[2].imshow(var_ref)
fig.savefig(path_figure + "SRGAN_result.png")

