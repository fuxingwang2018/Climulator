
import tensorflow as tf
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from matplotlib import pyplot as plt
import numpy as np

def scale(scaler, var):

    # Scale data to same magnitude
    shape = var.shape
    var = var.reshape((shape[0], -1))
    var = scaler.fit_transform(var)
    var = var.reshape(shape)
    return scaler, var

class PreProcess(object):
    """ Data pre-processing """

    #def __init__(self):
    #    self. = 

    def scale_dict(self, var_dict_in):

        var_dict_out = {}
        for key, values in var_dict_in.items():
            #print(type(values), values.shape)
            var_dict_out[key] = self.scale_var(values)
            print('scale key:', key)

        return var_dict_out


    def filter_dict(self, var_lr_dict, var_hr_dict, path_figure):

        var_lr_gen_dict = {}
        var_hr_list = list(var_hr_dict.items())
        var_hr_array = np.array(var_hr_list)

        for key, values in var_hr_dict.items():
            #var_lr_gen_dict[key] = self.filter_var(var_lr_dict[key], var_hr_array, path_figure)
            lr_shape = var_lr_dict[key].shape
            hr_shape = var_hr_dict[key].shape
            print('lr_shape', lr_shape, type(lr_shape))
            print('hr_shape', hr_shape, type(hr_shape))
            print('len hr_shape', len(hr_shape))
            # var_lr_dict[key] and var_hr_dict[key] should be divisible for dimension 2 & 3 (1 & 2 in python)
            residue_geo = []
            for i in range(len(lr_shape)):
                residue_geo.append(hr_shape[i] % lr_shape[i])
                if residue_geo[i] != 0:
                    print('can not divisible:', hr_shape[i], lr_shape[i], i, residue_geo[i])
                    if i == 2 and residue_geo[i] == 2:
                        var_hr_dict[key] = var_hr_dict[key][:, :, 1:-1, :] # remove the 1st and last elements


            # the size of the 1st dimension should be divisible by batch_size = 50 
            residue_time_lr = lr_shape[0] % 50
            residue_time_hr = hr_shape[0] % 50
            if residue_time_hr != 0 :
                var_hr_dict[key] = var_hr_dict[key][:-12, :, :, :] # remove the 1st and last elements
            if lr_shape[0] % 50 != 0:
                var_lr_dict[key] = var_lr_dict[key][:-12, :, :, :] # remove the 1st and last elements

            var_lr_gen_dict[key] = self.filter_var(var_lr_dict[key], var_hr_dict[key], path_figure)

        return var_lr_gen_dict


    def scale_var(self, varin):

        scalers_list = []
        print('len(varin)', len(varin))
        #for i in range(len(varin)):
        var = varin #[i]
        scaler = MinMaxScaler()
        scaler, var = scale(scaler, var)
        scalers_list.append(scaler)
        print('var', var.shape)
        var_t = np.expand_dims(var, axis=3)
        #var_t = var.transpose((1, 2, 3, 0))[:-12] # -12, why?
        #varin[i] = var_t
        print('var_t', var_t.shape)

        return var_t 
        #varout = varin[i].transpose((1, 2, 3, 0))[:-12] # -12, why?
        #varout = varout[:, :, 1:-1]  # for 3km, why?
        #return varout


    def filter_var(self, var_lr, var_hr, path_figure):

        # Filter 3km data to 12km
        # lr for low resolution, hr for high resolution

        Pool_Size = 4 
        print ('shape of var_hr, Pool_Size', var_hr.shape, Pool_Size)
        print ('shape of var_lr, Pool_Size', var_lr.shape, Pool_Size)
        max_pool_2d = layers.MaxPooling2D(pool_size=Pool_Size, padding='valid')
        var_lr_gen = max_pool_2d(var_hr)
        var_lr_gen = var_lr_gen.numpy()
        print('var_lr_gen', var_lr_gen.shape)
        print('var_lr', var_lr.shape)
        print('var_hr', var_hr.shape)

        n = 2000

        var_lr_test = var_lr_gen[n, :, :, 0]
        var_hr_test = var_hr[n, :, :, 0]

        fig, ax = plt.subplots(1, 2, figsize = (10, 4))

        ax[0].imshow(var_lr_test)
        ax[1].imshow(var_hr_test)
        fig.savefig(path_figure + "SRGAN_input.png")

        return var_lr_gen

    def split_data(self, var_lr_gen, var_hr, batch_size, variable):

        # Split dataset into subsets 

        #print('var_lr_gen, var_hr:', var_lr_gen.shape, var_hr.shape)
        X_train, X_test, y_train, y_test = train_test_split(var_lr_gen[variable], var_hr[variable], test_size = 3000, random_state = 24)

        dataset_train = tf.data.Dataset.from_tensor_slices((X_train, y_train))
        dataset_valid = tf.data.Dataset.from_tensor_slices((X_test, y_test))

        dataset_train = dataset_train.batch(batch_size)
        dataset_valid = dataset_valid.batch(batch_size)

        print('Training dataset shape:', X_train.shape, y_train.shape)
        print('Testing dataset shape:', X_test.shape, y_test.shape)
        print('dataset_train shape:', dataset_train.element_spec)
        print('dataset_valid shape:', dataset_valid.element_spec)

        shape = X_train.shape
        X_trainr = X_train.reshape((shape[0], -1))
        shape = X_test.shape
        X_testr = X_test.reshape((shape[0], -1))
        shape = y_train.shape
        y_trainr = y_train.reshape((shape[0], -1))
        shape = y_test.shape
        y_testr = y_test.reshape((shape[0], -1))

        #var = var.reshape(shape)
        #X_trainr = X_train.ravel()
        #X_testr = X_test.ravel()
        #y_trainr = y_train.ravel()
        #y_testr = y_test.ravel()

        print('Training dataset shape:', X_trainr.shape, y_trainr.shape)
        print('Testing dataset shape:', X_testr.shape, y_testr.shape)

        return dataset_train, dataset_valid, X_train, X_test, y_train, y_test
        #return dataset_train, dataset_valid, X_trainr, X_testr, y_trainr, y_testr

