
import tensorflow as tf
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from matplotlib import pyplot as plt
import numpy as np


class PreProcess(object):
    """ Data pre-processing """

    #def __init__(self):
    #    self. = 


    def scale(self, scaler, var):

        """
        Scale data to same magnitude

        :param scaler: 
        :type scaler: 
        :param var: The variable to scale
        :type var: array
        :return: scaled data, var
        :rtype: array
        """

        shape = var.shape
        var = var.reshape((shape[0], -1))
        var = scaler.fit_transform(var)
        var = var.reshape(shape)

        return scaler, var


    def scale_dict(self, var_dict_in):

        """
        Scale data to same magnitude

        :param scaler: 
        :type scaler: 
        :param var: The variable to scale
        :type var: array
        :return: scaled data, var
        :rtype: array
        """

        var_dict_out = {}
        for key, values in var_dict_in.items():
            #print(type(values), values.shape)
            var_dict_out[key] = self.scale_var(values)
            print('scale key:', key)

        return var_dict_out


    def filter_dict(self, var_low_res_dict, var_high_res_dict):

        """
        Filter data from high resolution (hr, eg, 3 km) to low resolution (lr, eg, 12 km)

        :param var_low_res_dict: variable at low resolution 
        :type var_low_res_dict: dictionary
        :param var_high_res_dict: variable at high resolution 
        :type var_high_res_dict: dictionary
        :return: var_low_res_gen_dict, generated variable at low resolution from high resolution variable
        :rtype: dictionary
        """

        var_low_res_gen_dict = {}
        for key, values in var_high_res_dict.items():
            var_low_res_gen_dict[key] = self.filter_var(var_low_res_dict[key], var_high_res_dict[key])

        return var_low_res_gen_dict


    def adjust_data_size(self, var_low_res_dict, var_high_res_dict, batch_size):

        """
        Filter data from high resolution (hr, eg, 3 km) to low resolution (lr, eg, 12 km)

        :param var_low_res_dict: variable at low resolution 
        :type var_low_res_dict: dictionary
        :param var_high_res_dict: variable at high resolution 
        :type var_high_res_dict: dictionary
        :return: size adjusted variable, var_low_res_adjusted, var_high_res_adjusted
        :rtype: dictionary
        """
 
        """
        low_res_shape = var_low_res.shape
        high_res_shape = var_high_res.shape

        print('low_res_shape', low_res_shape, type(low_res_shape))
        print('high_res_shape', high_res_shape, type(high_res_shape))
        print('len high_res_shape', len(high_res_shape))

        # var_low_res and var_high_res should be divisible for dimension 2 & 3 (1 & 2 in python)
        residue_geo = []
        for i in range(len(low_res_shape)):
            residue_geo.append(high_res_shape[i] % low_res_shape[i])
            if residue_geo[i] != 0:
                print('can not divisible:', high_res_shape[i], low_res_shape[i], i, residue_geo[i])
                if i == 2 and residue_geo[i] == 2:
                    var_high_res = var_high_res[:, :, 1:-1, :] # remove the 1st and last elements

        # the size of the 1st dimension should be divisible by batch_size = 50 
        residue_time_low_res = low_res_shape[0] % 50
        residue_time_high_res = high_res_shape[0] % 50
        if residue_time_high_res != 0 :
            var_high_res = var_high_res[:-12, :, :, :] # remove the 1st and last elements
        if low_res_shape[0] % 50 != 0:
            var_low_res = var_low_res[:-12, :, :, :] # remove the 1st and last elements

        """
        var_low_res_adjusted_dict = var_low_res_dict
        var_high_res_adjusted_dict = var_high_res_dict

        for key, values in var_high_res_adjusted_dict.items():
                        
            low_res_shape = var_low_res_adjusted_dict[key].shape
            high_res_shape = var_high_res_adjusted_dict[key].shape

            print('low_res_shape', low_res_shape, type(low_res_shape))
            print('high_res_shape', high_res_shape, type(high_res_shape))

            # var_low_res_adjusted_dict[key] and var_high_adjusted_res_dict[key] should be divisible for lon & lat dimension (1 & 2 in python)
            residue_geo = []
            for i in range(len(low_res_shape)):
                residue_geo.append(high_res_shape[i] % low_res_shape[i])
                if residue_geo[i] != 0:
                    print('Not divisible:', high_res_shape[i], low_res_shape[i], i, residue_geo[i])
                    if i == 2 and residue_geo[i] == 2:
                        var_high_res_adjusted_dict[key] = var_high_res_adjusted_dict[key][:, :, 1:-1, :] # remove the 1st and last elements

            # the size of the time (1st) dimension should be divisible by batch_size = 50 
            residue_time_low_res = low_res_shape[0] % batch_size
            residue_time_high_res = high_res_shape[0] % batch_size
            if residue_time_high_res != 0 :
                var_high_res_adjusted_dict[key] = var_high_res_adjusted_dict[key][:-residue_time_high_res, :, :, :] # remove the last few elements
            if residue_time_low_res != 0:
                var_low_res_adjusted_dict[key] = var_low_res_adjusted_dict[key][:-residue_time_low_res, :, :, :] # remove the last few elements

        return var_low_res_adjusted_dict, var_high_res_adjusted_dict


    def scale_var(self, var_before_scale):

        """
        Scale data to same magnitude

        :param var_before_scale: input variable to scale 
        :type var_before_scale: array
        :return: scaled variable, var_scaled
        :rtype: array
        """

        scaler = MinMaxScaler()
        scaler, var_before_scale = self.scale(scaler, var_before_scale)
        print('shape var_before_scale', var_before_scale.shape)
        var_scaled = np.expand_dims(var_before_scale, axis=3)
        print('var_scaled', var_scaled.shape)

        #scalers_list = []
        #scalers_list.append(scaler)
        #var_t = var.transpose((1, 2, 3, 0))[:-12] # -12, why?
        #varin[i] = var_scaled
        #varout = varin[i].transpose((1, 2, 3, 0))[:-12] # -12, why?
        #varout = varout[:, :, 1:-1]  # for 3km, why?

        return var_scaled


    def filter_var(self, var_low_res, var_high_res):

        """
        Filter data from high resolution (hr, eg, 3 km) to low resolution (lr, eg, 12 km)

        :param var_low_res: variable at low resolution 
        :type var_low_res: array
        :param var_high_res: variable at high resolution 
        :type var_high_res: array
        :return: var_low_res_gen, generated variable at low resolution from high resolution variable
        :rtype: array
        """

        Pool_Size = 4 
        max_pool_2d = layers.MaxPooling2D(pool_size = Pool_Size, padding = 'valid')
        var_low_res_gen = max_pool_2d(var_high_res)
        var_low_res_gen = var_low_res_gen.numpy()

        print('var_low_res_gen:', var_low_res_gen.shape)
        print('var_low_res:', var_low_res.shape)
        print('var_high_res:', var_high_res.shape)

        return var_low_res_gen


    def split_data(self, var_lr_gen, var_hr, batch_size, TEST_SIZE, RANDOM_STATE, variable):

        """
        Split dataset into subsets 

        :param var_low_res: variable at low resolution 
        :type var_low_res: array
        :param var_high_res: variable at high resolution 
        :type var_high_res: array
        :return: var_low_res_gen, generated variable at low resolution from high resolution variable
        :rtype: array
        """

        #print('var_lr_gen, var_hr:', var_lr_gen.shape, var_hr.shape)
        X_train, X_test, y_train, y_test = train_test_split(var_lr_gen[variable], var_hr[variable], test_size = TEST_SIZE, random_state = RANDOM_STATE)

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

