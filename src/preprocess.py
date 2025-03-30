
import tensorflow as tf
#from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from matplotlib import pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter

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


    def inverse(self, scaler, var, var_ref_to_fit):

        """
        Scale data from (0-1) back to real values

        :param scaler: 
        :type scaler: 
        :param var: The variable to scale
        :type var: array
        :return: scaled data, var
        :rtype: array
        """

        shape = var.shape
        var = var.reshape((shape[0], -1))
        scale.fit(var_ref_to_fit)
        var = scaler.inverse_transform(var)
        var = var.reshape(shape)

        return scaler, var


    def scale_dict(self, var_dict_in):

        """
        Scale data to same magnitude 

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


    def inverse_dict(self, var_dict_in, var_ref):

        """
        Inverse scaled data (0-1) to real data

        :param var: The variable to scale
        :type var: array
        :return: scaled data, var
        :rtype: array
        """

        var_dict_out = {}
        for key, values in var_dict_in.items():
            var_dict_out[key] = self.inverse_var(values, var_ref[key])
            print('inverse key:', key)

        return var_dict_out


    def scale_const_dict(self, var_dict_in):

        """
        Scale data to same magnitude 

        :param var: The variable to scale
        :type var: array
        :return: scaled data, var
        :rtype: array
        """

        var_dict_out = {}
        for key, values in var_dict_in.items():
            #print(type(values), values.shape)
            var_dict_out[key] = np.divide(values, np.nanmax(values))
            print('scale key:', key)

        return var_dict_out


    def inverse_const_dict(self, var_dict_in, var_array_ref):

        """
        Scale data to same magnitude 

        :param var: The variable to scale
        :type var: array
        :return: scaled data, var
        :rtype: array
        """

        var_dict_out = {}
        for key, values in var_dict_in.items():
            var_dict_out[key] = values * np.nanmax(var_array_ref)
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


    def adjust_data_size(self, var_low_res_dict, var_const_high_res_dict, var_high_res_dict, batch_size):

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
        var_low_res_adjusted_dict = var_low_res_dict.copy()
        var_const_high_res_adjusted_dict = var_const_high_res_dict.copy()
        var_high_res_adjusted_dict = var_high_res_dict.copy()

        common_var = list(set(var_low_res_adjusted_dict).intersection(var_high_res_adjusted_dict))
        print('common_var:', common_var)
        #for ivar in common_var:
        for key, values in var_high_res_adjusted_dict.items():

            high_res_shape = var_high_res_adjusted_dict[key].shape
            residue_time_high_res = high_res_shape[0] % batch_size
            if residue_time_high_res != 0 :
                #if key in var_high_res_adjusted_dict:
                for key_hr, values_hr in var_high_res_adjusted_dict.items():
                    print('key in var_high_res_adjusted_dict', key_hr)
                    var_high_res_adjusted_dict[key_hr] = var_high_res_adjusted_dict[key_hr][:-residue_time_high_res, ] # remove the last few elements
            print('residue_time_high_res', residue_time_high_res)


        residue_time_const_high_res = 0
        for key, values in var_const_high_res_adjusted_dict.items():

            high_res_shape = var_const_high_res_adjusted_dict[key].shape
            residue_time_const_high_res = high_res_shape[0] % batch_size
            if residue_time_const_high_res != 0 :
                #if key in var_high_res_adjusted_dict:
                for key_hr, values_hr in var_const_high_res_adjusted_dict.items():
                    print('key in var_high_res_adjusted_dict', key_hr)
                    var_const_high_res_adjusted_dict[key_hr] = var_const_high_res_adjusted_dict[key_hr][:-residue_time_const_high_res, ] # remove the last few elements
            print('residue_time_const_high_res', residue_time_const_high_res)

        ikey = 0                       
        for key, values in var_low_res_adjusted_dict.items():

            low_res_shape = var_low_res_adjusted_dict[key].shape

            print('key', key)
            #print('low_res_shape', low_res_shape, type(low_res_shape))
            #print('high_res_shape', high_res_shape, type(high_res_shape))

            # var_low_res_adjusted_dict[key] and var_high_adjusted_res_dict[key] should be divisible for lon & lat dimension (1 & 2 in python)
            #if key in var_high_res_adjusted_dict:
            for key_hr, values_hr in var_high_res_adjusted_dict.items():
                residue_geo = []
                for i in range(len(low_res_shape)):
                    residue_geo.append(high_res_shape[i] % low_res_shape[i])
                    if residue_geo[i] != 0:
                        print('Not divisible:', high_res_shape[i], low_res_shape[i], i, residue_geo[i])
                        if ikey == 0 and i == 2 and residue_geo[i] == 2:
                            var_high_res_adjusted_dict[key_hr] = var_high_res_adjusted_dict[key_hr][:, :, 1:-1] # remove the 1st and last elements

            for key_hr, values_hr in var_const_high_res_adjusted_dict.items():
                residue_geo = []
                for i in range(len(low_res_shape)):
                    residue_geo.append(high_res_shape[i] % low_res_shape[i])
                    if residue_geo[i] != 0:
                        print('Not divisible:', high_res_shape[i], low_res_shape[i], i, residue_geo[i])
                        if ikey == 0 and i == 2 and residue_geo[i] == 2:
                            var_const_high_res_adjusted_dict[key_hr] = var_const_high_res_adjusted_dict[key_hr][:, :, 1:-1] # remove the 1st and last elements
            residue_geo_dict = {'x':residue_geo[2], 'y':residue_geo[1]}

            # the size of the time (1st) dimension should be divisible by batch_size = 50 
            residue_time_low_res = low_res_shape[0] % batch_size
            #residue_time_high_res = high_res_shape[0] % batch_size
            #if residue_time_high_res != 0 :
            #    #if key in var_high_res_adjusted_dict:
            #    for key_hr, values_hr in var_high_res_adjusted_dict.items():
            #        print('key in var_high_res_adjusted_dict', key_hr)
            #        var_high_res_adjusted_dict[key_hr] = var_high_res_adjusted_dict[key_hr][:-residue_time_high_res, ] # remove the last few elements
            if residue_time_low_res != 0:
                var_low_res_adjusted_dict[key] = var_low_res_adjusted_dict[key][:-residue_time_low_res, ] # remove the last few elements
            print('residue_time_low_res', residue_time_low_res)
            ikey += 1
            
        return var_low_res_adjusted_dict, var_const_high_res_adjusted_dict, var_high_res_adjusted_dict, \
            residue_time_low_res, residue_time_const_high_res, residue_time_high_res, residue_geo_dict


    def scale_var(self, var_origin):

        """
        Scale data to same magnitude

        :param var_before_scale: input variable to scale 
        :type var_before_scale: array
        :return: scaled variable, var_scaled
        :rtype: array
        """

        shape = var_origin.shape
        var = var_origin.reshape((shape[0], -1))

        scaler = MinMaxScaler()
        var_scaled = scaler.fit_transform(var)
        var_scaled = var_scaled.reshape(shape)

        #scaler, var_before_scale = self.scale(scaler, var_before_scale)
        print('var_origin', var_origin.shape, np.nanmin(var_origin), np.nanmax(var_origin))
        #var_scaled = np.expand_dims(var_before_scale, axis=3)
        #var_scaled = np.copy(var_before_scale)
        print('var_scaled', var_scaled.shape, np.nanmin(var_scaled), np.nanmax(var_scaled))

        #scalers_list = []
        #scalers_list.append(scaler)
        #var_t = var.transpose((1, 2, 3, 0))[:-12] # -12, why?
        #varin[i] = var_scaled
        #varout = varin[i].transpose((1, 2, 3, 0))[:-12] # -12, why?
        #varout = varout[:, :, 1:-1]  # for 3km, why?

        return var_scaled


    def inverse_var(self, var_origin, var_ref_to_fit):

        """
        inverse scaled data (0-1) to real data

        :param var_before_inverse: input variable to inverse 
        :type var_before_inverse: array
        :return: inversed variable, var_inverse
        :rtype: array
        """
        
        shape = var_origin.shape
        var = var_origin.reshape((shape[0], -1))
        shape_ref = var_ref_to_fit.shape
        var_ref_reshape = var_ref_to_fit.reshape((shape_ref[0], -1))
        print('shape, shape_ref', shape, shape_ref)

        scaler = MinMaxScaler()
        scaler.fit(var_ref_reshape)
        var_inverse = scaler.inverse_transform(var)

        var_inverse = var_inverse.reshape(shape)

        #scaler, var_inverse = self.inverse(scaler, var_before_inverse)
        print('var_before_inverse', var_origin.shape, np.nanmin(var_origin), np.nanmax(var_origin))
        print('var_inverse', var_inverse.shape, np.nanmin(var_inverse), np.nanmax(var_inverse))

        return var_inverse

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
        max_pool_2d = tf.keras.layers.MaxPooling2D(pool_size = Pool_Size, padding = 'valid')
        if var_high_res.ndim == 3:
            var_high_res_4dim = np.expand_dims(var_high_res, axis=3)
        var_low_res_gen = max_pool_2d(var_high_res_4dim)
        var_low_res_gen = var_low_res_gen.numpy()
        print('var_high_res_4dim:', var_high_res_4dim.shape)
        del var_high_res_4dim

        if var_low_res_gen.ndim == 4 and var_low_res_gen.shape[-1] == 1:
            var_low_res_gen = var_low_res_gen[:, :, :, 0]

        print('var_low_res_gen:', var_low_res_gen.shape)
        print('var_low_res:', var_low_res.shape)
        print('var_high_res:', var_high_res.shape)

        return var_low_res_gen

    def augment(self, data_x_const, data_y):

        # Function to apply the augmentation to low-resolution and high-resolution images
        #Apply Data Augmentation
        data_augmentation = tf.keras.Sequential([
            tf.keras.layers.RandomFlip("horizontal"),  # Random horizontal flip
            tf.keras.layers.RandomRotation(0.3),      # Small random rotation
            tf.keras.layers.RandomZoom(0.3),          # Random zoom
            tf.keras.layers.RandomContrast(0.3),          # Random contrast
        ])

        data_x, data_const = data_x_const
        data_x = data_augmentation(data_x)
        #data_const = data_augmentation(data_const)
        data_y = data_augmentation(data_y)
        return (data_x, data_const), data_y


    def split_data(self, var_lr, var_const_hr, var_hr, batch_size, TEST_SIZE, VALIDATION_SPLIT, RANDOM_STATE, DATA_AUGMENTATION, \
            variable, downscale_mode):

        """
        Split dataset into subsets 

        :param var_low_res: variable at low resolution 
        :type var_low_res: array
        :param var_high_res: variable at high resolution 
        :type var_high_res: array
        :return: var_low_res_gen, generated variable at low resolution from high resolution variable
        :rtype: array
        """

        if downscale_mode == 'upscale':
            #print('var_lr, var_hr:', var_lr.shape, var_hr.shape)
            
            if var_lr[variable].ndim == 3:
                var_lr_4dim = np.expand_dims(var_lr[variable], axis=3)
            if var_hr[variable].ndim == 3:
                var_hr_4dim = np.expand_dims(var_hr[variable], axis=3)

            #X_train, X_test, y_train, y_test = train_test_split(var_lr[variable], var_hr[variable], test_size = TEST_SIZE, random_state = RANDOM_STATE)
            X_train, X_test, y_train, y_test = train_test_split(var_lr_4dim, var_hr_4dim, test_size = TEST_SIZE, random_state = RANDOM_STATE)

        elif downscale_mode == 'direct':
            # https://stackoverflow.com/questions/70953355/create-a-tensorflow-dataset-based-on-a-multi-input
            # https://stackoverflow.com/questions/49829023/train-test-split-with-multiple-features
            #X_train, y_train, X_test, y_test = (), (), (), ()

            """
            for key, values in var_lr.items():
                x_tr, x_te = train_test_split(var_lr[key], test_size = TEST_SIZE, random_state = RANDOM_STATE)
                X_train = X_train + (x_tr, )
                X_test = X_test + (x_te, )
                print('key lr', key)
                print('x_tr', type(x_tr), len(x_tr))
                print('x_te', len(x_te))
            for key, values in var_hr.items():
                y_tr, y_te = train_test_split(var_hr[key], test_size = TEST_SIZE, random_state = RANDOM_STATE)
                y_train  = y_train + (y_tr, )
                y_test  = y_test + (y_te, )
                print('key hr', key)
                print('y_tr', type(y_tr), len(y_tr))
                print('y_te', len(y_te))
            """
            x_tr, x_te, y_tr, y_te = [None]*len(var_lr), [None]*len(var_lr), [None]*len(var_hr), [None]*len(var_hr)
            const_tr, const_te = [None]*len(var_const_hr), [None]*len(var_const_hr)
            i = 0
            for key, values in var_lr.items():
                x_tr_arr, x_te_arr = train_test_split(var_lr[key], test_size = TEST_SIZE, random_state = RANDOM_STATE, shuffle = False)
                x_tr[i], x_te[i] = x_tr_arr, x_te_arr
                print('key lr', key, i, len(var_lr))
                #print('x_tr', type(x_tr[i]), np.shape(x_tr[i]))
                #print('x_te', np.shape(x_te[i]))
                i += 1
            X_train = np.stack(x_tr, axis = 3)
            X_test  = np.stack(x_te, axis = 3)
            i = 0
            for key, values in var_hr.items():
                y_tr_arr, y_te_arr = train_test_split(var_hr[key], test_size = TEST_SIZE, random_state = RANDOM_STATE, shuffle = False)
                y_tr[i], y_te[i] = y_tr_arr, y_te_arr
                print('key hr', key, i, len(var_hr))
                print('y_tr', type(y_tr[i]), np.shape(y_tr[i]))
                print('y_te', np.shape(y_te[i]))
                i += 1
            y_train = np.stack(y_tr, axis = 3)
            y_test  = np.stack(y_te, axis = 3)
            i = 0
            for key, values in var_const_hr.items():
                const_tr_arr, const_te_arr = train_test_split(var_const_hr[key], test_size = TEST_SIZE, random_state = RANDOM_STATE, shuffle = False)
                const_tr[i], const_te[i] = const_tr_arr, const_te_arr
                print('key const_hr', key, i, len(var_const_hr))
                print('const_tr', type(const_tr[i]), np.shape(const_tr[i]))
                print('const_te', np.shape(const_te[i]))
                i += 1
            if var_const_hr:
                const_train = np.stack(const_tr, axis = 3)
                const_test  = np.stack(const_te, axis = 3)
            else:
                const_train = np.empty_like(y_train)
                const_test  = np.empty_like(y_test)

        #X_train = np.asarray(X_train)
        #y_train = np.asarray(y_train)
        #X_test = np.asarray(X_test)
        #y_test = np.asarray(y_test)
        print('Training dataset shape:', X_train.shape, const_train.shape, y_train.shape)
        print('Testing dataset shape:', X_test.shape, const_test.shape, y_test.shape)

        #if var_const_hr:
        dataset_train_all = tf.data.Dataset.from_tensor_slices(((X_train, const_train), y_train))
        dataset_test = tf.data.Dataset.from_tensor_slices(((X_test, const_test), y_test))
        #else: 
        #    dataset_train_all = tf.data.Dataset.from_tensor_slices((X_train, y_train))
        #    dataset_test = tf.data.Dataset.from_tensor_slices((X_test, y_test))
        print('dataset_train_all 0 shape:', dataset_train_all.element_spec)
        print('dataset_test 0 shape:', dataset_test.element_spec)


        if DATA_AUGMENTATION:
            # Apply the augmentations to the dataset before batching
            dataset_train_all = dataset_train_all.map(self.augment, num_parallel_calls=tf.data.AUTOTUNE)
            dataset_test = dataset_test.map(self.augment, num_parallel_calls=tf.data.AUTOTUNE)
            #dataset_train_all = dataset_train_all.map(lambda x: self.augment(x), num_parallel_calls=tf.data.AUTOTUNE)
            #dataset_test = dataset_test.map(lambda x: self.augment(x), num_parallel_calls=tf.data.AUTOTUNE)

        # Define the validation split fraction
        val_size = int(len(dataset_train_all) * VALIDATION_SPLIT)
        print('val_size', val_size)

        # Split dataset into training and validation
        dataset_train = dataset_train_all.skip(val_size)
        dataset_valid = dataset_train_all.take(val_size)


        print('dataset_train_all type:', type(dataset_train_all))
        print('dataset_test type:', type(dataset_test))
        print('dataset_valid type:', type(dataset_valid))
        print('dataset_train_all shape:', dataset_train_all.element_spec)
        print('dataset_test shape:', dataset_test.element_spec)
        print('dataset_valid shape:', dataset_valid.element_spec)
        print('dataset_train_all len:', np.shape(dataset_train_all.element_spec[0]), np.shape(dataset_train_all.element_spec[0])[0], np.shape(dataset_train_all.element_spec[1]))
        print('dataset_test len:', np.shape(dataset_test.element_spec[0]), np.shape(dataset_test.element_spec[1])[0], np.shape(dataset_test.element_spec[1]))
        print('dataset_valid len:', np.shape(dataset_valid.element_spec[0]), np.shape(dataset_valid.element_spec[1])[0], np.shape(dataset_valid.element_spec[1]))

        dataset_train = dataset_train.batch(batch_size, drop_remainder=True)
        dataset_test = dataset_test.batch(batch_size, drop_remainder=True)
        dataset_valid = dataset_valid.batch(batch_size, drop_remainder=True)

        """
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
        """

        return dataset_train, dataset_valid, dataset_test, X_train, X_test, const_train, const_test, y_train, y_test
        #return dataset_train, dataset_test, X_trainr, X_testr, y_trainr, y_testr




    def fill_missing_with_interpolation(self, data):

        mask = np.isnan(data)
        data_filled = gaussian_filter(np.nan_to_num(data), sigma=1)  # Smooth out missing values
        data[mask] = data_filled[mask]  # Replace only missing values

        return data


    def fill_missing_with_interpolation_dict(self, var_in_dict):

        """
        Filter data from high resolution (hr, eg, 3 km) to low resolution (lr, eg, 12 km)

        :param var_low_res_dict: variable at low resolution 
        :type var_low_res_dict: dictionary
        :param var_high_res_dict: variable at high resolution 
        :type var_high_res_dict: dictionary
        :return: var_low_res_gen_dict, generated variable at low resolution from high resolution variable
        :rtype: dictionary
        """

        var_out_dict = {}

        for key, values in var_in_dict.items():
            var_out_dict[key] = self.fill_missing_with_interpolation(var_in_dict[key])


        return var_out_dict
