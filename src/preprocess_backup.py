
import tensorflow as tf
#from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import PowerTransformer
from matplotlib import pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter
from joblib import dump, load
import psutil, os, gc

def log_mem(tag):
    mem = psutil.Process(os.getpid()).memory_info().rss / 1e9
    print(f"[MEM] {tag}: {mem:.2f} GB", flush=True)

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
        self.scale.fit(var_ref_to_fit)
        var = scaler.inverse_transform(var)
        var = var.reshape(shape)

        return scaler, var

    def scale_dict(self, var_dict_in, scaler_dict, method, predonly, scaler_path, resolution):
        
        """
        Scale data to same magnitude 

        :param var: The variable to scale
        :type var: array
        :return: scaled data, var
        :rtype: array
        """
        """

        var_dict_out = {}
        for key, values in var_dict_in.items():
            #print(type(values), values.shape)
            var_dict_out[key] = self.scale_var(values, scaler_dict, method, key, predonly, scaler_path, resolution)
            #print('scale key:', key)
            del values

        return var_dict_out
        """

        var_dict_out = {}
        keys = list(var_dict_in.keys())
        for key in keys:
            log_mem(f"before scaling {key}")
            values = var_dict_in[key]
            #values = var_dict_in.pop(key)
            #filepath = self.scale_var_chunked(values, scaler_dict, method, key, predonly, scaler_path, resolution, chunk_size=2000)
            filepath = self.scale_var(values, scaler_dict, method, key, predonly, scaler_path, resolution)
            del values
            gc.collect()
            var_dict_out[key] = self.load_scaled_var(filepath, mmap_mode='r')  # memory-mapped array, not loaded into RAM
            log_mem(f"after scaling {key}")

        return var_dict_out


    def inverse_dict(self, var_dict_in, var_ref, scaler_dict, scaler_path, resolution):

        """
        Inverse scaled data (0-1) to real data

        :param var: The variable to scale
        :type var: array
        :return: scaled data, var
        :rtype: array
        """

        var_dict_out = {}
        print('var_ref.keys:', var_ref.keys())
        for key, values in var_dict_in.items():
            var_dict_out[key] = self.inverse_var(values, var_ref[key], key, scaler_dict, scaler_path, resolution)
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

        """
        var_low_res_adjusted_dict = var_low_res_dict.copy()
        var_const_high_res_adjusted_dict = var_const_high_res_dict.copy()
        var_high_res_adjusted_dict = var_high_res_dict.copy()
        residue_time_high_res, residue_time_const_high_res, residue_time_low_res = 0, 0, 0

        common_var = list(set(var_low_res_adjusted_dict).intersection(var_high_res_adjusted_dict))
        print('common_var:', common_var)
        #for ivar in common_var:
        for key, values in var_high_res_adjusted_dict.items():

            high_res_shape = var_high_res_adjusted_dict[key].shape
            #residue_time_high_res = high_res_shape[0] % batch_size
            #if residue_time_high_res != 0 :
            #    var_high_res_adjusted_dict[key] = var_high_res_adjusted_dict[key][:-residue_time_high_res, ] # remove the last few elements
            print('residue_time_high_res', residue_time_high_res)


        for key, values in var_const_high_res_adjusted_dict.items():

            high_res_shape = var_const_high_res_adjusted_dict[key].shape
            #residue_time_const_high_res = high_res_shape[0] % batch_size
            #if residue_time_const_high_res != 0 :
            #    var_const_high_res_adjusted_dict[key] = var_const_high_res_adjusted_dict[key][:-residue_time_const_high_res, ] # remove the last few elements
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
            #residue_time_low_res = low_res_shape[0] % batch_size
            #if residue_time_low_res != 0:
            #    var_low_res_adjusted_dict[key] = var_low_res_adjusted_dict[key][:-residue_time_low_res, ] # remove the last few elements
            print('residue_time_low_res', residue_time_low_res)
            ikey += 1
            
        return var_low_res_adjusted_dict, var_const_high_res_adjusted_dict, var_high_res_adjusted_dict, \
            residue_time_low_res, residue_time_const_high_res, residue_time_high_res, residue_geo_dict


    def get_scaler(self, var_name, scaler_var):

        if scaler_var == 'PowerTransformer': 
            scaler = PowerTransformer(method='yeo-johnson')
        elif scaler_var == 'MinMaxScaler': 
            scaler = MinMaxScaler() #feature_range=(-1, 1))
        elif scaler_var == 'StandardScaler': 
            scaler = StandardScaler()
        elif scaler_var == 'RobustScaler': 
            scaler = RobustScaler()
        print('var_name, scaler_var', var_name, scaler_var)

        return scaler

    def scale_var_chunked(self, var_origin, scaler_dict, method, var_name, predonly, scaler_path, resolution, chunk_size=2000):
        shape = var_origin.shape
        scaler_var = scaler_dict[var_name]

        if method == 'scale_over_time':
            log_mem(f"in scale_var_chunked before scalier.fit {var_name}")
            if not predonly:
                # Fit scaler on full data first (StandardScaler.fit needs to see all data, but fit() is cheaper than transform's output copy)
                var_flat = var_origin.reshape((shape[0], -1))
                scaler = self.get_scaler(var_name, scaler_var)
                scaler.fit(var_flat)
                dump(scaler, f"{scaler_path}/{scaler_var}_{resolution}_{var_name}.joblib")
                del var_flat
                gc.collect()
            else:
                scaler = load(f"{scaler_path}/{scaler_var}_{resolution}_{var_name}.joblib")
            log_mem(f"in scale_var_chunked aftwr scalier.fit {var_name}")

            out_path = os.path.join(scaler_path, f"{var_name}_{resolution}_scaled.npy")

            # Create the output file on disk first, sized correctly, without holding it in RAM
            var_scaled_mmap = np.lib.format.open_memmap(out_path, mode='w+', dtype=var_origin.dtype, shape=shape)

            log_mem(f"in scale_var_chunked before scalier.transform {var_name}")
            # Transform in chunks along the time axis
            for start in range(0, shape[0], chunk_size):
                end = min(start + chunk_size, shape[0])
                chunk = var_origin[start:end].reshape((end - start, -1))
                chunk_scaled = scaler.transform(chunk)
                var_scaled_mmap[start:end] = chunk_scaled.reshape((end - start,) + shape[1:])
                del chunk, chunk_scaled
            log_mem(f"in scale_var_chunked after scalier.transform {var_name}")

            var_scaled_mmap.flush()
            del var_scaled_mmap
            gc.collect()
            return out_path


    def scale_var(self, var_origin, scaler_dict, method, var_name, predonly, scaler_path, resolution):

        """
        Scale data (along time axis) to same magnitude, first along time axis, if all 0, scale it along space

        :param var_before_scale: input variable to scale 
        :type var_before_scale: array
        :return: scaled variable, var_scaled
        :rtype: array
        """

        missing_value = 1e20
        shape = var_origin.shape
        var = var_origin.reshape((shape[0], -1))
        #var[var >= missing_value] = np.nan #0.0
        #if var_name == 'mrsol':
        #    var[var == 0] = missing_value
        #    print('mrsol = 0 set to missing value:', missing_value)
        #if var_name == 'mrsol':
            #epsilon = 1e-10
            #var = np.log(var + epsilon)

        scaler_var = scaler_dict[var_name] 

        if method == 'scale_over_time':
            if not predonly:
                scaler = self.get_scaler(var_name, scaler_var)
                scaler.fit(var)
                dump(scaler, f"{scaler_path}/{scaler_var}_{resolution}_{var_name}.joblib")
            else:
                scaler = load(f"{scaler_path}/{scaler_var}_{resolution}_{var_name}.joblib")
            var_scaled = scaler.transform(var)
            #var_scaled = scaler.fit_transform(var)
            var_scaled = var_scaled.reshape(shape)

        elif method == 'scale_over_space':
        #all_equal = np.all(var_scaled == var_scaled[0])
        #if all_equal:
            #mask = var < threshold 
            var_T = np.transpose(var)
            scaler = self.get_scaler(var_name, scaler_var)
            var_scaled_T = scaler.fit_transform(var_T)
            var_scaled = np.transpose(var_scaled_T)
            var_scaled = var_scaled.reshape(shape)

        if var_name == 'mrsol':
            invalid = (var_origin <= 0) | (var_origin > 1e10)
            var_scaled[invalid] = 0.0

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

        del var  # free the reshaped copy
        gc.collect()

        if scaler_path is not None:
            os.makedirs(scaler_path, exist_ok=True)
            out_path = os.path.join(scaler_path, f"{var_name}_{resolution}_scaled.npy")
            np.save(out_path, var_scaled)
            del var_scaled
            gc.collect()
            return out_path   # return path instead of array

        return var_scaled

    def load_scaled_var(self, filepath, mmap_mode='r'):
        """
        Helper to lazily load a scaled variable from disk.
        mmap_mode='r' memory-maps the file instead of loading it fully into RAM —
        only the slices you actually index will be read from disk.
        """
        return np.load(filepath, mmap_mode=mmap_mode)

    """
    def scale_var(self, var_origin, method):

        #Scale data (along time axis) to same magnitude, first along time axis, if all 0, scale it along space

        #:param var_before_scale: input variable to scale
        #:type var_before_scale: array
        #:return: scaled variable, var_scaled
        #:rtype: array

        # --- Constants ---
        # Define the threshold for treating a value as "missing" or an outlier
        MISSING_VALUE_THRESHOLD = 1e10

        shape = var_origin.shape
        # Reshape to (time_steps, total_pixels)
        var = var_origin.reshape((shape[0], -1))
        var_scaled = np.zeros_like(var) # Initialize scaled array

        if method == 'scale_over_time':
            # Scaling along the time axis (for each pixel)
            scaler = MinMaxScaler()
            var_scaled = scaler.fit_transform(var)
            var_scaled = var_scaled.reshape(shape)

        elif method == 'scale_over_space':
            # Scaling along the space axis (for each time step)

            # --- Missing Value Handling ---
            # 1. Create a mask: True for valid data (less than threshold)
            mask = var < MISSING_VALUE_THRESHOLD

            # 2. Transpose for scaling over space (total_pixels, time_steps)
            var_T = np.transpose(var)

            # 3. Initialize the scaled result array
            var_scaled_T = np.zeros_like(var_T)

            # Iterate through each time step (row in var_T)
            for i in range(var_T.shape[0]):
                time_step_data = var_T[i, :] # Data for one time step across all space
                time_step_mask = mask[i, :]   # Mask for one time step

                valid_data = time_step_data[time_step_mask]

                if valid_data.size > 0:
                    scaler = MinMaxScaler()
                    # Reshape for scaler (num_valid_pixels, 1)
                    valid_data_scaled = scaler.fit_transform(valid_data.reshape(-1, 1))

                    # Put the scaled values back into the correct positions
                    scaled_step = np.zeros_like(time_step_data, dtype=float)
                    scaled_step[time_step_mask] = valid_data_scaled.flatten()

                    # Replace non-valid data with its original value
                    scaled_step[~time_step_mask] = time_step_data[~time_step_mask]

                    var_scaled_T[i, :] = scaled_step
                else:
                    # If all data is missing/outlier, keep original values
                    var_scaled_T[i, :] = time_step_data

            var_scaled = np.transpose(var_scaled_T)
            var_scaled = var_scaled.reshape(shape)

        elif method == 'global_scaling':
            # --- New Option: Scaling over all dimensions (time and space) ---

            # 1. Create a mask for valid data across all pixels
            mask_flat = var.flatten() < MISSING_VALUE_THRESHOLD

            # 2. Extract valid data for global scaling
            valid_data_flat = var.flatten()[mask_flat]

            if valid_data_flat.size > 0:
                scaler = MinMaxScaler()
                # Reshape for scaler (num_valid_pixels, 1)
                valid_data_scaled = scaler.fit_transform(valid_data_flat.reshape(-1, 1))

                # 3. Put the scaled values back into the correct positions
                var_scaled_flat = np.zeros_like(var.flatten(), dtype=float)
                var_scaled_flat[mask_flat] = valid_data_scaled.flatten()

                # 4. Replace non-valid data with its original value
                var_scaled_flat[~mask_flat] = var.flatten()[~mask_flat]

                var_scaled = var_scaled_flat.reshape(shape)
            else:
                # If all data is missing/outlier, keep original values
                var_scaled = var_origin

        else:
            print(f"Error: Unknown scaling method: {method}")
            return var_origin # Return original array on error

        # --- Logging/Return (Original End of Function) ---
        print('var_origin', var_origin.shape, np.nanmin(var_origin), np.nanmax(var_origin))
        print('var_scaled', var_scaled.shape, np.nanmin(var_scaled), np.nanmax(var_scaled))

        return var_scaled
    """


    def scale_var_wrong(self, var_origin, method):

        """
        Scale data using MinMaxScaler.

        method options:
            - 'scale_over_time'
            - 'scale_over_space'
            - 'global_scaling' (new)

        Missing values (>1e10) are treated as NaN and left as 0 after scaling.
        """

        shape = var_origin.shape
        var = var_origin.copy().reshape((shape[0], -1))

        # Convert missing values to NaN
        var = np.where(var > 1e10, np.nan, var)

        if method == 'scale_over_time':

            scaler = MinMaxScaler()
            var_scaled = np.zeros_like(var) #, dtype=float)

            for i in range(var.shape[1]):          # loop over grid cells
                col = var[:, i]
                mask = ~np.isnan(col)
                if np.any(mask):
                    var_scaled[mask, i] = scaler.fit_transform(col[mask, None]).ravel()
                else:
                    var_scaled[:, i] = 0           # no valid data

            var_scaled = var_scaled.reshape(shape)

        elif method == 'scale_over_space':

            scaler = MinMaxScaler()
            var_T = var.T                          # shape: (space, time)
            var_scaled_T = np.zeros_like(var_T) #, dtype=float)

            for i in range(var_T.shape[0]):        # loop over each grid cell across time
                row = var_T[i]
                mask = ~np.isnan(row)
                if np.any(mask):
                    var_scaled_T[i, mask] = scaler.fit_transform(row[mask, None]).ravel()
                else:
                    var_scaled_T[i] = 0            # all missing

            var_scaled = var_scaled_T.T.reshape(shape)

        elif method == 'scale_global':

            scaler = MinMaxScaler()
            flat = var.flatten()
            mask = ~np.isnan(flat)

            flat_scaled = np.zeros_like(flat) #, dtype=float)
            flat_scaled[mask] = scaler.fit_transform(flat[mask, None]).ravel()

            var_scaled = flat_scaled.reshape(shape)

        else:
            raise ValueError(f"Unknown scaling method: {method}")

        print('var_origin', var_origin.shape, np.nanmin(var_origin), np.nanmax(var_origin))
        print('var_scaled', var_scaled.shape, np.nanmin(var_scaled), np.nanmax(var_scaled))

        return var_scaled



    def inverse_var(self, var_origin, var_ref_to_fit, var_name, scaler_dict, scaler_path, resolution):

        """
        inverse scaled data (0-1) to real data

        :param var_before_inverse: input variable to inverse 
        :type var_before_inverse: array
        :return: inversed variable, var_inverse
        :rtype: array
        """

        missing_value = 1e20 
        shape = var_origin.shape
        var = var_origin.reshape((shape[0], -1))
        shape_ref = var_ref_to_fit.shape
        var_ref_reshape = var_ref_to_fit.reshape((shape_ref[0], -1))
        print('shape, shape_ref', shape, shape_ref)
        scaler_var = scaler_dict[var_name] 

        #scaler = MinMaxScaler()
        #scaler.fit(var_ref_reshape)
        scaler = load(f"{scaler_path}/{scaler_var}_{resolution}_{var_name}.joblib")
        var_inverse = scaler.inverse_transform(var)

        #if var_name == 'mrsol':
        #    epsilon = 1e-10
        #    var_inverse = np.exp(var_inverse)
        #    var_inverse = var_inverse - epsilon
        #    var_inverse = np.maximum(var_inverse, 0)

        var_inverse = var_inverse.reshape(shape)
        #if var_name == 'mrsol':
        #    var_inverse[var_inverse == 0] = np.nan
        #    print('mrsol = 0 set to nan')

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


    def random_tile_sample_v1(self, inputs, target, tile_size_lr, tile_size_hr):
        """
        Randomly crop a spatial tile from (lr, const, hr) triple.
     
        Args:
            inputs:      tuple of (lr_sample, const_sample), shapes (H_lr, W_lr, C_lr) and (H_hr, W_hr, C_const)
            target:      hr_sample, shape (H_hr, W_hr, C_hr)
            tile_size_lr: spatial tile size in LR space (e.g. 64)
            tile_size_hr: spatial tile size in HR space (= tile_size_lr * upscale_factor)
     
        Returns:
            ((lr_tile, const_tile), hr_tile)
        """
        lr_sample, const_sample = inputs
     
        lr_h = tf.shape(lr_sample)[0]
        lr_w = tf.shape(lr_sample)[1]
     
        # Random top-left corner in LR space
        max_y = lr_h - tile_size_lr
        max_x = lr_w - tile_size_lr
        y_lr = tf.random.uniform((), 0, max_y, dtype=tf.int32)
        x_lr = tf.random.uniform((), 0, max_x, dtype=tf.int32)
     
        # Corresponding top-left corner in HR space
        y_hr = y_lr * (tile_size_hr // tile_size_lr)
        x_hr = x_lr * (tile_size_hr // tile_size_lr)
     
        # Crop LR
        lr_tile = lr_sample[y_lr:y_lr + tile_size_lr, x_lr:x_lr + tile_size_lr, :]
     
        # Crop const (HR resolution)
        const_tile = const_sample[y_hr:y_hr + tile_size_hr, x_hr:x_hr + tile_size_hr, :]
     
        # Crop HR target
        hr_tile = target[y_hr:y_hr + tile_size_hr, x_hr:x_hr + tile_size_hr, :]
     
        return (lr_tile, const_tile), hr_tile


    # <<< CHANGED v2 >>> Added scale_factor derived from SUBSAMPLING_LR, added shape assertions
    def random_tile_sample_v2(self, inputs, target, tile_size_lr, scale_factor):
        """
        Randomly crop a spatial tile from (lr, const, hr) triple.
        tile_size_hr is derived automatically as tile_size_lr * scale_factor.

        Args:
            inputs:       tuple of (lr_sample, const_sample)
                          lr_sample shape:    (H_lr, W_lr, C_lr)
                          const_sample shape: (H_hr, W_hr, C_const)
            target:       hr_sample, shape (H_hr, W_hr, C_hr)
            tile_size_lr: spatial tile size in LR space (e.g. 64)
            scale_factor: upsampling factor = SUBSAMPLING_LR (e.g. 4)
                          tile_size_hr = tile_size_lr * scale_factor computed here
        Returns:
            ((lr_tile, const_tile), hr_tile)
        """
        lr_sample, const_sample = inputs

        tile_size_hr = tile_size_lr * scale_factor   # <<< CHANGED v2 >>> derived, not passed in

        lr_h = tf.shape(lr_sample)[0]
        lr_w = tf.shape(lr_sample)[1]

        # Random top-left corner in LR space
        # Ensure max_y/max_x >= 1 to avoid invalid range
        max_y = tf.maximum(lr_h - tile_size_lr, 1)   # <<< CHANGED v2 >>> added tf.maximum guard
        max_x = tf.maximum(lr_w - tile_size_lr, 1)   # <<< CHANGED v2 >>> added tf.maximum guard
        y_lr = tf.random.uniform((), 0, max_y, dtype=tf.int32)
        x_lr = tf.random.uniform((), 0, max_x, dtype=tf.int32)

        # Corresponding top-left corner in HR space
        y_hr = y_lr * scale_factor
        x_hr = x_lr * scale_factor

        # Crop LR
        lr_tile = lr_sample[y_lr:y_lr + tile_size_lr, x_lr:x_lr + tile_size_lr, :]

        # Crop const (HR resolution)
        const_tile = const_sample[y_hr:y_hr + tile_size_hr, x_hr:x_hr + tile_size_hr, :]

        # Crop HR target
        hr_tile = target[y_hr:y_hr + tile_size_hr, x_hr:x_hr + tile_size_hr, :]

        return (lr_tile, const_tile), hr_tile
    # <<< END CHANGED v2 >>>


    # <<< CHANGED v3 >>> Support rectangular tiles with separate x/y tile sizes
    def random_tile_sample(self, inputs, target, tile_size_lr_y, tile_size_lr_x, scale_factor):
        """
        tile_size_lr_y: LR tile height (NY // scale_factor)
        tile_size_lr_x: LR tile width  (NX // scale_factor)
        """
        lr_sample, const_sample = inputs

        tile_size_hr_y = tile_size_lr_y * scale_factor   # = NY
        tile_size_hr_x = tile_size_lr_x * scale_factor   # = NX

        lr_h = tf.shape(lr_sample)[0]
        lr_w = tf.shape(lr_sample)[1]

        max_y = tf.maximum(lr_h - tile_size_lr_y, 1)
        max_x = tf.maximum(lr_w - tile_size_lr_x, 1)
        y_lr = tf.random.uniform((), 0, max_y, dtype=tf.int32)
        x_lr = tf.random.uniform((), 0, max_x, dtype=tf.int32)

        y_hr = y_lr * scale_factor
        x_hr = x_lr * scale_factor

        lr_tile    = lr_sample   [y_lr:y_lr + tile_size_lr_y, x_lr:x_lr + tile_size_lr_x, :]
        const_tile = const_sample[y_hr:y_hr + tile_size_hr_y, x_hr:x_hr + tile_size_hr_x, :]
        hr_tile    = target      [y_hr:y_hr + tile_size_hr_y, x_hr:x_hr + tile_size_hr_x, :]

        return (lr_tile, const_tile), hr_tile
    # <<< END CHANGED v3 >>>


    def split_data(self, var_lr, var_const_hr, var_hr, batch_size, TEST_SIZE, VALIDATION_SPLIT, RANDOM_STATE, DATA_AUGMENTATION, \
            variable, downscale_mode,  \
            #TILE_SIZE_LR=None,          # <<< CHANGED v2 >>> only LR tile size needed now
            NX=None,            # <<< CHANGED v3 >>> HR tile width  (e.g. 424)
            NY=None,            # <<< CHANGED v3 >>> HR tile height (e.g. 352)
            SUBSAMPLING_LR=1):          # <<< CHANGED v2 >>> added SUBSAMPLING_LR (= scale factor)
            #TILE_SIZE_LR=None, TILE_SIZE_HR=None):

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
            X_train, X_test, y_train, y_test = train_test_split(var_lr_4dim, var_hr_4dim, test_size = TEST_SIZE, random_state = RANDOM_STATE, shuffle = False)

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
            # Define pre-selected test indices
            test_idx = np.concatenate([
                #np.arange(start, end + 1)  # inclusive of 'end'
                np.arange(start, end)  
                for start, end in zip(TEST_SIZE['start_idx'], TEST_SIZE['end_idx'])
            ])

            x_tr, x_te, y_tr, y_te = [None]*len(var_lr), [None]*len(var_lr), [None]*len(var_hr), [None]*len(var_hr)
            const_tr, const_te = [None]*len(var_const_hr), [None]*len(var_const_hr)
            i = 0
            for key, values in var_lr.items():

                if isinstance(TEST_SIZE, int):
                    x_tr_arr, x_te_arr = train_test_split(var_lr[key], test_size = TEST_SIZE, random_state = RANDOM_STATE, shuffle = False)

                elif isinstance(TEST_SIZE, dict):

                    # Create a mask for test and train sets
                    all_indices = np.arange(len(var_lr[key]))
                    train_idx = np.setdiff1d(all_indices, test_idx)
                    # Split manually
                    #print ('len all_indices', len(all_indices))
                    #print ('train_idx', train_idx)
                    #print ('test_idx', test_idx)
                    #print ('len var_lr', len(var_lr[key]))
                    x_tr_arr, x_te_arr = var_lr[key][train_idx], var_lr[key][test_idx]

                x_tr[i], x_te[i] = x_tr_arr, x_te_arr
                print('key lr', key, i, len(var_lr))
                #print('x_tr', type(x_tr[i]), np.shape(x_tr[i]))
                #print('x_te', np.shape(x_te[i]))
                i += 1
            X_train = np.stack(x_tr, axis = 3)
            X_test  = np.stack(x_te, axis = 3)

            i = 0
            for key, values in var_hr.items():

                if isinstance(TEST_SIZE, int):
                    y_tr_arr, y_te_arr = train_test_split(var_hr[key], test_size = TEST_SIZE, random_state = RANDOM_STATE, shuffle = False)

                elif isinstance(TEST_SIZE, dict):
                    # Create a mask for test and train sets
                    all_indices = np.arange(len(var_hr[key]))
                    train_idx = np.setdiff1d(all_indices, test_idx)
                    # Split manually
                    y_tr_arr, y_te_arr = var_hr[key][train_idx], var_hr[key][test_idx]

                y_tr[i], y_te[i] = y_tr_arr, y_te_arr
                print('key hr', key, i, len(var_hr))
                print('y_tr', type(y_tr[i]), np.shape(y_tr[i]))
                print('y_te', np.shape(y_te[i]))
                i += 1
            y_train = np.stack(y_tr, axis = 3)
            y_test  = np.stack(y_te, axis = 3)

            i = 0
            for key, values in var_const_hr.items():

                if isinstance(TEST_SIZE, int):
                    const_tr_arr, const_te_arr = train_test_split(var_const_hr[key], test_size = TEST_SIZE, random_state = RANDOM_STATE, shuffle = False)
                elif isinstance(TEST_SIZE, dict):
                    # Create a mask for test and train sets
                    all_indices = np.arange(len(var_const_hr[key]))
                    train_idx = np.setdiff1d(all_indices, test_idx)
                    # Split manually
                    const_tr_arr, const_te_arr = var_const_hr[key][train_idx], var_const_hr[key][test_idx]

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

        """
        # <<< CHANGED v2 >>> Print tile sizes to verify they match model NX/NY before training
        if TILE_SIZE_LR is not None:
            tile_size_hr = TILE_SIZE_LR * SUBSAMPLING_LR
            print(f'Tiling: LR tile={TILE_SIZE_LR}, HR tile={tile_size_hr} '
                  f'(= {TILE_SIZE_LR} x SUBSAMPLING_LR={SUBSAMPLING_LR})')
            print(f'>>> Verify this matches your config NX={tile_size_hr}, NY={tile_size_hr} <<<')
        # <<< END CHANGED v2 >>>
        """

        #if var_const_hr:
        with tf.device("CPU"):
            dataset_train_all = tf.data.Dataset.from_tensor_slices(((X_train, const_train), y_train))
            dataset_test = tf.data.Dataset.from_tensor_slices(((X_test, const_test), y_test))
        #else: 
        #    dataset_train_all = tf.data.Dataset.from_tensor_slices((X_train, y_train))
        #    dataset_test = tf.data.Dataset.from_tensor_slices((X_test, y_test))
        print('dataset_train_all 0 shape:', dataset_train_all.element_spec)
        print('dataset_test 0 shape:', dataset_test.element_spec)

        """
        # This replaces full-domain samples with random spatial tiles,
        # reducing tensor size fed to ResizeNearestNeighbor and avoiding CUDA kernel limit crash.
        # Set TILE_SIZE_LR and TILE_SIZE_HR in your config/call to enable.
        # Set both to None to disable tiling and keep original full-domain behaviour.
        if TILE_SIZE_LR is not None and TILE_SIZE_HR is not None:
            print(f'Tiling enabled: LR tile={TILE_SIZE_LR}, HR tile={TILE_SIZE_HR}')  # <<< NEW >>>
            tile_fn = lambda inputs, target: self.random_tile_sample(            # <<< NEW >>>
                inputs, target, TILE_SIZE_LR, TILE_SIZE_HR)                 # <<< NEW >>>
            dataset_train_all = dataset_train_all.map(                      # <<< NEW >>>
                tile_fn, num_parallel_calls=tf.data.AUTOTUNE)               # <<< NEW >>>
            dataset_test = dataset_test.map(                                # <<< NEW >>>
                tile_fn, num_parallel_calls=tf.data.AUTOTUNE)               # <<< NEW >>>
        else:                                                               # <<< NEW >>>
            print('Tiling disabled: using full spatial domain')             # <<< NEW >>>
        """
        """
        # <<< CHANGED v2 >>> Tiling applied after augmentation, before batching
        # scale_factor = SUBSAMPLING_LR ensures HR tile matches generator output shape exactly
        if TILE_SIZE_LR is not None:
            tile_fn = lambda inputs, target: self.random_tile_sample(
                inputs, target,
                tile_size_lr=TILE_SIZE_LR,
                scale_factor=SUBSAMPLING_LR)           # <<< CHANGED v2 >>> pass scale_factor not tile_size_hr
            dataset_train_all = dataset_train_all.map(
                tile_fn, num_parallel_calls=tf.data.AUTOTUNE)
            dataset_test = dataset_test.map(
                tile_fn, num_parallel_calls=tf.data.AUTOTUNE)
            print(f'Tiling applied. New element spec after tiling:')
            print('  dataset_train_all:', dataset_train_all.element_spec)
            print('  dataset_test:     ', dataset_test.element_spec)
        else:
            print('Tiling disabled: using full spatial domain')
        # <<< END CHANGED V2 >>>
        """

        # <<< CHANGED v3 >>> derive LR tile sizes from NX, NY, SUBSAMPLING_LR
        if NX is not None and NY is not None:
            tile_size_lr_x = NX // SUBSAMPLING_LR   # = 106
            tile_size_lr_y = NY // SUBSAMPLING_LR   # = 88
            print(f'Tiling: LR=({tile_size_lr_y},{tile_size_lr_x}), '
                  f'HR=({NY},{NX}), scale={SUBSAMPLING_LR}')

            tile_fn = lambda inputs, target: self.random_tile_sample(
                inputs, target,
                tile_size_lr_y=tile_size_lr_y,
                tile_size_lr_x=tile_size_lr_x,
                scale_factor=SUBSAMPLING_LR)

            dataset_train_all = dataset_train_all.map(
                tile_fn, num_parallel_calls=tf.data.AUTOTUNE)
            dataset_test = dataset_test.map(
                tile_fn, num_parallel_calls=tf.data.AUTOTUNE)
        else:
            print('Tiling disabled: using full spatial domain')
        # <<< END CHANGED v3 >>>


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
        with tf.device("CPU"):
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

        with tf.device("CPU"):
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
