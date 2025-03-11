import sys
import os
import preprocess
import postprocess
import get_configuration
sys.path.insert(0, '..')
from utils import read, checkdir, file_writer, gpus_func 
from SRGANs import feature_selection, train
import numpy as np
import glob
import tensorflow as tf
from tensorflow.keras import mixed_precision

def main():

    # Get configuration 
    args = get_configuration.get_args()
    config_file = args.config
    if not os.path.isfile(config_file):
        raise ValueError(f"\nConfig file, '{config_file}', does not exist!")
    cdict = get_configuration.get_settings(config_file)

    METHOD = cdict['stats_conf']['SRGAN']['METHOD']
    BATCH_SIZE = cdict['stats_conf']['SRGAN']['BATCH_SIZE']
    EPOCHS = cdict['stats_conf']['SRGAN']['EPOCHS']
    EPOCH_INIT = cdict['stats_conf']['SRGAN']['EPOCH_INIT']
    TEST_SIZE = cdict['stats_conf']['SRGAN']['TEST_SIZE']
    RANDOM_STATE = cdict['stats_conf']['SRGAN']['RANDOM_STATE']
    print('SRGAN PARAMETERS:', BATCH_SIZE, EPOCHS, EPOCH_INIT, TEST_SIZE, RANDOM_STATE)

    SUBSAMPLING_LR = cdict['stats_conf']['TRAINING']['SUBSAMPLING_LR']
    N_RES_BLOCK = cdict['stats_conf']['TRAINING']['N_RES_BLOCK']
    #INPUT_CHANNELS = cdict['stats_conf']['TRAINING']['INPUT_CHANNELS']
    #OUTPUT_CHANNELS = cdict['stats_conf']['TRAINING']['OUTPUT_CHANNELS']
    NX = cdict['stats_conf']['TRAINING']['NX']
    NY = cdict['stats_conf']['TRAINING']['NY']
    LEARNING_RATE = cdict['stats_conf']['TRAINING']['LEARNING_RATE']
    DROPOUT_RATE = cdict['stats_conf']['TRAINING']['DROPOUT_RATE']
    EARLY_STOP = cdict['stats_conf']['TRAINING']['EARLY_STOP']

    experiment_name = cdict['experiment_name']
    path_main, path_x, path_y = cdict['path_main'], cdict['path_x'], cdict['path_y']
    wdir = path_main + '/SRGAN_OUT/' + 'EPOCH' + str(EPOCHS) + '_' + str(experiment_name) + '/'
    path_figure = path_main + '/Figure/' + 'EPOCH' + str(EPOCHS) + '_' + str(experiment_name) + '/'

    #file_filter = '' #'hr_2000' # 'July'
    file_filter = cdict['file_filter'] #'hr_2000' # 'July'
    file_filter_const = cdict['file_filter_const'] #'fx' 

    print(cdict['variables'])

    resolution_low = cdict['variables']['low resolution']['resolution']
    frequency_low_res = cdict['variables']['low resolution']['freq']
    varname_predictor_low_res = cdict['variables']['low resolution']['var names']

    resolution_const = cdict['variables']['constant fields']['resolution']
    frequency_const = cdict['variables']['constant fields']['freq']
    varname_const = cdict['variables']['constant fields']['var names']

    varname_predictor = varname_predictor_low_res #+ varname_const

    INPUT_CHANNELS = len(varname_predictor) #+ len(varname_const) 

    resolution_high = cdict['variables']['high resolution']['resolution']
    frequency_high_res = cdict['variables']['high resolution']['freq']
    varname_predictand_high_res = cdict['variables']['high resolution']['var names'] #+ varname_const

    OUTPUT_CHANNELS = len(varname_predictand_high_res) #+ len(varname_const)

    print('TRAINING PARAMETERS:', SUBSAMPLING_LR, N_RES_BLOCK, INPUT_CHANNELS, OUTPUT_CHANNELS, \
            NX, NY, LEARNING_RATE, DROPOUT_RATE, EARLY_STOP)
    dir_low_res = path_x + '/' + resolution_low + '/' + frequency_low_res + '/' 
    dir_high_res = path_y + '/' + resolution_high + '/' + frequency_high_res + '/'
    downscale_mode = cdict['downscale mode']

    dir_const = path_x + '/'  + resolution_const + '/' + frequency_const # + '/orog/' 

    checkdir.checkdir(path_figure)
    checkdir.checkdir(wdir)

    '''
    preproc = preprocess.PreProcess()
    var_lr_scaled = preproc.scale_var(var_low_res)
    var_hr_scaled = preproc.scale_var(var_high_res)
    var_lr_filtered = preproc.filter_var(var_lr_scaled, var_hr_scaled)
    dataset_train, dataset_valid, X_train, X_test, y_train, y_test = preproc.split_data(var_lr_filtered, var_hr_scaled, BATCH_SIZE)
    '''
    # reading data
    readin = read.Read('netcdf')
    var_low_res_dict = readin.read_netcdf(dir_low_res, varname_predictor_low_res,  file_filter)
    var_high_res_dict = readin.read_netcdf(dir_high_res, varname_predictand_high_res, file_filter)

    # constant fields (eg, orography)
    var_const_high_res_dict = {}
    if len(varname_const) > 0:
        #preproc = preprocess.PreProcess()
        var_const_dict = readin.read_netcdf(dir_const, varname_const, file_filter_const)
        nt, nx, ny = np.shape(var_high_res_dict[varname_predictand_high_res[0]])
        for key, values in var_const_dict.items():
            var_const_add_time = np.repeat(values[None, :, :], nt, axis=0)
            var_const_high_res_dict[key] = var_const_add_time
            print('var_const_high_res', np.shape(var_const_add_time))
            #var_const_filtered = preproc.filter_var(var_low_res_dict[varname_predictor_low_res[0]], var_const_add_time)
            ##var_low_res_dict[key] = np.squeeze(var_const_filtered, axis=2)
            #var_low_res_dict[key] = var_const_filtered[0,:,:]
            ##var_low_res_dict[key] = np.delete(var_const_filtered, axis = 0)

    preproc = preprocess.PreProcess()

    """
    threshold = 1e10
    if any((value > threshold).any() for value in var_low_res_dict.values()):
        print('missing values in var_low_res_dict', var_low_res_dict.keys)
        var_low_res_dict  = preproc.fill_missing_with_interpolation_dict(var_low_res_dict)

    if any((value > threshold).any() for value in var_high_res_dict.values()):
        print('missing values in var_high_res_dict', var_high_res_dict.keys)
        var_high_res_dict = preproc.fill_missing_with_interpolation_dict(var_high_res_dict)
    """

    var_low_res_adjusted_dict, var_const_high_res_adjusted_dict, var_high_res_adjusted_dict, \
        residue_time_low_res, residue_time_const_high_res, residue_time_high_res, residue_geo_dict = \
        preproc.adjust_data_size(var_low_res_dict, var_const_high_res_dict, var_high_res_dict, BATCH_SIZE)

    var_low_res_scaled_dict = preproc.scale_dict(var_low_res_adjusted_dict)
    var_const_high_res_scaled_dict = preproc.scale_const_dict(var_const_high_res_adjusted_dict)
    var_high_res_scaled_dict = preproc.scale_dict(var_high_res_adjusted_dict)

    for key, values in var_low_res_adjusted_dict.items():
        print('shape var_low_res_adjusted_dict:', key, values.shape)
    for key, values in var_const_high_res_adjusted_dict.items():
        print('shape var_const_high_res_adjusted_dict:', key, values.shape)
    for key, values in var_high_res_adjusted_dict.items():
        print('shape var_high_res_adjusted_dict:', key, values.shape)

    print('downscale_mode', downscale_mode)
    if downscale_mode == 'upscale':
        # First upscale from 3km to 12km
        var_low_res_filtered_dict = preproc.filter_dict(var_low_res_scaled_dict, var_high_res_scaled_dict)
    elif downscale_mode == 'direct':
        var_low_res_filtered_dict = var_low_res_scaled_dict.copy()
    for key, values in var_low_res_filtered_dict.items():
        print('shape var_low_res_filtered:', values.shape)

    #os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    num_gpus = gpus_func.get_num_gpus()
    #print('num_gpus', num_gpus)

    #if num_gpus > 0:
    #    mixed_precision.set_global_policy("mixed_float16")

    multiple_GPUs_with_virtual_devices = False #False

    # Limit GPU memory usage (optional)
    if num_gpus <= 1:
        # one gpu 
        gpus = tf.config.list_physical_devices('GPU')
        print('gpus:', gpus)
        if gpus:
            try:
                tf.config.set_visible_devices(gpus[0], 'GPU')
                # Currently, memory growth needs to be the same across GPUs
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                    #tf.config.experimental.set_virtual_device_configuration(gpus[0],
                    #    [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=4096)]  # Set memory limit in MB
                    #    )
                logical_gpus = tf.config.list_logical_devices('GPU')
                print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")

                if multiple_GPUs_with_virtual_devices:
                    tf.config.set_logical_device_configuration(
                        gpus[0],
                        [tf.config.LogicalDeviceConfiguration(memory_limit=1024),
                         tf.config.LogicalDeviceConfiguration(memory_limit=1024)])
            except RuntimeError as e:
                # Memory growth must be set before GPUs have been initialized
                print('No GPU Error!')
                print(e)


    postproc = postprocess.PostProcess()
    postproc.plot_input_data(\
        var_low_res_filtered_dict, \
        var_high_res_scaled_dict, path_figure)
        #var_low_res_filtered_dict[varname_predictand_high_res[0]], \
        #var_high_res_scaled_dict[varname_predictand_high_res[0]], path_figure)

    #preproc.split_data(var_low_res_filtered_dict, var_high_res_scaled_dict, \
    dataset_train, dataset_valid, X_train, X_test, const_train, const_test, y_train, y_test = \
        preproc.split_data(var_low_res_filtered_dict, var_const_high_res_scaled_dict, var_high_res_scaled_dict, \
        BATCH_SIZE, TEST_SIZE, RANDOM_STATE, \
        varname_predictand_high_res[0], downscale_mode)
    
    #feature_selection.stepwise_algorithm(X_train, X_test, y_train, y_test)
    print('X_test, const_test, y_test:', np.shape(X_test), np.shape(const_test), np.shape(y_test))
    postproc.plot_result(y_test, X_test, y_test, path_figure, varname_predictor, varname_predictand_high_res)

    """
    #Apply Data Augmentation
    data_augmentation = tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal"),  # Random horizontal flip
        tf.keras.layers.RandomRotation(0.1),      # Small random rotation
        tf.keras.layers.RandomZoom(0.1),          # Random zoom
    ])
    # Apply augmentation to LR and HR images
    dataset_train = data_augmentation(dataset_train)
    dataset_valid = data_augmentation(dataset_valid)
    """


    if num_gpus <= 1:
        # training
        trainmodel = train.TrainModel(wdir)
        generator = trainmodel.training(BATCH_SIZE, EPOCH_INIT, EPOCHS, 
            SUBSAMPLING_LR, N_RES_BLOCK, INPUT_CHANNELS, OUTPUT_CHANNELS, NX, NY, 
            METHOD, LEARNING_RATE, DROPOUT_RATE, EARLY_STOP,
            dataset_train, dataset_valid)

    elif num_gpus > 1:

        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            try:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                    tf.config.experimental.set_virtual_device_configuration(gpus[0],
                        [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=4096)]  # Set memory limit in MB
                        )

                print("Enabled GPU memory growth")
            except RuntimeError as e:
                print(e)

        # Enable multi-GPU strategy
        tf.debugging.set_log_device_placement(True)
        strategy = gpus_func.get_strategy()

        print(f"Number of GPUs Available: {strategy.num_replicas_in_sync}")

        with strategy.scope():
            # training
            trainmodel = train.TrainModel(wdir)
            generator = trainmodel.training(BATCH_SIZE, EPOCH_INIT, EPOCHS, 
                SUBSAMPLING_LR, N_RES_BLOCK, INPUT_CHANNELS, OUTPUT_CHANNELS, NX, NY, 
                METHOD, LEARNING_RATE, DROPOUT_RATE, EARLY_STOP,
                dataset_train, dataset_valid)

    y_pred = trainmodel.prediction(generator, X_test, const_test, y_test, BATCH_SIZE)

    print('min max of y_test', np.nanmin(y_test), np.nanmax(y_test))
    print('min max of y_pred', np.nanmin(y_pred), np.nanmax(y_pred))

    # write results to netcdf
    print('high res file', glob.glob(dir_high_res + '*')[0])
    print('varname_predictor_low_res', varname_predictor_low_res)
    print('varname_predictand_high_res', varname_predictand_high_res)

    var_to_write_x, var_to_write_ypred, var_to_write_ytest = {}, {}, {}
    ivar = 0
    for ivar_predictor in varname_predictor_low_res:
        var_to_write_x[ivar_predictor] = X_test[:, :, :, ivar] if X_test.ndim == 4 else X_test[:, :, :]
        ivar += 1
    ivar = 0
    for ivar_predictand in varname_predictand_high_res:
        var_to_write_ypred[ivar_predictand] = y_pred[:, :, :, ivar] if y_pred.ndim == 4 else y_pred[:, :, :]
        var_to_write_ytest[ivar_predictand] = y_test[:, :, :, ivar] if y_test.ndim == 4 else y_test[:, :, :]
        ivar += 1
    #var_to_write_x['x_test'] = X_test[:, :, :, 0] if X_test.ndim == 4 else x_test[:, :, :]
    #var_to_write_y['y_test'] = y_test[:, :, :, 0] if y_test.ndim == 4 else y_test[:, :, :]
    #var_to_write_y['y_pred'] = y_pred[:, :, :, 0] if y_pred.ndim == 4 else y_pred[:, :, :]


    var_to_write_x_inverse = preproc.inverse_dict(var_to_write_x, var_low_res_adjusted_dict)
    var_to_write_ypred_inverse = preproc.inverse_dict(var_to_write_ypred, var_high_res_adjusted_dict)
    var_to_write_ytest_inverse = preproc.inverse_dict(var_to_write_ytest, var_high_res_adjusted_dict)

    filewriter_ypred = file_writer.FileWriter(wdir + '/' + 'predictant_ypred.nc')
    nc_files_to_read_y = glob.glob(dir_high_res + '/' + varname_predictand_high_res[0] + '/' + '*')
    nc_files_to_read_y.sort()
    #filewriter_y.Write_NC(glob.glob(dir_high_res + varname_predictand_high_res[0] + '*')[0], \
    print('high res file', nc_files_to_read_y[0])
    filewriter_ypred.Write_NC(nc_files_to_read_y[0], \
                        varname_predictand_high_res[0], \
                        #['y_pred', 'y_test'], \
                        varname_predictand_high_res,
                        residue_time_high_res, TEST_SIZE, residue_geo_dict, var_to_write_ypred_inverse)


    filewriter_ytest = file_writer.FileWriter(wdir + '/' + 'predictant_ytest.nc')
    nc_files_to_read_y = glob.glob(dir_high_res + '/' + varname_predictand_high_res[0] + '/' + '*')
    nc_files_to_read_y.sort()
    #filewriter_y.Write_NC(glob.glob(dir_high_res + varname_predictand_high_res[0] + '*')[0], \
    #print('high res file', nc_files_to_read_y[0])
    filewriter_ytest.Write_NC(nc_files_to_read_y[0], \
                        varname_predictand_high_res[0], \
                        varname_predictand_high_res,
                        residue_time_high_res, TEST_SIZE, residue_geo_dict, var_to_write_ytest_inverse)

    filewriter_x = file_writer.FileWriter(wdir + '/' + 'predictor.nc')
    nc_files_to_read_x = glob.glob(dir_low_res +  '/' + varname_predictor_low_res[0] + '/' + '*')
    #filewriter_x.Write_NC(glob.glob(dir_low_res + varname_predictor_low_res[0] + '*')[0], \
    nc_files_to_read_x.sort()
    print('low res file', nc_files_to_read_x[0])
    filewriter_x.Write_NC(nc_files_to_read_x[0], \
                        varname_predictor_low_res[0], \
                        #['x_test'], \
                        varname_predictor_low_res, \
                        residue_time_low_res, TEST_SIZE, {'x':0, 'y':0}, var_to_write_x_inverse)

    postproc.plot_result(y_pred, X_test, y_test, path_figure, varname_predictor, varname_predictand_high_res)
    print('Completed!')


if __name__ == "__main__":
    main()

