import sys
import os
import preprocess
import postprocess
import get_configuration
sys.path.insert(0, '..')
from utils import read, checkdir, file_writer, gpus_func, seed, get_time_range
from SRGANs import feature_selection, train
import numpy as np
import glob
import tensorflow as tf
import gc


def main():

    #os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    #tf.keras.backend.clear_session()
    #gc.collect()

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
    VALIDATION_SPLIT = cdict['stats_conf']['SRGAN']['VALIDATION_SPLIT']
    RANDOM_STATE = cdict['stats_conf']['SRGAN']['RANDOM_STATE']
    print('SRGAN PARAMETERS:', BATCH_SIZE, EPOCHS, EPOCH_INIT, TEST_SIZE, VALIDATION_SPLIT, RANDOM_STATE)

    SUBSAMPLING_LR = cdict['stats_conf']['TRAINING']['SUBSAMPLING_LR']
    N_RES_BLOCK = cdict['stats_conf']['TRAINING']['N_RES_BLOCK']
    #INPUT_CHANNELS = cdict['stats_conf']['TRAINING']['INPUT_CHANNELS']
    #OUTPUT_CHANNELS = cdict['stats_conf']['TRAINING']['OUTPUT_CHANNELS']
    NX = cdict['stats_conf']['TRAINING']['NX']
    NY = cdict['stats_conf']['TRAINING']['NY']
    LEARNING_RATE = cdict['stats_conf']['TRAINING']['LEARNING_RATE']
    DROPOUT_RATE = cdict['stats_conf']['TRAINING']['DROPOUT_RATE']
    EARLY_STOP = cdict['stats_conf']['TRAINING']['EARLY_STOP']
    DATA_AUGMENTATION = cdict['stats_conf']['TRAINING']['DATA_AUGMENTATION']
    DISABLE_PARALLEL = cdict['stats_conf']['TRAINING']['DISABLE_PARALLEL']
    ENFORCE_DETERMINISM = cdict['stats_conf']['TRAINING']['ENFORCE_DETERMINISM']
    USE_SEED = cdict['stats_conf']['TRAINING']['USE_SEED']

    TRAINING_MODE = cdict['stats_conf']['RUNNING_MODE']['TRAINING_MODE']
    PREDICTION_MODE = cdict['stats_conf']['RUNNING_MODE']['PREDICTION_MODE']
    PRETRAINED_GENERATOR = cdict['stats_conf']['RUNNING_MODE']['PRETRAINED_GENERATOR']

    if USE_SEED:
        seed.set_seed(seed = 42, disable_parallel = DISABLE_PARALLEL, enforce_determinism = ENFORCE_DETERMINISM)

    experiment_name = cdict['experiment_name']
    path_main = cdict['path_main']
    file_x_mode, path_x = cdict['path_x']['file_mode'], cdict['path_x']['path']
    file_y_mode, path_y = cdict['path_y']['file_mode'], cdict['path_y']['path']
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

    dir_const = path_y + '/'  + resolution_const + '/' + frequency_const # + '/orog/' 

    checkdir.checkdir(path_figure)
    checkdir.checkdir(wdir)

    step_hours = int(cdict['time_range']['step_hours'])
    start_date_all_lowres, end_date_all_lowres= \
            cdict['time_range']['all_lowres']['start_date'], cdict['time_range']['all_lowres']['end_date']
    start_date_all_highres, end_date_all_highres= \
            cdict['time_range']['all_highres']['start_date'], cdict['time_range']['all_highres']['end_date']
    start_date_target, end_date_target = \
            cdict['time_range']['target']['start_date'], cdict['time_range']['target']['end_date']


    all_times_lowres  = get_time_range.generate_time_series(start_date_all_lowres, end_date_all_lowres, step_hours)
    all_times_highres = get_time_range.generate_time_series(start_date_all_highres, end_date_all_highres, step_hours)
    start_idx_lowres,  end_idx_lowres  = get_time_range.get_time_indices(all_times_lowres, start_date_target, end_date_target)
    start_idx_highres, end_idx_highres = get_time_range.get_time_indices(all_times_highres, start_date_target, end_date_target)
    time_idx_range_lowres = {'start_idx': start_idx_lowres, 'end_idx': end_idx_lowres}
    time_idx_range_highres = {'start_idx': start_idx_highres, 'end_idx': end_idx_highres}
    print(f"Start index lowres: {start_idx_lowres}, End index: {end_idx_lowres}")
    print(f"Start time lowres: {all_times_lowres[start_idx_lowres]}, End time: {all_times_lowres[end_idx_lowres]}")
    print(f"Start index highres: {start_idx_highres}, End index: {end_idx_highres}")
    print(f"Start time highres: {all_times_highres[start_idx_highres]}, End time: {all_times_highres[end_idx_highres]}")


    '''
    preproc = preprocess.PreProcess()
    var_lr_scaled = preproc.scale_var(var_low_res)
    var_hr_scaled = preproc.scale_var(var_high_res)
    var_lr_filtered = preproc.filter_var(var_lr_scaled, var_hr_scaled)
    dataset_train, dataset_test, X_train, X_test, y_train, y_test = preproc.split_data(var_lr_filtered, var_hr_scaled, BATCH_SIZE)
    '''
    # reading data
    with tf.device("CPU"):
        # https://stackoverflow.com/questions/72122939/resourceexhaustederror-graph-execution-error-when-trying-to-train-tensorflow
        readin = read.Read('netcdf')
        if file_x_mode == 'one_var_per_file': 
            var_low_res_dict = readin.read_netcdf_one_var_per_file(dir_low_res, varname_predictor_low_res,  file_filter, time_idx_range_lowres)
        elif file_x_mode == 'multivar_singlefile': 
            var_low_res_dict = readin.read_netcdf_multivar_singlefile(dir_low_res, varname_predictor_low_res,  file_filter, time_idx_range_lowres)

        if file_y_mode == 'one_var_per_file': 
            var_high_res_dict = readin.read_netcdf_one_var_per_file(dir_high_res, varname_predictand_high_res, file_filter, time_idx_range_highres)
        elif file_y_mode == 'multivar_singlefile': 
            var_high_res_dict = readin.read_netcdf_multivar_singlefile(dir_high_res, varname_predictand_high_res,  file_filter, time_idx_range_highres)

    # constant fields (eg, orography)
    var_const_high_res_dict = {}
    if len(varname_const) > 0:
        #preproc = preprocess.PreProcess()
        var_const_dict = readin.read_netcdf_one_var_per_file(dir_const, varname_const, file_filter_const)
        nt, nx, ny = np.shape(var_high_res_dict[varname_predictand_high_res[0]])
        for key, values in var_const_dict.items():
            var_const_add_time = np.repeat(values[None, :, :], nt, axis=0)
            var_const_high_res_dict[key] = var_const_add_time
            print('var_const_high_res', np.shape(var_const_add_time))
            #var_const_filtered = preproc.filter_var(var_low_res_dict[varname_predictor_low_res[0]], var_const_add_time)
            ##var_low_res_dict[key] = np.squeeze(var_const_filtered, axis=2)
            #var_low_res_dict[key] = var_const_filtered[0,:,:]
            ##var_low_res_dict[key] = np.delete(var_const_filtered, axis = 0)


    num_gpus = gpus_func.get_num_gpus()
    if num_gpus > 0: assert BATCH_SIZE % num_gpus == 0

    print('num_gpus, BATCH_SIZE', num_gpus, BATCH_SIZE)

    #os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    gpus_func.set_gpus(num_gpus)

    multiple_GPUs_with_virtual_devices = False #False


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


    postproc = postprocess.PostProcess()
    postproc.plot_input_data(\
        var_low_res_filtered_dict, \
        var_high_res_scaled_dict, path_figure)
        #var_low_res_filtered_dict[varname_predictand_high_res[0]], \
        #var_high_res_scaled_dict[varname_predictand_high_res[0]], path_figure)

    #preproc.split_data(var_low_res_filtered_dict, var_high_res_scaled_dict, \
    dataset_train, dataset_valid, dataset_test, X_train, X_test, const_train, const_test, y_train, y_test = \
        preproc.split_data(var_low_res_filtered_dict, var_const_high_res_scaled_dict, var_high_res_scaled_dict, \
        BATCH_SIZE, TEST_SIZE, VALIDATION_SPLIT, RANDOM_STATE, DATA_AUGMENTATION, \
        varname_predictand_high_res[0], downscale_mode)

    """
    for batch in dataset_train.take(1):
      if isinstance(batch, tuple):
        for i, element in enumerate(batch):
            print(f'batch[{i}] type:', type(element))
            if hasattr(element, 'shape'):
                print(f'batch[{i}].shape:', element.shape)
            else:
                print(f'batch[{i}] is not a tensor')
      else:
        print("batch is not a tuple")
    """

    #feature_selection.stepwise_algorithm(X_train, X_test, y_train, y_test)
    print('X_test, const_test, y_test:', np.shape(X_test), np.shape(const_test), np.shape(y_test))
    postproc.plot_result(y_test, X_test, y_test, path_figure, varname_predictor, varname_predictand_high_res)

    
    if TRAINING_MODE == True:
        if num_gpus <= 1:

            # training
            trainmodel = train.TrainModel(wdir)
            generator = trainmodel.training(BATCH_SIZE, EPOCH_INIT, EPOCHS,
                SUBSAMPLING_LR, N_RES_BLOCK, INPUT_CHANNELS, OUTPUT_CHANNELS, NX, NY, 
                METHOD, LEARNING_RATE, DROPOUT_RATE, EARLY_STOP,
                dataset_train, dataset_valid)

        elif num_gpus > 1:

            #tf.debugging.set_log_device_placement(True)
            strategy = gpus_func.get_strategy()

            #strategy.reduce(tf.distribute.ReduceOp.SUM, tensor, axis=None)
            print(f"Number of GPUs Available: {strategy.num_replicas_in_sync}")

            with strategy.scope():
                # training
                trainmodel = train.TrainModel(wdir)
                generator = trainmodel.training(BATCH_SIZE, EPOCH_INIT, EPOCHS,
                    SUBSAMPLING_LR, N_RES_BLOCK, INPUT_CHANNELS, OUTPUT_CHANNELS, NX, NY, 
                    METHOD, LEARNING_RATE, DROPOUT_RATE, EARLY_STOP,
                    dataset_train, dataset_valid)
        """
        trainmodel = train.TrainModel(wdir)
        generator = trainmodel.training(BATCH_SIZE, num_gpus, EPOCH_INIT, EPOCHS, 
            SUBSAMPLING_LR, N_RES_BLOCK, INPUT_CHANNELS, OUTPUT_CHANNELS, NX, NY, 
            METHOD, LEARNING_RATE, DROPOUT_RATE, EARLY_STOP,
            dataset_train, dataset_valid)
        """
    else:

        generator = tf.keras.models.load_model(PRETRAINED_GENERATOR)

    if PREDICTION_MODE == True:
        trainmodel = train.TrainModel(wdir)
        y_pred = trainmodel.prediction(generator, X_test, const_test, y_test, BATCH_SIZE)


    """
    # Step 4: Get generator output
    output_image = generator.predict(X_test, const_test)
    print("\n Generator Output:")
    print("Shape:", output_image.shape)
    print("Sample pixel (0,0):", output_image[0, 0, 0])

    # Step 5: Extract and print intermediate activation
    layer_name = 'Conv2d'  # this is the first conv layer we named earlier
    intermediate_model = models.Model(inputs=generator.input,
                                  outputs=generator.get_layer(layer_name).output)

    activations = intermediate_model.predict(X_test, const_test)
    print("\n⚡ Intermediate Activation (first_conv):")
    print("Shape:", activations.shape)
    print("First channel value at (0,0):", activations[0, 0, 0, 0])
    """

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

