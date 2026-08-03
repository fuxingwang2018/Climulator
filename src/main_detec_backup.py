import sys
import os
import preprocess
import postprocess
import get_configuration
sys.path.insert(0, '..')
from utils import read, checkdir, file_writer 
from SRGANs import feature_selection, train
import numpy as np
import glob

def main():

    # Get configuration 
    args = get_configuration.get_args()
    config_file = args.config
    if not os.path.isfile(config_file):
        raise ValueError(f"\nConfig file, '{config_file}', does not exist!")
    cdict = get_configuration.get_settings(config_file)

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

    path_main, path_x, path_y = cdict['path_main'], cdict['path_x'], cdict['path_y']
    wdir = path_main + '/SRGAN_OUT/' + 'EPOCH' + str(EPOCHS) + '/'
    path_figure = path_main + '/Figure/' + 'EPOCH' + str(EPOCHS) + '/'
    print(cdict['variables'])

    resolution_low = cdict['variables']['low resolution']['resolution']
    frequency_low_res = cdict['variables']['low resolution']['freq']
    varname_predictor_low_res = cdict['variables']['low resolution']['var names']

    resolution_const = cdict['variables']['constant fields']['resolution']
    frequency_const = cdict['variables']['constant fields']['freq']
    varname_const = cdict['variables']['constant fields']['var names']

    varname_predictor = varname_predictor_low_res + varname_const

    INPUT_CHANNELS = len(varname_predictor) 

    resolution_high = cdict['variables']['high resolution']['resolution']
    frequency_high_res = cdict['variables']['high resolution']['freq']
    varname_predictand_high_res = cdict['variables']['high resolution']['var names'] #+ varname_const

    OUTPUT_CHANNELS = len(varname_predictand_high_res) #+ len(varname_const)

    print('TRAINING PARAMETERS:', SUBSAMPLING_LR, N_RES_BLOCK, INPUT_CHANNELS, OUTPUT_CHANNELS, NX, NY)
    dir_low_res = path_x + '/' + resolution_low + '/' + frequency_low_res + '/' 
    dir_high_res = path_y + '/' + resolution_high + '/' + frequency_high_res + '/'
    file_filter = '' #'hr_2000' # 'July'
    downscale_mode = cdict['downscale mode']

    dir_const = path_x + '/'  + resolution_const + '/' + frequency_const + '/orog/' 
    file_filter_const = 'fx' 

    checkdir.checkdir(path_figure)
    checkdir.checkdir(wdir)

    '''
    preproc = preprocess.PreProcess()
    var_lr_scaled = preproc.scale_var(var_low_res)
    var_hr_scaled = preproc.scale_var(var_high_res)
    var_lr_filtered = preproc.filter_var(var_lr_scaled, var_hr_scaled)
    dataset_train, dataset_test, X_train, X_test, y_train, y_test = preproc.split_data(var_lr_filtered, var_hr_scaled, BATCH_SIZE)
    '''
    # reading data
    readin = read.Read('netcdf')
    var_low_res_dict = readin.read_netcdf_one_var_per_file(dir_low_res, varname_predictor_low_res,  file_filter)
    var_high_res_dict = readin.read_netcdf_one_var_per_file(dir_high_res, varname_predictand_high_res, file_filter)

    # constant fields (eg, orography)
    nt, nx, ny = np.shape(var_high_res_dict[varname_predictand_high_res[0]])
    if len(varname_const) > 0:
        preproc = preprocess.PreProcess()
        var_const_dict = readin.read_netcdf(dir_const, varname_const, file_filter_const)
        for key, values in var_const_dict.items():
            #var_high_res_dict[key] = values #np.repeat(values[None, :, :], nt, axis=0)
            var_const_add_time = np.repeat(values[None, :, :], nt, axis=0)
            var_const_filtered = preproc.filter_var(var_low_res_dict[varname_predictor_low_res[0]], var_const_add_time)
            print('var_const_filtered shape', np.shape(var_const_filtered))
            #var_low_res_dict[key] = np.squeeze(var_const_filtered, axis=2)
            var_low_res_dict[key] = var_const_filtered[0,:,:]
            #var_low_res_dict[key] = np.delete(var_const_filtered, axis = 0)

    preproc = preprocess.PreProcess()
    var_low_res_scaled_dict = preproc.scale_dict(var_low_res_dict)
    for var_const in varname_const:
        var_low_res_scaled_dict[var_const] = np.repeat(var_low_res_scaled_dict[var_const][None, :, :], nt, axis=0) 
    var_high_res_scaled_dict = preproc.scale_dict(var_high_res_dict)
    var_low_res_adjusted_dict, var_high_res_adjusted_dict = \
        preproc.adjust_data_size(var_low_res_scaled_dict, var_high_res_scaled_dict, BATCH_SIZE)

    for key, values in var_low_res_adjusted_dict.items():
        print('shape var_low_res_adjusted_dict:', key, values.shape)
    for key, values in var_high_res_adjusted_dict.items():
        print('shape var_high_res_adjusted_dict:', key, values.shape)

    print('downscale_mode', downscale_mode)
    if downscale_mode == 'upscale':
        # First upscale from 3km to 12km
        var_low_res_filtered_dict = preproc.filter_dict(var_low_res_adjusted_dict, var_high_res_adjusted_dict)
    elif downscale_mode == 'direct':
        var_low_res_filtered_dict = var_low_res_adjusted_dict.copy()
    for key, values in var_low_res_filtered_dict.items():
        print('shape var_low_res_filtered:', values.shape)

    postproc = postprocess.PostProcess()
    postproc.plot_input_data(\
        var_low_res_filtered_dict, \
        var_high_res_scaled_dict, path_figure)

    #preproc.split_data(var_low_res_filtered_dict, var_high_res_scaled_dict, \
    dataset_train, dataset_test, X_train, X_test, y_train, y_test = \
        preproc.split_data(var_low_res_filtered_dict, var_high_res_adjusted_dict, \
        BATCH_SIZE, TEST_SIZE, RANDOM_STATE, \
        varname_predictand_high_res[0], downscale_mode)
    #feature_selection.stepwise_algorithm(X_train, X_test, y_train, y_test)
    print('X_test, y_test:', np.shape(X_test), np.shape(y_test))
    postproc.plot_result(y_test, X_test, y_test, path_figure, varname_predictor, varname_predictand_high_res)
    #sys.exit()

    # training
    trainmodel = train.TrainModel(wdir)
    generator = trainmodel.training(BATCH_SIZE, EPOCH_INIT, EPOCHS, 
        SUBSAMPLING_LR, N_RES_BLOCK, INPUT_CHANNELS, OUTPUT_CHANNELS, NX, NY,
        dataset_train, dataset_test)
    y_pred = trainmodel.prediction(generator, X_test, y_test)

    print('min max of y_test', np.nanmin(y_test), np.nanmax(y_test))
    print('min max of y_pred', np.nanmin(y_pred), np.nanmax(y_pred))

    # write results to netcdf
    print('varname_predictand_high_res', varname_predictand_high_res)
    print('varname_predictor_low_res', varname_predictor_low_res)

    var_to_write_x, var_to_write_y = {}, {}
    varname_to_write_x_list = []
    nt_x_test, ny_x_test, nx_x_test, nvar_x_test = np.shape(X_test)
    for ivar_x_test in range(nvar_x_test):
        varnam_to_write_x = 'x_test_' + varname_predictor_low_res[ivar_x_test]
        var_to_write_x[varnam_to_write_x] = X_test[:, :, :, ivar_x_test] 
        varname_to_write_x_list.append(varnam_to_write_x)
    #var_to_write_x['x_test'] = X_test[:, :, :, 0] if X_test.ndim == 4 else x_test[:, :, :]
    var_to_write_y['y_test'] = y_test[:, :, :, 0] if y_test.ndim == 4 else y_test[:, :, :]
    var_to_write_y['y_pred'] = y_pred[:, :, :, 0] if y_pred.ndim == 4 else y_pred[:, :, :]

    filewriter_y = file_writer.FileWriter(wdir + '/' + 'predictant.nc')
    nc_files_to_read_y = glob.glob(dir_high_res + '/' + varname_predictand_high_res[0] + '/' + '*')
    nc_files_to_read_y.sort()
    print('high res file', nc_files_to_read_y[0])
    filewriter_y.Write_NC(nc_files_to_read_y[0], \
                        varname_predictand_high_res[0], \
                        ['y_pred', 'y_test'], \
                        TEST_SIZE, var_to_write_y)

    filewriter_x = file_writer.FileWriter(wdir + '/' + 'predictor.nc')
    nc_files_to_read_x = glob.glob(dir_low_res +  '/' + varname_predictor_low_res[0] + '/' + '*')
    nc_files_to_read_x.sort()
    print('low res file', nc_files_to_read_x[0])
    filewriter_x.Write_NC(nc_files_to_read_x[0], \
                        varname_predictor_low_res[0], \
                        varname_to_write_x_list, \
                        TEST_SIZE, var_to_write_x)


    postproc.plot_result(y_pred, X_test, y_test, path_figure, varname_predictor, varname_predictand_high_res)
    print('Completed!')


if __name__ == "__main__":
    main()

