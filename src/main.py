import sys
import os
import preprocess
import postprocess
import get_configuration
sys.path.insert(0, '..')
from utils import read, checkdir
from SRGANs import feature_selection, train


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
    print(BATCH_SIZE, EPOCHS, EPOCH_INIT, TEST_SIZE, RANDOM_STATE)

    path_main = cdict['path_main']
    wdir = path_main + '/SRGAN_OUT/' + 'EPOCH' + str(EPOCHS) + '/'
    path_figure = path_main + '/Figure/' + 'EPOCH' + str(EPOCHS) + '/'
    print(cdict['variables'])

    resolution_low = cdict['variables']['low resolution']['resolution']
    frequency_low_res = cdict['variables']['low resolution']['freq']
    varname_predictor_low_res = cdict['variables']['low resolution']['var names']

    resolution_high = cdict['variables']['high resolution']['resolution']
    frequency_high_res = cdict['variables']['high resolution']['freq']
    varname_predictand_high_res = cdict['variables']['high resolution']['var names']

    dir_low_res = path_main + '/' + resolution_low + '/' + frequency_low_res + '/' 
    dir_high_res = path_main + '/' + resolution_high + '/' + frequency_high_res + '/'
    file_filter = 'hr_2000' # 'July'

    checkdir.checkdir(path_figure)
    checkdir.checkdir(wdir)

    '''
    preproc = preprocess.PreProcess()
    var_lr_scaled = preproc.scale_var(var_low_res)
    var_hr_scaled = preproc.scale_var(var_high_res)
    var_lr_filtered = preproc.filter_var(var_lr_scaled, var_hr_scaled)
    dataset_train, dataset_valid, X_train, X_test, y_train, y_test = preproc.split_data(var_lr_filtered, var_hr_scaled, BATCH_SIZE)
    '''
    readin = read.Read('netcdf')
    var_low_res_dict = readin.read_netcdf(dir_low_res, varname_predictor_low_res,  file_filter)
    var_high_res_dict = readin.read_netcdf(dir_high_res, varname_predictand_high_res, file_filter)

    preproc = preprocess.PreProcess()
    var_low_res_scaled_dict = preproc.scale_dict(var_low_res_dict)
    var_high_res_scaled_dict = preproc.scale_dict(var_high_res_dict)
    var_low_res_adjusted_dict, var_high_res_adjusted_dict = \
        preproc.adjust_data_size(var_low_res_scaled_dict, var_high_res_scaled_dict, BATCH_SIZE)
    var_low_res_filtered_dict = preproc.filter_dict(var_low_res_adjusted_dict, var_high_res_adjusted_dict)

    postproc = postprocess.PostProcess()
    postproc.plot_input_data(\
        var_low_res_filtered_dict[varname_predictand_high_res[0]], \
        var_high_res_scaled_dict[varname_predictand_high_res[0]], path_figure)

    dataset_train, dataset_valid, X_train, X_test, y_train, y_test = \
        preproc.split_data(var_low_res_filtered_dict, var_high_res_scaled_dict, \
        BATCH_SIZE, TEST_SIZE, RANDOM_STATE, \
        varname_predictand_high_res[0])
    #feature_selection.stepwise_algorithm(X_train, X_test, y_train, y_test)


    trainmodel = train.TrainModel(wdir)
    generator = trainmodel.training(BATCH_SIZE, EPOCH_INIT, EPOCHS, dataset_train, dataset_valid)
    pr_pred = trainmodel.prediction(generator, X_test, y_test)

    postproc.plot_result(pr_pred, X_test, y_test, path_figure)
    sys.exit()


if __name__ == "__main__":
    main()

