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
    print ('var', type(cdict['stats_conf']), cdict['stats_conf'])

    batch_size = cdict['stats_conf']['SRGAN']['batch_size']
    EPOCHS = cdict['stats_conf']['SRGAN']['EPOCHS']
    EPOCH_INIT = cdict['stats_conf']['SRGAN']['EPOCH_INIT']
    TEST_SIZE = cdict['stats_conf']['SRGAN']['TEST_SIZE']
    RANDOM_STATE = cdict['stats_conf']['SRGAN']['RANDOM_STATE']
    print(batch_size, EPOCHS, EPOCH_INIT, TEST_SIZE, RANDOM_STATE)

    path_main = '/nobackup/rossby26/users/sm_fuxwa/AI/'
    wdir = path_main + '/SRGAN_OUT/' + 'EPOCH' + str(EPOCHS) + '/'
    path_figure = path_main + '/Figure/' + 'EPOCH' + str(EPOCHS) + '/'


    dir_lr = path_main + '12km/6hr/' 
    dir_hr = path_main + '3km/6hr/'

    var_list_lr = ['phi500', 'phi700', 'phi850', 'phi950', \
	'hus500', 'hus700', 'hus850', 'hus950', \
	'ta500', 'ta700', 'ta850', 'ta950', \
	'ua500', 'ua700', 'ua850', 'ua950', \
	'va500', 'va700', 'va850', 'va950', 'pr']
    var_list_hr = ['pr']
    file_filter = 'hr_2000' # 'July'
    variable = var_list_hr[0]

    checkdir.checkdir(path_figure)
    checkdir.checkdir(wdir)

    '''
    preproc = preprocess.PreProcess()
    var_lr_scaled = preproc.scale_var(var_lr)
    var_hr_scaled = preproc.scale_var(var_hr)
    var_lr_filtered = preproc.filter_var(var_lr_scaled, var_hr_scaled)
    dataset_train, dataset_valid, X_train, X_test, y_train, y_test = preproc.split_data(var_lr_filtered, var_hr_scaled, batch_size)
    '''
    readin = read.Read('netcdf')
    var_lr_dict = readin.read_netcdf(dir_lr, var_list_lr, '12km', file_filter)
    var_hr_dict = readin.read_netcdf(dir_hr, var_list_hr, '3km',  file_filter)

    preproc = preprocess.PreProcess()
    var_lr_scaled_dict = preproc.scale_dict(var_lr_dict)
    var_hr_scaled_dict = preproc.scale_dict(var_hr_dict)
    var_lr_adjusted_dict, var_hr_adjusted_dict = preproc.adjust_data_size(var_lr_scaled_dict, var_hr_scaled_dict, batch_size)
    var_lr_filtered_dict = preproc.filter_dict(var_lr_adjusted_dict, var_hr_adjusted_dict)

    postproc = postprocess.PostProcess()
    postproc.plot_input_data(var_lr_filtered_dict[variable], var_hr_scaled_dict[variable], path_figure)
    sys.exit()

    dataset_train, dataset_valid, X_train, X_test, y_train, y_test = \
        preproc.split_data(var_lr_filtered_dict, var_hr_scaled_dict, batch_size, TEST_SIZE, RANDOM_STATE, variable)
    #feature_selection.stepwise_algorithm(X_train, X_test, y_train, y_test)


    trainmodel = train.TrainModel(wdir)
    generator = trainmodel.training(batch_size, EPOCH_INIT, EPOCHS, dataset_train, dataset_valid)
    pr_pred = trainmodel.prediction(generator, X_test, y_test)


    #postproc = postprocess.PostProcess()
    #postproc.plot_input_data(variable_low_res_gen[variable], variable_high_res[variable], path_figure)
    postproc.plot_result(pr_pred, X_test, y_test, path_figure)



if __name__ == "__main__":
    main()

