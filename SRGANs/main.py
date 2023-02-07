import read
import preprocess
import train
import postprocess
import checkdir
import feature_selection
import sys

batch_size = 50
EPOCHS = 4 #100
EPOCH_INIT = 0 #0

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

def main():

    checkdir.checkdir(path_figure)
    checkdir.checkdir(wdir)

    '''
    readin = read.Read('netcdf')
    var_lr = readin.read_netcdf_old(dir_lr, var_list_lr, '12km', file_filter)
    var_hr = readin.read_netcdf_old(dir_hr, var_list_hr, '3km',  file_filter)

    preproc = preprocess.PreProcess()
    var_lr_scaled = preproc.scale_var(var_lr)
    var_hr_scaled = preproc.scale_var(var_hr)
    var_lr_filtered = preproc.filter_var(var_lr_scaled, var_hr_scaled, path_figure)
    dataset_train, dataset_valid, X_train, X_test, y_train, y_test = preproc.split_data(var_lr_filtered, var_hr_scaled, batch_size)

    '''
    readin = read.Read('netcdf')
    var_lr_dict = readin.read_netcdf(dir_lr, var_list_lr, '12km', file_filter)
    var_hr_dict = readin.read_netcdf(dir_hr, var_list_hr, '3km',  file_filter)


    preproc = preprocess.PreProcess()
    var_lr_scaled_dict = preproc.scale_dict(var_lr_dict)
    var_hr_scaled_dict = preproc.scale_dict(var_hr_dict)
    var_lr_filtered_dict = preproc.filter_dict(var_lr_scaled_dict, var_hr_scaled_dict, path_figure)
    #sys.exit()
    variable = 'pr'
    dataset_train, dataset_valid, X_train, X_test, y_train, y_test = \
        preproc.split_data(var_lr_filtered_dict, var_hr_scaled_dict, batch_size, variable)

    #feature_selection.stepwise_algorithm(X_train, X_test, y_train, y_test)

    trainmodel = train.TrainModel(wdir)
    generator = trainmodel.training(batch_size, EPOCH_INIT, EPOCHS, dataset_train, dataset_valid)
    #sys.exit()
    pr_pred = trainmodel.prediction(generator, X_test, y_test)

    postproc = postprocess.PostProcess(pr_pred, X_test, y_test)
    postproc.plot_result(path_figure)

if __name__ == "__main__":
    main()

