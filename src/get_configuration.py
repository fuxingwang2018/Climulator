
import sys
import os
sys.path.insert(0, '..')
from utils import ini_reader

def get_args():
    """
    Read configuration file
    Args:

    Returns:
    """
    import argparse

    # Configuring argument setup and handling
    parser = argparse.ArgumentParser(
        description='Main script for model/obs validation')
    parser.add_argument('--config', '-c',  metavar='name config file',
                        type=str, help='<Required> Full path to config file',
                        required=True)
    return parser.parse_args()


def get_settings(config_file):
    """
    Retrieve information from main configuration file
    """
    conf_dict = ini_reader.get_config_dict(config_file)
    d = {
        'downscale mode': conf_dict['SETTINGS']['downscale mode'],
        'variables': conf_dict['SETTINGS']['variables'],
        #'regions': conf_dict['SETTINGS']['regions'],
        'coordinates_of_detection_area': conf_dict['SETTINGS']['coordinates_of_detection_area'], 
        'time_range': conf_dict['SETTINGS']['time_range'], 
        'requested_stats': conf_dict['STATISTICS']['stats'],
        'stats_conf': mod_stats_config(conf_dict['STATISTICS']['stats']),
        'path_main': conf_dict['SETTINGS']['path_main'],
        'path_x': conf_dict['SETTINGS']['path_x'],
        'path_y': conf_dict['SETTINGS']['path_y'],
        'file_filter': conf_dict['SETTINGS']['file_filter'],
        'experiment_name': conf_dict['SETTINGS']['experiment_name'],
    }

    return d


def default_stats_config(stats):
    """
    Get default statistics configurations of stats

    :param stats: A list of statistics
    :type stats: List of strings
    :return: A dictionary with default statistics configurations for a selection of statistics given by input stats
    :rtype: dictionary
    """
    stats_dict = {
        'SRGAN': {
            'batch_size': 50, 
            'EPOCHS': 100, 
            'EPOCH_INIT': 0, 
            'TEST_SIZE': 1460,
            'VALIDATION_SPLIT': 0.1,
            'RANDOM_STATE': 24}, 
        'TRAINING': {
            'SUBSAMPLING_LR': 4, 
            'N_RES_BLOCK': 8, 
            'NX': 104,
            'NY': 88,
            'LEARNING_RATE': {'GENERATOR':1e-4, 'DISCRIMINATOR':2e-4},
            'DROPOUT_RATE': -1, 
            'EARLY_STOP':False, 
            'DATA_AUGMENTATION':False, 
            'USE_SEED':False, 
            'DISABLE_PARALLEL':False, 
            'ENFORCE_DETERMINISM':False},
        'RUNNING_MODE': {
            'TRAINING_MODE': True, 
            'PREDICTION_MODE': True, 
            'PRETRAINED_PATH': ''},
        'PREPROCESS': {
            'SCALER_DEF': 'StandardScaler',
            'SCALER_HUS': 'PowerTransformer',
            'SCALE_METHOD': 'scale_over_time'}, 
        'LOSS': {
            'WEIGHT_ADVERSARIAL': 1e-5,
            'WEIGHT_CONTENT': 1.0,
            'WEIGHT_CROSS_ENTROPY': 0.2,
            'LAMBDA_CORR': 0.0, 
            'CONTENT_LOSS_FUNC': 'MSE',
            'LOSSES_REDUCTION': 'NONE',
            'LABEL_SMOOTHING_REAL': 1.0,
            'DISCRIMINATOR_NOISE_STD': 0.03,
            'DISCRIMINATOR_LABEL_NOISE': 0.03,
            }, 
            }
        #'INPUT_CHANNELS': 1, 
        #'OUTPUT_CHANNELS': 1, 

    return {k: stats_dict[k] for k in stats}


def mod_stats_config(requested_stats):
    """
    Get the configuration for the input statistics 'requested_stats'.

    :param stats: A list of statistics
    :type stats: List of strings
    :return: A dictionary with modified statistics configurations for input requested_stats
    :rtype: dictionary
    """
    print('requested_stats', type(requested_stats), requested_stats)
    print('list keys', list(requested_stats.keys()))
    stats_dd = default_stats_config(list(requested_stats.keys()))

    # Update dictionary based on input
    for k in requested_stats:
        if requested_stats[k] == 'default':
            pass
        else:
            for m in requested_stats[k]:
                msg = "For statistic {}, the configuration key {} is not "\
                        "available. Check possible configurations  in "\
                        "default_stats_config in stats_template "\
                        "module.".format(k, m)
                try:
                    stats_dd[k][m] = requested_stats[k][m]
                except KeyError:
                    print(msg)

    return stats_dd

