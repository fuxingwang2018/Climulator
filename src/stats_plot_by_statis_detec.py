import os
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from scipy.stats import wasserstein_distance
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.ndimage import zoom
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
#from multiprocessing import Pool

# Define function to calculate RMSE
def calculate_rmse(reference, comparison):
    return np.sqrt(np.mean((reference - comparison)**2, axis=0))

# Define function to calculate Mean Bias
def calculate_mean_bias(reference, comparison):
    return np.mean(comparison - reference, axis=0)

# Define function to calculate Ratio of Variance
def calculate_ratio_of_variance(reference, comparison):
    reference_variance = np.var(reference, axis=0)
    comparison_variance = np.var(comparison, axis=0)
    return (comparison_variance / reference_variance) * 100

# Define function to calculate Pearson Correlation Coefficient
def calculate_correlation(reference, comparison):
    cor_map = np.zeros(reference.shape[1:])
    for i in range(reference.shape[1]):
        for j in range(reference.shape[2]):
            cor_map[i, j], _ = pearsonr(reference[:, i, j], comparison[:, i, j])
    return cor_map

# Define function to calculate Wasserstein Distance
def calculate_wasserstein_distance(reference, comparison):
    wass_map = np.zeros(reference.shape[1:])
    for i in range(reference.shape[1]):
        for j in range(reference.shape[2]):
            wass_map[i, j] = wasserstein_distance(reference[:, i, j], comparison[:, i, j])
    return wass_map

# Define function to calculate 99th Percentile
def calculate_99th_percentile(data):
    return np.percentile(data, 99, axis=0)

# Define function to calculate Mean Value
def calculate_mean_value(data):
    return np.mean(data, axis=0)

# Upscale function
def upsample_2d_array(low_res_array, upscale_factor):
    # Initialize high-resolution array
    time_records = low_res_array.shape[0]
    high_res_array = np.empty((time_records, 
                               low_res_array.shape[1] * upscale_factor, 
                               low_res_array.shape[2] * upscale_factor))
    
    # Apply zoom for each time record
    for t in range(time_records):
        high_res_array[t] = zoom(low_res_array[t], zoom=(upscale_factor, upscale_factor), order=1)
    
    return high_res_array


# Define function to plot and save maps
def plot_and_save_maps(statistics, titles, output_file, vmin=None, vmax=None, cmap='coolwarm'):
    fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(15, 15))
    axes = axes.flatten()

    for i, (stat, title) in enumerate(zip(statistics, titles)):
        print(titles)
        print('stat', stat, len(stat))
        im = axes[i].imshow(stat, cmap=cmap, vmin=vmin[i], vmax=vmax[i])
        axes[i].set_title(title, fontsize=10)
        
        divider = make_axes_locatable(axes[i])
        cax = divider.append_axes("right", size="5%", pad=0.05)
        cbar = plt.colorbar(im, cax=cax)

        # Dynamically adjust colorbar size to match the axis height
        #cbar = plt.colorbar(im, ax=axes[i], fraction=0.046, pad=0.04)
        #cbar_height = axes[i].get_position().height  # Get the height of the axis
        #cbar.ax.set_aspect(cbar_height / cbar.ax.get_position().height)

    # Hide unused subplots if there are any
    for i in range(len(statistics), len(axes)):
        axes[i].axis('off')

    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()


def compute_metrics(ground_truth, predictions_list, threshold=0.5):
    """
    Compute metrics per (latitude, longitude) for each experiment in predictions_list.
    
    Args:
        ground_truth: 3D numpy array (time, latitude, longitude) of binary ground truth (0 or 1).
        predictions_list: List of 3D numpy arrays (time, latitude, longitude) of predicted probabilities.
        threshold: Threshold to convert probabilities to binary predictions.
    
    Returns:
        A dictionary where each metric is a list of 2D arrays (latitude, longitude),
        one for each experiment.
    """
    # Initialize the metrics
    metrics = ["accuracy", "precision", "recall", "f1", "roc_auc"]
    results = {metric: [] for metric in metrics}

    # Iterate through each experiment in the predictions list
    for predictions in predictions_list:
        # Ensure shapes match
        assert ground_truth.shape == predictions.shape, "Shape mismatch between ground truth and predictions."

        # Get dimensions
        _, lat, lon = ground_truth.shape

        # Initialize 2D arrays for metrics
        accuracy_map = np.zeros((lat, lon))
        precision_map = np.zeros((lat, lon))
        recall_map = np.zeros((lat, lon))
        f1_map = np.zeros((lat, lon))
        roc_auc_map = np.zeros((lat, lon))

        # Loop over each grid point
        for i in range(lat):
            for j in range(lon):
                # Extract time-series data for the grid point
                y_true = ground_truth[:, i, j]
                y_pred_probs = predictions[:, i, j]

                # Only calculate metrics if there are positive cases in the ground truth
                if np.sum(y_true) > 0:
                    # Convert probabilities to binary predictions using the threshold
                    y_pred = (y_pred_probs >= threshold).astype(int)

                    # Compute metrics
                    accuracy_map[i, j] = accuracy_score(y_true, y_pred)
                    precision_map[i, j] = precision_score(y_true, y_pred, zero_division=0)
                    recall_map[i, j] = recall_score(y_true, y_pred, zero_division=0)
                    f1_map[i, j] = f1_score(y_true, y_pred, zero_division=0)
                    roc_auc_map[i, j] = roc_auc_score(y_true, y_pred_probs)
                else:
                    # If no positive cases, metrics are set to NaN (not applicable)
                    accuracy_map[i, j] = np.nan
                    precision_map[i, j] = np.nan
                    recall_map[i, j] = np.nan
                    f1_map[i, j] = np.nan
                    roc_auc_map[i, j] = np.nan

        # Append the results for this experiment
        results["accuracy"].append(accuracy_map)
        results["precision"].append(precision_map)
        results["recall"].append(recall_map)
        results["f1"].append(f1_map)
        results["roc_auc"].append(roc_auc_map)

    return results

def compute_metrics_vectorized(ground_truth, predictions, threshold):

    # Initialize the metrics
    metrics = ["accuracy", "precision", "recall", "f1", "roc_auc"]
    results = {metric: [] for metric in metrics}

    # Flatten time axis for ROC-AUC computation (time, lat, lon -> (time, lat*lon))
    ground_truth_flat = ground_truth.reshape(ground_truth.shape[0], -1)
    
    for prediction in predictions:
        # Convert probabilities to binary predictions using the threshold
        prediction = (prediction >= threshold).astype(int)
        prediction_flat = prediction.reshape(prediction.shape[0], -1)

        # Vectorized true/false positive/negative calculations
        tp = (prediction == 1) & (ground_truth == 1)
        tn = (prediction == 0) & (ground_truth == 0)
        fp = (prediction == 1) & (ground_truth == 0)
        fn = (prediction == 0) & (ground_truth == 1)

        # Use np.sum to quickly calculate counts
        tp_sum = np.sum(tp, axis=0)
        tn_sum = np.sum(tn, axis=0)
        fp_sum = np.sum(fp, axis=0)
        fn_sum = np.sum(fn, axis=0)

        # Compute metrics
        precision = tp_sum / (tp_sum + fp_sum + 1e-8)
        recall = tp_sum / (tp_sum + fn_sum + 1e-8)
        accuracy = (tp_sum + tn_sum) / (tp_sum + tn_sum + fp_sum + fn_sum + 1e-8)
        f1_score = 2 * (precision * recall) / (precision + recall + 1e-8)

        # Flatten the arrays for ROC-AUC computation
        #ground_truth_flat = ground_truth.ravel()
        #prediction_flat = prediction.ravel()

        # ROC-AUC calculation
        roc_auc = np.zeros((ground_truth.shape[1], ground_truth.shape[2]))
        for i in range(ground_truth.shape[1]):  # Latitude
            for j in range(ground_truth.shape[2]):  # Longitude
                try:
                    roc_auc[i, j] = roc_auc_score(ground_truth_flat[:, i * ground_truth.shape[2] + j],
                                                  prediction_flat[:, i * prediction.shape[2] + j])
                except ValueError:
                    roc_auc[i, j] = np.nan  # Handle case where only one class is present

        """
        # Compute ROC-AUC (check if ground_truth contains both classes)
        if len(np.unique(ground_truth_flat)) > 1:  # Avoid errors when there's only one class
            roc_auc = roc_auc_score(ground_truth_flat, prediction_flat)
        else:
            roc_auc = np.nan  # Not computable
        """

        # Append the results for this experiment
        results["accuracy"].append(accuracy)
        results["precision"].append(precision)
        results["recall"].append(recall)
        results["f1"].append(f1_score)
        results["roc_auc"].append(roc_auc)

    return results



def compute_single_metrics(prediction, ground_truth):
    tp = (prediction == 1) & (ground_truth == 1)
    tn = (prediction == 0) & (ground_truth == 0)
    fp = (prediction == 1) & (ground_truth == 0)
    fn = (prediction == 0) & (ground_truth == 1)

    tp_sum = np.sum(tp, axis=0)
    tn_sum = np.sum(tn, axis=0)
    fp_sum = np.sum(fp, axis=0)
    fn_sum = np.sum(fn, axis=0)

    precision = tp_sum / (tp_sum + fp_sum + 1e-8)
    recall = tp_sum / (tp_sum + fn_sum + 1e-8)
    accuracy = (tp_sum + tn_sum) / (tp_sum + tn_sum + fp_sum + fn_sum + 1e-8)
    f1_score = 2 * (precision * recall) / (precision + recall + 1e-8)

    return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1_score": f1_score}

#with Pool() as pool:
#    metrics_list = pool.starmap(compute_single_metrics, [(pred, ground_truth) for pred in predictions])


extreme_threshold = 0.2
experiment = 'extreme_detection'
var_name = 'pr' #'tas' #pr
var_name_mask = 'prdmax_mask' #'tas' #pr
unit_convert = {'pr': 1.0, 'tas': 1.0}
unit_convert_ref = {'pr': 86400.0, 'tas': 1.0}

# Load reference and comparison data
base_dir = '/nobackup/rossby27/users/sm_fuxwa/Extreme_Detection/'
reference_folder = str(base_dir) + '/SRGAN_OUT/' + 'EPOCH50_6var' + '/'
reference_file = "predictant_ytest.nc"

comp_experiment = ['EPOCH100_6var',  'EPOCH100_6var_BS10', 'EPOCH100_5var', 'EPOCH100_1var', 'EPOCH50_6var', 'HCLIM12']
comp_experiment_list = { \
    'Accuracy Maps': comp_experiment, \
    'Precision Maps': comp_experiment, \
    'Recall Maps': comp_experiment, \
    'F1 Maps': comp_experiment, \
    'roc_auc Maps': comp_experiment, \
    }


reference_lowres_folder = str(base_dir) + '/input_data/CORDEXFPS_HCLIM12/12km/day/prdmax_mask/' 
reference_lowres_file = "prdmax_mask_alp-12_regrid_3km_hclim_ec-earth_his_moments_D_max_native_grid_1996-2005_ANN.nc"

comparison_files = [\
                    str(base_dir) + '/SRGAN_OUT/' + 'EPOCH100_6var' + '/predictant_ypred.nc', \
                    str(base_dir) + '/SRGAN_OUT/' + 'EPOCH100_5var_BS10' + '/predictant_ypred.nc', \
                    str(base_dir) + '/SRGAN_OUT/' + 'EPOCH100_5var' + '/predictant_ypred.nc', \
                    str(base_dir) + '/SRGAN_OUT/' + 'EPOCH100_1var' + '/predictant_ypred.nc', \
                    str(base_dir) + '/SRGAN_OUT/' + 'EPOCH50_6var' + '/predictant_ypred.nc', \
                   ]

output_dir = str(base_dir) + "/Figure/" + 'thres_' + str(extreme_threshold)  # Update to desired output folder
os.makedirs(output_dir, exist_ok=True)

reference_ds = xr.open_dataset(reference_folder + '/' + reference_file)
reference_var = reference_ds[var_name].values * unit_convert[var_name]  # Convert to mm/day for pr

reference_lowres_ds = xr.open_dataset(reference_lowres_folder + '/' + reference_lowres_file, decode_cf=False)
reference_lowres_var = reference_lowres_ds[var_name_mask].values * unit_convert[var_name]  # Convert to mm/day for pr
reference_low2highres = upsample_2d_array(reference_lowres_var, upscale_factor = 4)

comparison_var = []
for file in comparison_files:
    ds = xr.open_dataset(file)
    comparison_var.append(ds[var_name].values * unit_convert[var_name])  # Convert to mm/day for pr

comparison_var.append(reference_low2highres[-753:-53,:,:])
print('comparison_var.shape:', comparison_var[0].shape, comparison_var[1].shape, comparison_var[2].shape)
print('comparison_var type:', type(comparison_var[0]))

comparison_var = np.array(comparison_var)

# Compute metrics
#results = compute_metrics(reference_var, comparison_var, threshold=extreme_threshold)
results = compute_metrics_vectorized(reference_var, comparison_var, threshold=extreme_threshold)
print('results', results)


# Access results for each metric
accuracy = results["accuracy"]  # List of 2D arrays (latitude, longitude) for accuracy
precision = results["precision"]  # List of 2D arrays (latitude, longitude) for precision
recall = results["recall"]       # List of 2D arrays for recall
f1 = results["f1"]               # List of 2D arrays for F1 score
roc_auc = results["roc_auc"]     # List of 2D arrays for ROC-AUC

accuracy_vmin, accuracy_vmax = [np.min(accuracy)] * len(accuracy), [np.max(accuracy)] * len(accuracy)
precision_vmin, precision_vmax = [np.min(precision)] * len(precision), [np.max(precision)] * len(precision)
recall_vmin, recall_vmax = [np.min(recall)] * len(recall), [np.max(recall)] * len(recall)
f1_vmin, f1_vmax = [np.min(f1)] * len(f1), [np.max(f1)] * len(f1)
roc_auc_vmin, roc_auc_vmax = [np.min(roc_auc)] * len(roc_auc), [np.max(roc_auc)] * len(roc_auc)

print('accuracy', accuracy, type(accuracy))


all_statistics = [
    (accuracy, 'Accuracy Maps', 'accuracy_maps', accuracy_vmin, accuracy_vmax, 'viridis'),
    (precision, 'Precision Maps', 'precision_maps', precision_vmin, precision_vmax, 'coolwarm'),
    (recall, 'Recall Maps', 'recall_maps', recall_vmin, recall_vmax, 'seismic'),
    (f1, 'F1 Maps', 'f1_maps', f1_vmin, f1_vmax, 'YlGnBu'),
    (roc_auc, 'roc_auc Maps', 'roc_auc_maps', roc_auc_vmin, roc_auc_vmax, 'plasma'),
]

# Plot and save each statistics set
for stats, title, filename, vmin, vmax, cmap in all_statistics:
    output_path = os.path.join(output_dir, f'{filename}_{experiment}_thres_{extreme_threshold}_{var_name}.png')
    plot_and_save_maps(stats, [f'{title} {comp_experiment_list[title][i]}' for i in range(len(stats))], output_path, vmin=vmin, vmax=vmax, cmap=cmap)


