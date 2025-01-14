import os
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from scipy.stats import wasserstein_distance
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.ndimage import zoom

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


var_name = 'pr' #'tas' #pr
unit_convert = {'pr': 86400.0, 'tas': 1.0}

# Load reference and comparison data
#reference_file = 'path/to/reference_file.nc'
#comparison_files = ['path/to/comparison_file1.nc', 'path/to/comparison_file2.nc', 
#                    'path/to/comparison_file3.nc', 'path/to/comparison_file4.nc', 
#                    'path/to/comparison_file5.nc']
#output_dir = 'path/to/output_dir'
base_dir = '/nobackup/rossby26/users/sm_fuxwa/AI/Emilia_Romagna/'
#base_dir = '/nobackup/rossby26/users/sm_fuxwa/AI/'
#reference_folder = str(base_dir) + '/SRGAN_OUT/EPOCH100_ps_mrso'
reference_folder = str(base_dir) + f'/SRGAN_OUT/EPOCH100_{var_name}_no_pr'
reference_file = "predictant_ytest.nc"


reference_lowres_folder = str(base_dir) + f'/SRGAN_OUT/EPOCH1_{var_name}_with_pr'
reference_lowres_file = "predictor.nc"

comparison_files = [\
                    str(base_dir) + f'/SRGAN_OUT/EPOCH100_{var_name}_no_pr/predictant_ypred.nc', \
                   ]
"""
reference_lowres_folder = str(base_dir) + '/SRGAN_OUT/EPOCH100_with_pr'
reference_lowres_file = "predictor.nc"
comparison_files = [\
                    str(base_dir) + "/SRGAN_OUT/EPOCH100_ps_mrso/predictant_ypred.nc", \
                    str(base_dir) + "/SRGAN_OUT/EPOCH100_cape_ps_tas/predictant_ypred.nc", \
                    str(base_dir) + "/SRGAN_OUT/EPOCH100_no_orog/predictant_ypred.nc", \
                    str(base_dir) + "/SRGAN_OUT/EPOCH100_no_pr/predictant_ypred.nc", \
                    str(base_dir) + "/SRGAN_OUT/EPOCH100_with_pr/predictant_ypred.nc", \
                   ]
"""

output_dir = str(base_dir) + "/SRGAN_OUT/statistic_figs"  # Update to desired output folder
os.makedirs(output_dir, exist_ok=True)

reference_ds = xr.open_dataset(reference_folder + '/' + reference_file)
reference_precip = reference_ds[var_name].values * unit_convert[var_name]  # Convert to mm/day for pr

reference_lowres_ds = xr.open_dataset(reference_lowres_folder + '/' + reference_lowres_file)
reference_lowres_var = reference_lowres_ds[var_name].values * unit_convert[var_name]  # Convert to mm/day for pr
reference_low2highres_precip = upsample_2d_array(reference_lowres_var, upscale_factor = 4)

comparison_precip = []
for file in comparison_files:
    ds = xr.open_dataset(file)
    comparison_precip.append(ds[var_name].values * unit_convert[var_name])  # Convert to mm/day for pr

comparison_precip.append(reference_low2highres_precip)

comparison_precip = np.array(comparison_precip)

# Calculate statistics
rmse_map = [calculate_rmse(reference_precip, comparison) for comparison in comparison_precip]
mean_bias_map = [calculate_mean_bias(reference_precip, comparison) for comparison in comparison_precip]
ratio_of_variance_map = [calculate_ratio_of_variance(reference_precip, comparison) for comparison in comparison_precip]
correlation_map = [calculate_correlation(reference_precip, comparison) for comparison in comparison_precip]
wasserstein_map = [calculate_wasserstein_distance(reference_precip, comparison) for comparison in comparison_precip]
percentile_99_ref = calculate_99th_percentile(reference_precip)
percentile_99_comparisons = [calculate_99th_percentile(comparison) for comparison in comparison_precip]
mean_value_ref = calculate_mean_value(reference_precip)
mean_value_comparisons = [calculate_mean_value(comparison) for comparison in comparison_precip]

# Find global color scale ranges for all statistics
rmse_all = rmse_map
mean_bias_all = mean_bias_map
ratio_of_variance_all = ratio_of_variance_map
correlation_all = correlation_map
wasserstein_all = wasserstein_map

rmse_vmin, rmse_vmax = np.min(rmse_all), np.max(rmse_all)
mean_bias_vmin, mean_bias_vmax = np.min(mean_bias_all), np.max(mean_bias_all)
ratio_of_variance_vmin, ratio_of_variance_vmax = np.min(ratio_of_variance_all), np.max(ratio_of_variance_all)
correlation_vmin, correlation_vmax = np.min(correlation_all), np.max(correlation_all)
wasserstein_vmin, wasserstein_vmax = np.min(wasserstein_all), np.max(wasserstein_all)
percentile_99_all = [percentile_99_ref] + percentile_99_comparisons
mean_value_all = [mean_value_ref] + mean_value_comparisons
percentile_99_vmin, percentile_99_vmax = np.min(percentile_99_all), np.max(percentile_99_all)
mean_value_vmin, mean_value_vmax = np.min(mean_value_all), np.max(mean_value_all)

# Define function to plot and save maps

def plot_and_save_maps(statistics, titles, output_file, vmin=None, vmax=None, cmap='coolwarm'):
    fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(15, 15))
    axes = axes.flatten()

    for i, (stat, title) in enumerate(zip(statistics, titles)):
        im = axes[i].imshow(stat, cmap=cmap, vmin=vmin, vmax=vmax)
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


# Prepare statistics for plotting
all_statistics = [
    (rmse_map, 'RMSE Maps', 'rmse_maps', rmse_vmin, rmse_vmax, 'viridis'),
    (correlation_map, 'Correlation Maps', 'correlation_maps', correlation_vmin, correlation_vmax, 'coolwarm'),
    (mean_bias_map, 'Mean Bias Maps', 'mean_bias_maps', mean_bias_vmin, mean_bias_vmax, 'seismic'),
    (ratio_of_variance_map, 'Ratio of Variance Maps', 'ratio_of_variance_maps', ratio_of_variance_vmin, ratio_of_variance_vmax, 'YlGnBu'),
    (wasserstein_map, 'Wasserstein Distance Maps', 'wasserstein_maps', wasserstein_vmin, wasserstein_vmax, 'plasma'),
    ([percentile_99_ref] + percentile_99_comparisons, '99th Percentile Maps', 'percentile_99_maps', percentile_99_vmin, percentile_99_vmax, 'inferno'),
    ([mean_value_ref] + mean_value_comparisons, 'Mean Value Maps', 'mean_value_maps', mean_value_vmin, mean_value_vmax, 'cividis')
]


# Plot and save each statistics set
for stats, title, filename, vmin, vmax, cmap in all_statistics:
    output_path = os.path.join(output_dir, f'{filename}_{var_name}.png')
    plot_and_save_maps(stats, [f'{title} {i+1}' for i in range(len(stats))], output_path, vmin=vmin, vmax=vmax, cmap=cmap)


