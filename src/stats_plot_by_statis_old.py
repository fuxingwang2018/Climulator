import os
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from scipy.stats import wasserstein_distance

# Function to calculate statistics
def calculate_statistics(reference, comparison):
    rmse = np.sqrt(((reference - comparison) ** 2).mean(axis=2))
    correlation = np.array([
        pearsonr(reference[..., i].flatten(), comparison[..., i].flatten())[0]
        for i in range(reference.shape[2])
    ]).reshape(reference.shape[:2])
    mean_bias = (comparison - reference).mean(axis=2)
    ratio_of_variance = (comparison.var(axis=2) / reference.var(axis=2)) * 100
    wasserstein_dist = np.array([
        wasserstein_distance(reference[..., i].flatten(), comparison[..., i].flatten())
        for i in range(reference.shape[2])
    ]).reshape(reference.shape[:2])
    
    return rmse, correlation, mean_bias, ratio_of_variance, wasserstein_dist

# Function to calculate 99th percentile and mean value
def calculate_percentile_and_mean(data):
    percentile_99 = np.percentile(data, 99, axis=2)
    mean_value = data.mean(axis=2)
    return percentile_99, mean_value

# Function to plot statistics
def plot_statistics(statistics, titles, output_dir, filename_prefix):
    num_stats = len(statistics)
    for stat_idx, stat in enumerate(statistics):
        fig, axs = plt.subplots(1, len(stat), figsize=(20, 5))
        vmin, vmax = None, None
        if '99th Percentile' in titles[stat_idx] or 'Mean Value' in titles[stat_idx]:
            vmin = min([s.min() for s in stat])
            vmax = max([s.max() for s in stat])
        for exp_idx, data in enumerate(stat):
            im = axs[exp_idx].imshow(data, origin="lower", cmap="coolwarm", vmin=vmin, vmax=vmax)
            axs[exp_idx].set_title(f"{titles[stat_idx]} Experiment {exp_idx + 1}")
            axs[exp_idx].axis('off')
            cbar = plt.colorbar(im, ax=axs[exp_idx], shrink=0.8)
        output_file = os.path.join(output_dir, f"{filename_prefix}_{titles[stat_idx].replace(' ', '_')}.png")
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()

# Main code
#reference_file = "path/to/reference_file.nc"
#comparison_files = ["path/to/comparison1.nc", "path/to/comparison2.nc", "path/to/comparison3.nc"]
#output_dir = "path/to/output_folder"
#os.makedirs(output_dir, exist_ok=True)

reference_folder = '/nobackup/rossby26/users/sm_fuxwa/AI/SRGAN_OUT/EPOCH100_ps_mrso'
reference_file = "predictant_ytest.nc"
comparison_files = ["/nobackup/rossby26/users/sm_fuxwa/AI/SRGAN_OUT/EPOCH100_ps_mrso/predictant_ypred.nc", \
                    "/nobackup/rossby26/users/sm_fuxwa/AI/SRGAN_OUT/EPOCH100_cape_ps_tas/predictant_ypred.nc", \
                    "/nobackup/rossby26/users/sm_fuxwa/AI/SRGAN_OUT/EPOCH100_no_orog/predictant_ypred.nc", \
                    "/nobackup/rossby26/users/sm_fuxwa/AI/SRGAN_OUT/EPOCH100_no_pr/predictant_ypred.nc", \
                    "/nobackup/rossby26/users/sm_fuxwa/AI/SRGAN_OUT/EPOCH100_with_pr/predictant_ypred.nc", \
                   ]
output_dir = "/nobackup/rossby26/users/sm_fuxwa/AI/SRGAN_OUT/statistic_figs"  # Update to desired output folder
os.makedirs(output_dir, exist_ok=True)


# Load reference data
reference_ds = xr.open_dataset(reference_folder + '/' + reference_file)
reference_data = reference_ds["pr"].values * 86400  # Convert mm/s to mm/day

# Initialize arrays for all statistics
rmse_maps = []
correlation_maps = []
mean_bias_maps = []
ratio_of_variance_maps = []
wasserstein_distance_maps = []
percentile_99_maps_ref = []
percentile_99_maps_comp = []
mean_value_maps_ref = []
mean_value_maps_comp = []

# Calculate statistics for each comparison
for comparison_file in comparison_files:
    comparison_ds = xr.open_dataset(comparison_file)
    comparison_data = comparison_ds["pr"].values * 86400  # Convert mm/s to mm/day

    rmse, correlation, mean_bias, ratio_of_variance, wasserstein_dist = calculate_statistics(
        reference_data, comparison_data
    )

    percentile_99_ref, mean_value_ref = calculate_percentile_and_mean(reference_data)
    percentile_99_comp, mean_value_comp = calculate_percentile_and_mean(comparison_data)

    rmse_maps.append(rmse)
    correlation_maps.append(correlation)
    mean_bias_maps.append(mean_bias)
    ratio_of_variance_maps.append(ratio_of_variance)
    wasserstein_distance_maps.append(wasserstein_dist)
    percentile_99_maps_ref.append(percentile_99_ref)
    percentile_99_maps_comp.append(percentile_99_comp)
    mean_value_maps_ref.append(mean_value_ref)
    mean_value_maps_comp.append(mean_value_comp)

# Combine statistics for all experiments
statistics = [
    rmse_maps,
    correlation_maps,
    mean_bias_maps,
    ratio_of_variance_maps,
    wasserstein_distance_maps,
    percentile_99_maps_ref + percentile_99_maps_comp,
    mean_value_maps_ref + mean_value_maps_comp
]

titles = [
    "RMSE Map",
    "Correlation Map",
    "Mean Bias Map",
    "Ratio of Variance Map",
    "Wasserstein Distance Map",
    "99th Percentile Map",
    "Mean Value Map"
]

# Plot and save each statistic
for stat_idx, stat_title in enumerate(titles):
    plot_statistics([statistics[stat_idx]], [stat_title], output_dir, f"{stat_title.replace(' ', '_')}")

