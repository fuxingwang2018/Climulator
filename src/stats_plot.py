import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, wasserstein_distance
from mpl_toolkits.axes_grid1 import make_axes_locatable
import os

# Statistics based on https://link.springer.com/article/10.1007/s00382-022-06343-9

# Define a function to calculate RMSE
def calculate_rmse(reference, comparison):
    return np.sqrt(((comparison - reference) ** 2).mean(axis=0))

# Define a function to calculate Pearson correlation coefficient
def calculate_correlation(reference, comparison):
    correlation_map = np.empty(reference.shape[1:])
    for i in range(reference.shape[1]):
        for j in range(reference.shape[2]):
            ref_series = reference[:, i, j]
            comp_series = comparison[:, i, j]
            if np.any(np.isnan(ref_series)) or np.any(np.isnan(comp_series)):
                correlation_map[i, j] = np.nan
            else:
                correlation_map[i, j], _ = pearsonr(ref_series, comp_series)
    return correlation_map

# Define a function to calculate mean bias
def calculate_mean_bias(reference, comparison):
    return (comparison - reference).mean(axis=0)

# Define a function to calculate Ratio of Variance
def calculate_ratio_of_variance(reference, comparison):
    ratio_map = np.var(comparison, axis=0) / np.var(reference, axis=0) * 100
    ratio_map[np.isnan(ratio_map)] = np.nan  # Handle NaN values explicitly
    return ratio_map

# Define a function to calculate Wasserstein Distance
def calculate_wasserstein_distance(reference, comparison):
    wd_map = np.empty(reference.shape[1:])
    for i in range(reference.shape[1]):
        for j in range(reference.shape[2]):
            ref_series = reference[:, i, j]
            comp_series = comparison[:, i, j]
            if np.any(np.isnan(ref_series)) or np.any(np.isnan(comp_series)):
                wd_map[i, j] = np.nan
            else:
                wd_map[i, j] = wasserstein_distance(ref_series, comp_series)
    return wd_map

# Define a function to calculate the 99th Percentile
def calculate_99th_percentile(data):
    return np.percentile(data, 99, axis=0)


# Define a function to calculate the Mean Value
def calculate_mean_value(data):
    return data.mean(axis=0)

# Define file paths
reference_folder = '/nobackup/rossby26/users/sm_fuxwa/AI/SRGAN_OUT/EPOCH100_ps_mrso'
reference_file = "predictant_ytest.nc"
comparison_files = ["/nobackup/rossby26/users/sm_fuxwa/AI/SRGAN_OUT/EPOCH100_ps_mrso/predictant_ypred.nc", \
                    "/nobackup/rossby26/users/sm_fuxwa/AI/SRGAN_OUT/EPOCH100_cape_ps_tas/predictant_ypred.nc", \
                    "/nobackup/rossby26/users/sm_fuxwa/AI/SRGAN_OUT/EPOCH100_no_orog/predictant_ypred.nc", \
                    "/nobackup/rossby26/users/sm_fuxwa/AI/SRGAN_OUT/EPOCH100_no_pr/predictant_ypred.nc", \
                    "/nobackup/rossby26/users/sm_fuxwa/AI/SRGAN_OUT/EPOCH100_with_pr/predictant_ypred.nc", \
                   ]
output_folder = "/nobackup/rossby26/users/sm_fuxwa/AI/SRGAN_OUT/statistic_figs"  # Update to desired output folder
os.makedirs(output_folder, exist_ok=True)

# Convert precipitation unit from mm/s to mm/d
mms_to_mmd = 86400.0

# Load the reference NetCDF file
reference_ds = xr.open_dataset(reference_folder + '/' + reference_file)
reference_precip = reference_ds["pr"] * mms_to_mmd # Adjust variable name if necessary

# Loop through comparison files and compute statistics
for comparison_file in comparison_files:
    comparison_ds = xr.open_dataset(comparison_file)
    comparison_precip = comparison_ds["pr"] * mms_to_mmd # Adjust variable name if necessary

    # Ensure dimensions are aligned
    reference_precip, comparison_precip = xr.align(reference_precip, comparison_precip)

    # Convert precipitation units from mm/s to mm/day
    #reference_precip *= 86400 #mms_to_mmd
    #comparison_precip *= 86400 #mms_to_mmd

    # Compute statistics
    rmse_map = calculate_rmse(reference_precip.values, comparison_precip.values)
    correlation_map = calculate_correlation(reference_precip.values, comparison_precip.values)
    mean_bias_map = calculate_mean_bias(reference_precip.values, comparison_precip.values)
    ratio_of_variance_map = calculate_ratio_of_variance(reference_precip.values, comparison_precip.values)
    wasserstein_distance_map = calculate_wasserstein_distance(reference_precip.values, comparison_precip.values)
    reference_99th_percentile_map = calculate_99th_percentile(reference_precip.values)
    comparison_99th_percentile_map = calculate_99th_percentile(comparison_precip.values)
    reference_mean_value_map = calculate_mean_value(reference_precip.values)
    comparison_mean_value_map = calculate_mean_value(comparison_precip.values)

    # Determine common scales for specific statistics
    percentile_vmin = min(reference_99th_percentile_map.min(), comparison_99th_percentile_map.min())
    percentile_vmax = max(reference_99th_percentile_map.max(), comparison_99th_percentile_map.max())

    mean_value_vmin = min(reference_mean_value_map.min(), comparison_mean_value_map.min())
    mean_value_vmax = max(reference_mean_value_map.max(), comparison_mean_value_map.max())

    # Plot statistics
    fig, axes = plt.subplots(3, 4, figsize=(20, 15))
    axes = axes.flatten()

    def add_colorbar(im, ax):
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(im, cax=cax)

    im1 = axes[0].imshow(rmse_map, origin='lower', cmap='viridis')
    axes[0].set_title('RMSE (mm/day)')
    add_colorbar(im1, axes[0])

    im2 = axes[1].imshow(correlation_map, origin='lower', cmap='plasma', vmin=0, vmax=1)
    axes[1].set_title('Correlation Coefficient')
    add_colorbar(im2, axes[1])

    im3 = axes[2].imshow(mean_bias_map, origin='lower', cmap='RdBu', vmin=-np.abs(mean_bias_map).max(), vmax=np.abs(mean_bias_map).max())
    axes[2].set_title('Mean Bias (mm/day)')
    add_colorbar(im3, axes[2])

    im4 = axes[3].imshow(ratio_of_variance_map, origin='lower', cmap='cividis', vmin=0, vmax=200)
    axes[3].set_title('Ratio of Variance (%)')
    add_colorbar(im4, axes[3])

    im5 = axes[4].imshow(wasserstein_distance_map, origin='lower', cmap='magma')
    axes[4].set_title('Wasserstein Distance')
    add_colorbar(im5, axes[4])

    im6 = axes[5].imshow(reference_99th_percentile_map, origin='lower', cmap='coolwarm', vmin=percentile_vmin, vmax=percentile_vmax)
    axes[5].set_title('99th Percentile (Reference) (mm/day)')
    add_colorbar(im6, axes[5])

    im7 = axes[6].imshow(comparison_99th_percentile_map, origin='lower', cmap='coolwarm', vmin=percentile_vmin, vmax=percentile_vmax)
    axes[6].set_title('99th Percentile (Comparison) (mm/day)')
    add_colorbar(im7, axes[6])

    im8 = axes[7].imshow(reference_mean_value_map, origin='lower', cmap='YlGnBu', vmin=mean_value_vmin, vmax=mean_value_vmax)
    axes[7].set_title('Mean Value (Reference) (mm/day)')
    add_colorbar(im8, axes[7])

    im9 = axes[8].imshow(comparison_mean_value_map, origin='lower', cmap='YlGnBu', vmin=mean_value_vmin, vmax=mean_value_vmax)
    axes[8].set_title('Mean Value (Comparison) (mm/day)')
    add_colorbar(im9, axes[8])


    """
    im1 = axes[0].imshow(rmse_map, origin='lower', cmap='viridis')
    axes[0].set_title('RMSE (mm/d)')
    plt.colorbar(im1, ax=axes[0])

    im2 = axes[1].imshow(correlation_map, origin='lower', cmap='YlGnBu', vmin=0, vmax=1)
    axes[1].set_title('Correlation Coefficient')
    plt.colorbar(im2, ax=axes[1])

    im3 = axes[2].imshow(mean_bias_map, origin='lower', cmap='RdBu', vmin=-np.abs(mean_bias_map).max(), vmax=np.abs(mean_bias_map).max())
    axes[2].set_title('Mean Bias (mm/d)')
    plt.colorbar(im3, ax=axes[2])

    im4 = axes[3].imshow(ratio_of_variance_map, origin='lower', cmap='cividis', vmin=0, vmax=200)
    axes[3].set_title('Ratio of Variance (%)')
    plt.colorbar(im4, ax=axes[3])

    im5 = axes[4].imshow(wasserstein_distance_map, origin='lower', cmap='magma')
    axes[4].set_title('Wasserstein Distance')
    plt.colorbar(im5, ax=axes[4])

    im6 = axes[5].imshow(reference_99th_percentile_map, origin='lower', cmap='coolwarm')
    axes[5].set_title('99th Percentile (Reference, mm/d)')
    plt.colorbar(im6, ax=axes[5])

    im7 = axes[6].imshow(comparison_99th_percentile_map, origin='lower', cmap='coolwarm')
    axes[6].set_title('99th Percentile (Comparison, mm/d)')
    plt.colorbar(im7, ax=axes[6])

    im8 = axes[7].imshow(reference_mean_value_map, origin='lower', cmap='YlGnBu')
    axes[7].set_title('Mean Value (Reference, mm/s)')
    plt.colorbar(im8, ax=axes[7])

    im9 = axes[8].imshow(comparison_mean_value_map, origin='lower', cmap='YlGnBu')
    axes[8].set_title('Mean Value (Comparison, mm/d)')
    plt.colorbar(im9, ax=axes[8])
    """

    # Turn off the remaining subplots
    for ax in axes[9:]:
        ax.axis('off')

    plt.suptitle(f"Statistics for {os.path.basename(comparison_file)}")
    plt.tight_layout()

    # Save the plots as PNG and PDF
    #output_base = os.path.join(output_folder, os.path.splitext(os.path.basename(comparison_file))[-2])
    output_base = os.path.join(output_folder, os.path.normpath(comparison_file).split(os.sep)[-2])
    #print('os.path.splitext(os.path.basename(comparison_file))[-2]', os.path.splitext(os.path.basename(comparison_file))[-2])
    #print('os.path.splitext(os.path.basename(comparison_file))', os.path.splitext(os.path.basename(comparison_file)))
    #print('os.path.basename(comparison_file)', os.path.basename(comparison_file))
    #print('os.path.normpath(comparison_file).split(os.sep)', os.path.normpath(comparison_file).split(os.sep))
    plt.savefig(f"{output_base}_statistics.png", format="png")
    #plt.savefig(f"{output_base}_statistics.pdf", format="pdf")
    plt.close()  # Close the figure to free memory

    # Save maps if needed
    # Example: np.save(f"{output_base}_rmse_map.npy", rmse_map)

