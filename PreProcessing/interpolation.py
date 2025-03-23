import xarray as xr
import numpy as np
from scipy.interpolate import griddata
import yaml
import sys
import argparse
import os

def interpolate_netcdf(variable, input_nc, target_nc):
    # Open input NetCDF file (Low resolution)
    ds_lowres = xr.open_dataset(input_nc)
    
    # Open target NetCDF file (High resolution) to get target grid
    ds_highres = xr.open_dataset(target_nc)
    
    # Get coordinates
    #lon_lowres, lat_lowres = np.meshgrid(ds_lowres.lon, ds_lowres.lat)
    #lon_highres, lat_highres = np.meshgrid(ds_highres.lon, ds_highres.lat)
    lon_lowres, lat_lowres   = ds_lowres.lon.values, ds_lowres.lat.values
    lon_highres, lat_highres = ds_highres.lon.values, ds_highres.lat.values
    
    ## Prepare output dataset
    #ds_interp = xr.Dataset()
    print('ds_lowres.data_vars:', ds_lowres.data_vars)
   
    # Flatten the original grid
    points = np.column_stack([lon_lowres.ravel(), lat_lowres.ravel()])

    # Create a new dataset based on the 3km grid
    ds_interp = ds_highres.copy()  # Copy the original 3km dataset to retain metadata

    # Copy global attributes
    ds_interp.attrs = ds_highres.attrs

    ds_interp = ds_interp.drop_vars([variable], errors='ignore')  # Remove old precipitation if exists
 
    # Interpolate each variable
    for var in ds_lowres.data_vars:
      if var == variable:
        data_lowres = ds_lowres[var].values.reshape(ds_lowres[var].shape[0], -1) # Reshape for each time step
        print('var', var, lon_lowres.shape, lat_lowres.shape, data_lowres.shape)
        data_highres = np.array([\
            griddata(points, \
            data_lowres[t], \
            (lon_highres.ravel(), lat_highres.ravel()), \
            method='linear', \
            fill_value=np.nan)\
            .reshape(lon_highres.shape) \
            for t in range(ds_lowres[var].shape[0]) \
        ])

        """
        # Fill missing values using nearest neighbor interpolation
        #for t in range(ds_lowres[var].shape[0]):
        for t in range(10):
            mask = np.isnan(data_highres[t])
            if np.any(mask):
                data_highres[t][mask] = \
                    griddata(points, \
                    data_lowres[t], \
                    (lon_highres[mask], lat_highres[mask]), \
                    method='nearest')
        """

        ds_interp[var] = (('time', 'lat', 'lon'), data_highres)
        ds_interp[var].attrs = ds_highres[var].attrs  # Preserve attributes
    
    ## Assign interpolated precipitation
    #ds_interp[var] = (('time', 'y', 'x'), data_highres)
    
    return ds_interp


def calculate_difference(variable, interpolated_nc, reference_nc):
    # Open datasets
    ds_interp = xr.open_dataset(interpolated_nc)
    ds_ref = xr.open_dataset(reference_nc)
   
    # Prepare output dataset
    #ds_diff = xr.Dataset()

    # Create a new dataset based on the 3km grid
    ds_diff = ds_ref.copy()  # Copy the original 3km dataset to retain metadata

    # Copy global attributes
    ds_diff.attrs = ds_ref.attrs
    
    # Calculate difference for each variable
    for var in ds_interp.data_vars:
        #if var in ds_ref:
        if var == variable:
            ds_diff = ds_diff.drop_vars([var], errors='ignore')  # Remove old precipitation if exists
            print('var', var) 
            ds_diff[var] = (('time', 'lat', 'lon'), np.array(ds_interp[var].values - ds_ref[var].values))
            ds_diff[var].attrs = ds_ref[var].attrs  # Preserve attributes
    
    return ds_diff


def save_netcdf(ds, output_file):
    """Saves an xarray dataset to a NetCDF file, creating the directory if necessary."""
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    ds.to_netcdf(output_file)


# Load the YAML configuration file
def load_config(config_path):
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
    return config


def get_config(config_path):
    config = load_config(config_path)

    # Resolve environment variables or placeholders like ${base_dir}
    for key, value in config["data_path"].items():
        config["data_path"][key] = \
            value.replace("${base_dir}", config["base_dir"]).\
            replace("${variable}", config["variable"]).\
            replace("${period}", config["period"])

    return config


def main():

    # Set up argument parser to accept config file as a command-line argument
    parser = argparse.ArgumentParser(description="Run statistical analysis with a given config file.")
    parser.add_argument("-c", "--config", required=True, help="Path to the YAML config file")
    args = parser.parse_args()

    # get config
    config = get_config(args.config)

    input_nc = config['data_path']['input_lowres_nc']
    target_nc = config['data_path']['input_highres_nc']
    output_interp_nc = config['data_path']['output_interp_nc']
    output_diff_nc = config['data_path']['output_diff_nc']
    print('get config completed!')

    ds_interp = interpolate_netcdf(config["variable"], input_nc, target_nc)
    print('interpolation completed!')
    save_netcdf(ds_interp, output_interp_nc)
    print(f"Interpolated data saved to {output_interp_nc}")

    ds_diff = calculate_difference(config["variable"], output_interp_nc, target_nc)
    print('calculation of difference completed!')
    save_netcdf(ds_diff, output_diff_nc)
    print(f"Difference data saved to {output_diff_nc}")

if __name__ == "__main__":
    main()
