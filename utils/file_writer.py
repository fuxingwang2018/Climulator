
# coding=utf-8
# -*- coding:utf-8 -*-
#from __future__ import unicode_literals
import datetime as dt  # Python standard library datetime  module
from netCDF4 import Dataset  # http://code.google.com/p/netcdf4-python/
import os, io, sys
from utils import ncdump as nd
import pandas as pd
import numpy as np
#import yaml
#from importlib import reload
#reload(sys)
#sys.setdefaultencoding('utf-8')


# Author:  Fuxing Wang, 30 June, 2019

class FileWriter(object):
    """Reads a file"""

    def __init__(self, file_out):
        self.file_out = file_out


    def Write_Txt(self, var_dict):
        try:
            with open(self.file_out, 'w') as outFile:
                for key_lev1, value_lev1 in var_dict.items(): 
                    if type(value_lev1) is dict:
                        outFile.write('%s\n' % (key_lev1))
                        for key_lev2, value_lev2 in value_lev1.items(): 
                            outFile.write('%s:%s\n' % (key_lev2, value_lev2))
                    else:
                        outFile.write('%s:%s\n' % (key_lev1, value_lev1))

        except IndexError:
            print("Error - Please specify an output file.")
            sys.exit(2)

    """
    def Write_Yaml(self, var_pd, name):

        if not os.path.isfile(self.file_out):
            mode = "w"
        else:
            mode = "a"

        try:
            with io.open(self.file_out, mode, encoding="utf-8") as file:
            #with io.open(self.file_out, mode) as file:
                if type(var_pd) is dict:
                    #documents = yaml.dump({str(name): var_pd}, file, default_flow_style=False)
                    documents = yaml.dump(var_pd, file, default_flow_style=False)
                elif isinstance(var_pd, pd.DataFrame):
                    #documents = yaml.dump({str(name): var_pd.to_dict(orient='records')}, file, default_flow_style=False)
                    documents = yaml.dump({str(name): var_pd.to_dict()}, file, default_flow_style=False)
                else:
                    print('No such type defined')
                    sys.exit(1)

        except IndexError:
            print("Error - Please specify an output file.")
            sys.exit(2)

    """

    def Write_NC(self, nc_file_2D, nc_var_to_read, nc_vars_to_write, time_offset, time_length, geo_offset, var_2D_dict):

        nam_lon = 'lon'
        nam_lat = 'lat'

        #
        # Read 2D netcdf file
        #
        nc_file_2D_id = Dataset(nc_file_2D, 'r') # Dataset is the class behavior to open the file, and create an instance of the ncCDF4 class 
        nc_attrs_2d, nc_dims_2d, nc_vars_2d = nd.ncdump(nc_file_2D_id)

        # Extract data from NetCDF file
        #lon_2d = nc_file_2D_id.variables['longitude'][:]  # extract/copy the data
        #lat_2d = nc_file_2D_id.variables['latitude'][:]
        x_start, x_end = 0, 100000
        y_start, y_end = 0, 100000
        if 'x' in nc_file_2D_id.variables and 'y' in nc_file_2D_id.variables:
            if geo_offset['x'] != 0:
                x_start, x_end = int(geo_offset['x'] / 2), int(geo_offset['x'] / 2 * (-1))
            if geo_offset['y'] != 0:
                y_start, y_end = int(geo_offset['y'] / 2), int(geo_offset['y'] / 2 * (-1))
            x = nc_file_2D_id.variables['x'][x_start: x_end]
            y = nc_file_2D_id.variables['y'][y_start: y_end] 
        print('x_start, x_end', x_start, x_end)
        print('y_start, y_end', y_start, y_end)
        #time_2d = nc_file_2D_id.variables['time'][time_offset*(-1):]
        time_start = time_offset*(-1) + time_length*(-1)
        time_end = time_offset*(-1)
        time_2d = nc_file_2D_id.variables['time'][time_start:time_end]

        # Open a new NetCDF file to write the data to. 
        # Choose format from 'NETCDF3_CLASSIC', 'NETCDF3_64BIT', 'NETCDF4_CLASSIC', and 'NETCDF4'
        w_nc_file_out_id = Dataset(self.file_out, 'w', format='NETCDF4')
        w_nc_file_out_id.description = "" 
        w_nc_file_out_id.set_fill_off()

        # Using our previous dimension info, we can create the new time dimension
        # Even though we know the size, we are going to set the size to unknown
        data_dim = {}
        for dim in nc_dims_2d:
            print ('dim:', dim)
            w_nc_file_out_id.createDimension(dim, None)
            if dim in nc_file_2D_id.variables:
                data_dim[dim] = w_nc_file_out_id.createVariable(dim, nc_file_2D_id.variables[dim].dtype,\
                                   (dim,),  fill_value=-9999.9) 
                # You can do this step yourself but someone else did the work for us.
                for ncattr in nc_file_2D_id.variables[dim].ncattrs():
                    print('ncattr:', type(ncattr), ncattr)
                    if str(ncattr) != '_FillValue':
                        data_dim[dim].setncattr(ncattr, nc_file_2D_id.variables[dim].getncattr(ncattr))


        # Assign the dimension data to the new NetCDF file.
        w_nc_file_out_id.variables['time'][:] = time_2d
        if 'x' in nc_file_2D_id.variables and 'y' in nc_file_2D_id.variables:
            w_nc_file_out_id.variables['y'][:] = y
            w_nc_file_out_id.variables['x'][:] = x

        # Time varied variables
        data_var={}

        # Constant variable
        for var in nc_vars_2d:
            if nam_lon in var or nam_lat in var:
                #print('nc_file_2D_id.variables[var].dtype', var, nc_file_2D_id.variables[var].dtype)
                data_var[var] = w_nc_file_out_id.createVariable(var, nc_file_2D_id.variables[var].dtype,\
                    nc_file_2D_id.variables[var].dimensions, fill_value=-9999.9)
                for ncattr in nc_file_2D_id.variables[var].ncattrs():
                    if str(ncattr) != '_FillValue':
                        data_var[var].setncattr(ncattr, nc_file_2D_id.variables[var].getncattr(ncattr))
                    #print(var, np.shape(nc_file_2D_id.variables[var][:]))
                    #print(var, np.shape(nc_file_2D_id.variables[var][1:-1, :]))
                    w_nc_file_out_id.variables[var][:] = nc_file_2D_id.variables[var][y_start:y_end, x_start:x_end]
                    #w_nc_file_out_id.variables[var][:] = nc_file_2D_id.variables[var][:]

        for var in nc_vars_to_write:
            if var != 'time' and var != 'Projection_Type' and var != 'FRC_TIME_STP':

                #if nc_file_1D_id.variables[var].dimensions == ('time', 'Number_of_points'):
                    # Convert variables from 1D to 2D
                    #var_2D = np.reshape(var_1D, (len(time_1d), len(y), len(x))) # Shape is time, y, x
                    #print 'shape of var_2D:', np.shape(var_2D)

                # Create variable
                #print('nc_file_2D_id.variables[nc_var_to_read].dtype', nc_var_to_read, nc_file_2D_id.variables[nc_var_to_read].dtype)
                #data_var[var] = w_nc_file_out_id.createVariable(var, nc_file_2D_id.variables[nc_var_to_read].dtype, \
                data_var[var] = w_nc_file_out_id.createVariable(var, 'float64', \
                 ('time', 'y', 'x'), fill_value=-9999.9) 	

                # Attributes:
                for ncattr in nc_file_2D_id.variables[nc_var_to_read].ncattrs():
                    if str(ncattr) != '_FillValue':
                        data_var[var].setncattr(ncattr, nc_file_2D_id.variables[nc_var_to_read].getncattr(ncattr))

                # Assign values to variables
                #print ('shape of var_2D, min, max', var, var_2D_dict[var].shape, np.nanmin(var_2D_dict[var]), np.nanmax(var_2D_dict[var]))
                w_nc_file_out_id.variables[var][:] = var_2D_dict[var]
        #
        # Close NetCDF files.
        #
        w_nc_file_out_id.close()  

        #return var_out

"""
mask_threshold = {'HCLIM3': {'town': 0.1, 'water': 0.02, 'sea': 0.02, 'nature': 0.98}, 'HCLIM12': {'town': 0.1, 'water': 0.1, 'sea': 0.1, 'nature': 0.9}, 'SURFEX': {'town': 0.4, 'water': 0.02, 'sea': 0.02, 'nature': 0.98}}
test_statistic_dict = {'HCLIM3_EV3_PGW3_minus_HCLIM3_EV3_PGW1': 2.220011443866698, 'HCLIM12_EV3_PGW3_minus_HCLIM12_EV3_PGW1': 0.7030740939047835, 'SURFEX_NorrLink_EV3_PGW2_minus_SURFEX_NorrLink_EV3_PGW1': 0.11060852696284741, 'HCLIM12_EV3_PGW2_minus_HCLIM12_EV3_PGW1': -0.5986331478238309, 'SURFEX_NorrLink_EV3_PGW3_minus_SURFEX_NorrLink_EV3_PGW1': 1.70411833125291, 'HCLIM3_EV3_PGW2_minus_HCLIM3_EV3_PGW1': 0.8172074946873915}
probability_dict = {'HCLIM3_EV3_PGW3_minus_HCLIM3_EV3_PGW1': 0.027652699482056578, 'HCLIM12_EV3_PGW3_minus_HCLIM12_EV3_PGW1': 0.48290795018694865, 'SURFEX_NorrLink_EV3_PGW2_minus_SURFEX_NorrLink_EV3_PGW1': 0.9120486616046025, 'HCLIM12_EV3_PGW2_minus_HCLIM12_EV3_PGW1': 0.5501618660702905, 'SURFEX_NorrLink_EV3_PGW3_minus_SURFEX_NorrLink_EV3_PGW1': 0.09006559531069627, 'HCLIM3_EV3_PGW2_minus_HCLIM3_EV3_PGW1': 0.41487833243732797}
 
uhi_delta_ttest = {}
uhi_delta_ttest['test_statistic'], uhi_delta_ttest['probability'] = test_statistic_dict, probability_dict

test_dict = {}
test_dict['uhi_delta_ttest'] = uhi_delta_ttest
test_dict['mask_threshold'] = mask_threshold
 
#test_dict = {'uhi_delta_ttest': {'test_statistic': {'hclim3_ev3_pgw3_minus_hclim3_ev3_pgw1': 2.220011443866698, 'hclim12_ev3_pgw3_minus_hclim12_ev3_pgw1': 0.7030740939047835, 'surfex_norrlink_ev3_pgw2_minus_surfex_norrlink_ev3_pgw1': 0.11060852696284741, 'hclim12_ev3_pgw2_minus_hclim12_ev3_pgw1': -0.5986331478238309, 'surfex_norrlink_ev3_pgw3_minus_surfex_norrlink_ev3_pgw1': 1.70411833125291, 'hclim3_ev3_pgw2_minus_hclim3_ev3_pgw1': 0.8172074946873915}, 'probability': {'hclim3_ev3_pgw3_minus_hclim3_ev3_pgw1': 0.027652699482056578, 'hclim12_ev3_pgw3_minus_hclim12_ev3_pgw1': 0.48290795018694865, 'surfex_norrlink_ev3_pgw2_minus_surfex_norrlink_ev3_pgw1': 0.9120486616046025, 'hclim12_ev3_pgw2_minus_hclim12_ev3_pgw1': 0.5501618660702905, 'surfex_norrlink_ev3_pgw3_minus_surfex_norrlink_ev3_pgw1': 0.09006559531069627, 'hclim3_ev3_pgw2_minus_hclim3_ev3_pgw1': 0.41487833243732797}}, 'mask_threshold': {'hclim3': {'town': 0.1, 'water': 0.02, 'sea': 0.02, 'nature': 0.98}, 'hclim12': {'town': 0.1, 'water': 0.1, 'sea': 0.1, 'nature': 0.9}, 'surfex': {'town': 0.4, 'water': 0.02, 'sea': 0.02, 'nature': 0.98}}}

file_out = 'test.yaml'
#filewriter = FileWriter(file_out)
#filewriter.Write_Yaml(test_dict, ['exp_name'])
"""
