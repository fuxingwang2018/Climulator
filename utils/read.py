import numpy as np
import netCDF4
import glob
import sys
import numpy as np

# 3km:  tas, pr 
# 12km: ta500,  ta700,  ta850,  ta950, 
#	hus500, hus700, hus850, hus950, 
#	ua500,  ua700,  ua850,  ua950, 
#	va500,  va700,  va850,  va950, 
#	phi500, phi700, phi850, phi950,
# var = 'ta500' 
# exp_name = '12km' # '3km', '12km'

""" 
inputs: phi500, phi700, phi850, phi950, hus500, hus700, hus850, hus950,  ta500, ta700, ta850, ta950, ua500, ua700, ua850, ua950, va500, va700, va850, va950
outputs: pr 
"""

class Read(object):
    """ Read data """

    def __init__(self, filetype):

        self.filetype = filetype

    def read_netcdf(self, dir_nc, var_list, file_filter):

        nc_files_dict = {} 
        nc_files = glob.glob(dir_nc + '*')
        nc_files.sort()
        print('All NetCDF files:', len(nc_files), nc_files)

        for varname in var_list:
            for ifile in nc_files:
                if varname in ifile and file_filter in ifile:
                    nc_files_dict[varname] = ifile

        print('NetCDF files to read:', len(nc_files_dict), nc_files_dict)

        var_dict = {}
        var_list_nc = []

        for ivar, ifile in nc_files_dict.items():
            data = netCDF4.Dataset(ifile)
            var = list(data.variables.keys())[-1]
            var_dict[ivar] = np.array(data.variables[var])
            var_list_nc.append(var)
        print('Variables Read:', var_list_nc)

        #var_nc = np.array(var_dict['pr'])

        #print('type of var_nc:', type(var_nc))
        #print('shape of var_nc:', var_nc.shape)

        return var_dict


