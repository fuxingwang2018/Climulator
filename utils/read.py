import numpy as np
import netCDF4
import glob, re
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


    def read_netcdf_one_var_per_file(self, dir_nc, var_list, file_filter, time_idx_range = {'start_idx':0, 'end_idx':0}):

        pattern = re.compile(file_filter)
        nc_files_dict = {} 

        for varname in var_list:
            #print('varname:', varname)
            #print('dir_nc:', dir_nc)
            #nc_files_all = []
            #nc_files_all = glob.glob(dir_nc + '/' + str(varname) + '/' + '*')
            nc_files_list = []
            for item in dir_nc:
                nc_files_all = glob.glob(item + '/' + str(varname) + '/' + '*')
                nc_files_all.sort()
                #print('All NetCDF files:', len(nc_files_all), nc_files_all)
                for ifile in nc_files_all:
                    #print('ifile:', ifile)
                    if varname in ifile and pattern.search(ifile): #file_filter in ifile:
                        nc_files_list.append(ifile)
            nc_files_dict[varname] = nc_files_list
        #nc_files_dict['time'] = nc_files_dict[var_list[0]]

        print('NetCDF files to read:', len(nc_files_dict), nc_files_dict)

        var_dict = {}
        var_list_nc = []
        for ivar, nc_files in nc_files_dict.items():
            icount = 0
            #print('ivar, nc_files:', ivar, nc_files)
            for ifile in nc_files:
                data = netCDF4.Dataset(ifile)
                #var = list(data.variables.keys())[-1]
                var_data_ifile = np.array(data.variables[ivar])
                if var_data_ifile.ndim > 2:
                    var_data_ifile_cut = var_data_ifile[time_idx_range['start_idx'][icount]: time_idx_range['end_idx'][icount]]
                else:
                    var_data_ifile_cut = var_data_ifile

                if icount == 0:
                    #var_data = np.array(data.variables[ivar])
                    var_data = var_data_ifile_cut
                else:
                    if var_data_ifile.ndim > 2:
                        #var_data_ifile = np.array(data.variables[ivar])
                        var_data = np.concatenate((var_data, var_data_ifile_cut), axis = 0)
                    else:
                        #var_data = np.stack([var_data, var_data_ifile_cut], axis = 0)
                        var_data = var_data_ifile_cut

                icount += 1
                data.close()
            #if time_idx_range is not None:
            #if var_data.ndim > 2:
            #    var_dict[ivar] = var_data[time_idx_range['start_idx']: time_idx_range['end_idx']] #np.array(data.variables[var])
            #else:
            #var_data[var_data > 1e5] = np.nan
            var_dict[ivar] = var_data

            var_list_nc.append(ivar)
            print('shape var_dict[ivar]', np.shape(var_dict[ivar]))
        print('Variables Read:', var_list_nc)
        #print('var_dict time:', var_dict['time'])

        #var_nc = np.array(var_dict['pr'])

        #print('type of var_nc:', type(var_nc))
        #print('shape of var_nc:', var_nc.shape)

        return var_dict


    def read_netcdf_multivar_singlefile(self, dir_nc, var_list, file_filter, time_idx_range = {'start_idx':0, 'end_idx':0}):

        print('dir_nc =', dir_nc)
        nc_files_list = []
        for item in dir_nc:
            nc_files_all = glob.glob(item + '/' + '*')
            nc_files_all.sort()
            print('All NetCDF files:', len(nc_files_all), nc_files_all)
            for ifile in nc_files_all:
                #if varname in ifile and file_filter in ifile:
                if file_filter in ifile:
                    nc_files_list.append(ifile)
        print('NetCDF files to read:', len(nc_files_list), nc_files_list)

        var_dict = {}
        var_list_nc = []
        for ivar in var_list:
            icount = 0
            for ifile in nc_files_list:
                data = netCDF4.Dataset(ifile)
                ##var = list(data.variables.keys())[-1]
                #if icount == 0:
                #    var_data = np.array(data.variables[ivar])
                #else:
                #    var_data_ifile = np.array(data.variables[ivar])
                #    var_data = np.concatenate((var_data, var_data_ifile), axis = 0)


                var_data_ifile = np.array(data.variables[ivar])
                if var_data_ifile.ndim > 2:
                    var_data_ifile_cut = var_data_ifile[time_idx_range['start_idx'][icount]: time_idx_range['end_idx'][icount]]
                else:
                    var_data_ifile_cut = var_data_ifile

                if icount == 0:
                    #var_data = np.array(data.variables[ivar])
                    var_data = var_data_ifile_cut
                else:
                    if var_data_ifile.ndim > 2:
                        #var_data_ifile = np.array(data.variables[ivar])
                        var_data = np.concatenate((var_data, var_data_ifile_cut), axis = 0)
                    else:
                        #var_data = np.stack([var_data, var_data_ifile_cut], axis = 0)
                        var_data = var_data_ifile_cut
                icount += 1
                data.close()

            #if var_data.ndim > 2:
            #    var_dict[ivar] = var_data[time_idx_range['start_idx']: time_idx_range['end_idx']] #np.array(data.variables[var])
            #else:
            #    var_dict[ivar] = var_data
            var_list_nc.append(ivar)
            var_dict[ivar] = var_data

        print('shape var_dict[ivar]', np.shape(var_dict[ivar]))
        print('Variables Read:', var_list_nc)

        #var_nc = np.array(var_dict['pr'])
        #print('type of var_nc:', type(var_nc))
        #print('shape of var_nc:', var_nc.shape)

        return var_dict


    def read_ascii(self, file_in):
        try:
            with open(file_in, 'r') as inFile:
                lines = inFile.readlines()
                x = [line.split()[0] for line in lines]
                y = [line.split()[1] for line in lines]
                return x, y

        except IndexError:
            print("Error - Please specify an input file.")
            sys.exit(2)


#reader = Read('ascii')
#x, y = reader.read_ascii('/nobackup/rossby26/users/sm_fuxwa/AI/SRGAN_OUT/EPOCH1/checkpoint')
#print('x',x)
#print('y',y)
#print('y0=', y[0].split('_')[0].strip('"'))
#print('int y0=', int(y[0].split('_')[1].strip('"')))
