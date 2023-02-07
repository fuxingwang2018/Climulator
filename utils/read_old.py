






    def read_netcdf_old(self, dir_nc, var_list, resolution, file_filter):


        nc_files = glob.glob(dir_nc + '*')
        nc_files.sort()
        print('All NetCDF files:', len(nc_files), nc_files)

        if '12km' in dir_nc:
            nc_files = nc_files[16:17]
        elif '3km' in dir_nc:
            nc_files = nc_files[0:1]
        ### nc_files.pop(12)

        var_nc = []
        var_list_nc = []
        for file in nc_files:
            data = netCDF4.Dataset(file)
            var = list(data.variables.keys())[-1]
            var_nc.append(data.variables[var])
            var_list_nc.append(var)

        var_nc = np.array(var_nc)

        print('type of var_nc:', type(var_nc))
        print('shape of var_nc:', var_nc.shape)

        return var_nc

