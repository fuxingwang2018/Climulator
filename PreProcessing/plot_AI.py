
#%matplotlib inline
import matplotlib.pyplot as plt
import netCDF4
plt.switch_backend('agg')
from map_plot import MapPlot
import numpy as np

data_folder = '/nobackup/rossby26/users/sm_fuxwa/AI'
fig_out_path = '../../Figures/AI/'

# 3km:  tas, pr 
# 12km: ta500,  ta700,  ta850,  ta950, 
#	hus500, hus700, hus850, hus950, 
#	ua500,  ua700,  ua850,  ua950, 
#	va500,  va700,  va850,  va950, 
#	phi500, phi700, phi850, phi950,
#       tas, pr
var = 'tas' 
exp_name = '3km' # '3km', '12km'

if exp_name == '12km':
    filein = data_folder + '/' + str(var) + '_12km_199901_200912.nc'
    width_def = 16E5
    height_def = 8E5
    lat_0_def = 46.0
    lon_0_def = 14.0
elif exp_name == '3km':
    filein = data_folder + '/' + str(var) + '_3km_199901_200912.nc'
    width_def = 16E5
    height_def = 8E5
    lat_0_def = 45.5
    lon_0_def = 16.0

fig_title = " "
lat_name = 'lat' #'latitude'
lon_name = 'lon' #'longitude'
proj_def = 'lcc'
res_def = 'i'
fig_type = '.png'
label_def = ''
extend_def = 'max' #'min', 'max', 'neither', 'both'
cmap_def = 'rainbow'
#cmap_def = 'RdBu_r'
if 'tas' in var:
    variable_list = ['tas']
    scale_min_def = 275
    scale_max_def = 290
    unit = 'K'
elif var == 'ta950_fp':
    variable_list = ['ta950']
    scale_min_def = 2.75e2
    scale_max_def = 2.9e2
    unit = 'K'

nc = netCDF4.Dataset(filein)
lat_sim_2d = nc.variables[lat_name][:,:]
lon_sim_2d = nc.variables[lon_name][:,:]
#print nc.variables.keys()

# examine the variables
for var_to_plot in variable_list:

	# average over all steps
        var_sim_3d = nc.variables[var_to_plot][:8771,:,:]
        var_sim_2d = np.nanmean(var_sim_3d, axis=0)
    
        title_def = var_to_plot + '(' + unit + ')' 
        fig_out =  str(fig_out_path) + exp_name + '_' + var_to_plot + fig_type

        map_plot = MapPlot(fig_out, proj_def, res_def, width_def, height_def, lat_0_def, lon_0_def)
        map_plot.Plot_2DField(lat_sim_2d, lon_sim_2d, var_sim_2d[:,:], scale_min_def, scale_max_def, title_def, label_def, cmap_def, extend_def)
        #map_plot.Plot_ortho(lat_sim_2d, lon_sim_2d, title_def)

