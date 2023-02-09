
#%matplotlib inline
import matplotlib.pyplot as plt
import netCDF4
plt.switch_backend('agg')
from map_plot import MapPlot
import numpy as np

data_folder = '/nobackup/rossby26/users/sm_fuxwa/RCAT_OUT/HCLIMAI/precip/stats/annual_cycle'
fig_out_path = '/home/sm_fuxwa/Figures/AI/'

# 3km:  tas, pr 
# 12km: ta500,  ta700,  ta850,  ta950, 
#	hus500, hus700, hus850, hus950, 
#	ua500,  ua700,  ua850,  ua950, 
#	va500,  va700,  va850,  va950, 
#	phi500, phi700, phi850, phi950,
#       tas, pr
var = 'tas' #'pr' 
exp_name = 'EOBS20' # 'EOBS20', '3km', '12km'
month = 1
month_dic = {	'1':'January',
		'2':'February',
		'3':'March',
		'4':'April',
		'5':'May',
		'6':'June',
		'7':'July',
		'8':'August',
		'9':'September',
		'10':'October',
		'11':'November',
		'12':'December'		}

filein = data_folder + '/' + str(exp_name) + '_annual_cycle_' + str(var) + '_1hr_mean_native_grid_2000-2009_ANN.nc'
#filein = data_folder + '/' + str(exp_name) + '_annual_cycle_' + str(var) + '_1hr_mean_grid_12km_2000-2009_ANN.nc'
#width_def = 12E5
#height_def = 8E5
#lat_0_def = 46.0
#lon_0_def = 11.0

width_def =  4E5
height_def = 3E5
lat_0_def = 46.6
lon_0_def = 11.0
#if exp_name == '12km':
#    width_def = 12E5
#    height_def = 8E5
#    lat_0_def = 46.0
#    lon_0_def = 11.0 #14.0
#elif exp_name == '3km':
#    width_def = 12E5
#    height_def = 8E5
#    lat_0_def = 46.0 #45.5
#    lon_0_def = 11.0 #16.0


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
    unit = 'K'
    extend_def = 'both'
    if month == 1:
        scale_min_def = 255
        scale_max_def = 285
    elif month == 7:
        scale_min_def = 275
        scale_max_def = 305
elif var == 'pr':
    variable_list = ['pr']
    scale_min_def = 0.0
    scale_max_def = 8.0
    unit = 'mm/day'

nc = netCDF4.Dataset(filein)
lat_sim = nc.variables[lat_name][:]
lon_sim = nc.variables[lon_name][:]
if np.array(lat_sim).ndim == 1 and np.array(lon_sim).ndim == 1:  
    lon_sim_2d, lat_sim_2d = np.meshgrid(lon_sim, lat_sim)
elif np.array(lat_sim).ndim == 2 and np.array(lon_sim).ndim == 2:  
    lon_sim_2d = lon_sim
    lat_sim_2d = lat_sim
#print nc.variables.keys()

# examine the variables
for var_to_plot in variable_list:

        # average over all steps
        var_sim_3d = nc.variables[var_to_plot][:,:,:]
        var_sim_2d = var_sim_3d[month - 1,:,:] #np.nanmean(var_sim_3d, axis=0)
    
        title_def = var_to_plot + '(' + unit + ')' 
        fig_out =  str(fig_out_path) + exp_name + '_' + var_to_plot + '_' + month_dic[str(month)] + fig_type

        map_plot = MapPlot(fig_out, proj_def, res_def, width_def, height_def, lat_0_def, lon_0_def)
        map_plot.Plot_2DField(lat_sim_2d, lon_sim_2d, var_sim_2d[:,:], scale_min_def, scale_max_def, title_def, label_def, cmap_def, extend_def)
        #map_plot.Plot_ortho(lat_sim_2d, lon_sim_2d, title_def)

