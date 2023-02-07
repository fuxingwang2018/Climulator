
#%matplotlib inline
import matplotlib.pyplot as plt
plt.switch_backend('agg')
from map_plot import MapPlot

fig_out_path = '/nobackup/rossby26/users/sm_fuxwa/AI/Figure/'
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
var = 'tas' 
exp_name = '3km' # '3km', '12km'

def plot_test(lat_2d, lon_2d, var_2d, var_to_plot):

    if exp_name == '12km':
        width_def = 16E5
        height_def = 8E5
        lat_0_def = 46.0
        lon_0_def = 14.0
    elif exp_name == '3km':
        width_def = 16E5
        height_def = 8E5
        lat_0_def = 45.5
        lon_0_def = 16.0

    if 'tas' in var:
        scale_min_def = 275
        scale_max_def = 290
        unit = 'K'
    elif var == 'ta950_fp':
        scale_min_def = 2.75e2
        scale_max_def = 2.9e2
        unit = 'K'

    title_def = var_to_plot + '(' + unit + ')' 
    fig_out =  str(fig_out_path) + exp_name + '_' + var_to_plot + fig_type

    map_plot = MapPlot(fig_out, proj_def, res_def, width_def, height_def, lat_0_def, lon_0_def)
    map_plot.Plot_2DField(lat_2d, lon_2d, var_2d, scale_min_def, scale_max_def, title_def, label_def, cmap_def, extend_def)

