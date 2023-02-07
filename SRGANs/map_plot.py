
#%matplotlib inline
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap

# Author:  Fuxing Wang, 30 June, 2019
# Reference: https://jakevdp.github.io/PythonDataScienceHandbook/04.13-geographic-data-with-basemap.html

class MapPlot(object):
    def __init__(self, fig_name, proj_def, res_def, 
		width_def, height_def, lat_0_def, lon_0_def):
        self.fig_name = fig_name
        self.proj_def = proj_def
        self.res_def = res_def
        self.width_def = width_def
        self.height_def = height_def
        self.lat_0_def = lat_0_def
        self.lon_0_def = lon_0_def
        self.fig = plt.figure(figsize=(8, 8))
        self.Plot_Basemap()

    def Plot_Basemap(self):
        # Draw the map background

        if self.proj_def == 'ortho':
            m = Basemap(projection = self.proj_def, resolution = self.res_def, 
                lat_0 = self.lat_0_def, lon_0 = self.lon_0_def)
            m.bluemarble(scale=0.5);

        elif self.proj_def == 'lcc':
            m = Basemap(projection = self.proj_def, resolution = self.res_def,
                width = self.width_def, height = self.height_def, 
                lat_0 = self.lat_0_def, lon_0 = self.lon_0_def,)
            #m.etopo(scale=0.5, alpha=0.5)  # the green background
            #m.bluemarble(scale=0.5)

        elif self.proj_def == 'cyl':
            m = Basemap(projection = self.proj_def, resolution = self.res_def,
                llcrnrlat = self.lat_0_def - 1.0, urcrnrlat = self.lat_0_def + 1.0,
                llcrnrlon = self.lon_0_def - 1.0, urcrnrlon = self.lon_0_def + 1.0,)
            #m.etopo(scale=0.5, alpha=0.5)

        #m.shadedrelief()
        #m.shadedrelief(scale=0.5) # the green background
        m.drawcoastlines(color='gray')
        m.drawcountries(color='gray')
        m.drawstates(color='gray')
        self.m = m

    def Plot_2DField(self, lat, lon, var2d_to_plot, scale_min_def, scale_max_def, title_def, label_def, cmap_def, extend_def):


        #var2d_to_plot_mask = np.ma.masked_where(np.isnan(var2d_to_plot),var2d_to_plot)

        self.m.pcolormesh(lon, lat, var2d_to_plot,
            latlon=True, cmap=cmap_def, vmin=scale_min_def, vmax=scale_max_def)

        #plt.clim(scale_min_def, scale_max_def)

        self.m.drawcoastlines(color='lightgray')

        plt.title(title_def, fontsize=20)
        plt.colorbar(label=label_def, shrink=0.8, extend=extend_def)

        self.fig.tight_layout()
        self.fig.savefig(self.fig_name)

    def Plot_ortho(self, lat, lon, title_def):

        self.m.drawcoastlines(color='lightgray')
        self.m.drawmapboundary(fill_color='aqua')
        self.m.fillcontinents(color='coral',lake_color='aqua')
        # draw parallels and meridians.
        self.m.drawparallels(np.arange(-90.,120.,30.), linewidth=0.5)
        self.m.drawmeridians(np.arange(0.,360.,60.), linewidth=0.5)
        self.m.drawmapboundary()
        plt.title(title_def, fontsize=20)
        self.fig.tight_layout()
        self.fig.savefig(self.fig_name)

