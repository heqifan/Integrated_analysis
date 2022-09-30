# -- coding: utf-8 --
# import mpl_scatter_density
# import numpy as np
# # Fake data for testingpip ins
# x = np.random.normal(size=100000)
# y = x * 3 + np.random.normal(size=100000)
# from matplotlib.colors import LinearSegmentedColormap
# import matplotlib.pyplot as plt
# import matplotlib as mpl
# # # "Viridis-like" colormap with white background
# # white_viridis = LinearSegmentedColormap.from_list('white_viridis', [
# #     (0, '#ffffff'),
# #     (1e-20, '#440053'),
# #     (0.2, '#404388'),
# #     (0.4, '#2a788e'),
# #     (0.6, '#21a784'),
# #     (0.8, '#78d151'),
# #     (1, '#fde624'),
# # ], N=256)
# #
# # def using_mpl_scatter_density(fig, x, y):
# #     ax = fig.add_subplot(1, 1, 1, projection='scatter_density')
# #     density = ax.scatter_density(x, y, cmap=white_viridis)
# #     fig.colorbar(density, label='Number of points per pixel')
# #
# # fig = plt.figure()
# # using_mpl_scatter_density(fig, x, y)
# # plt.show()
#
# plt.rcParams['font.sans-serif'] = ['SimHei']
# mpl.rcParams['axes.unicode_minus'] = False
# import datashader as ds
# from datashader.mpl_ext import dsshow
# import pandas as pd
#
#
# def using_datashader(ax, x, y):
#
#     df = pd.DataFrame(dict(x=x, y=y))
#     dsartist = dsshow(
#         df,
#         ds.Point("x", "y"),
#         ds.count(),
#         vmin=0,
#         vmax=35,
#         norm="linear",
#         aspect="auto",
#         ax=ax,
#     )
#
#     plt.colorbar(dsartist)
#
#
# fig, ax = plt.subplots()
# using_datashader(ax, x, y)
# plt.show()

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import Normalize
from scipy.interpolate import interpn

def density_scatter( x , y, ax = None, sort = True, bins = 20, **kwargs )   :
    """
    Scatter plot colored by 2d histogram
    """
    if ax is None :
        fig , ax = plt.subplots()
    data , x_e, y_e = np.histogram2d( x, y, bins = bins, density = True )
    z = interpn( ( 0.5*(x_e[1:] + x_e[:-1]) , 0.5*(y_e[1:]+y_e[:-1]) ) , data , np.vstack([x,y]).T , method = "splinef2d", bounds_error = False)

    #To be sure to plot all data
    z[np.where(np.isnan(z))] = 0.0

    # Sort the points by density, so that the densest points are plotted last
    if sort :
        idx = z.argsort()
        x, y, z = x[idx], y[idx], z[idx]

    ax.scatter( x, y, c=z, **kwargs )

    norm = Normalize(vmin = np.min(z), vmax = np.max(z))
    cbar = fig.colorbar(cm.ScalarMappable(norm = norm), ax=ax)
    cbar.ax.set_ylabel('Density')

    return ax


if "__main__" == __name__ :

    x = np.random.normal(size=100000)
    y = x * 3 + np.random.normal(size=100000)
    density_scatter( x, y, bins = [30,30] )