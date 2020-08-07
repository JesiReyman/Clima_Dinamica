#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 25 01:24:32 2020

@author: jesica
"""

import glob
import xarray as xr
import numpy as np
from scipy import stats

import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import matplotlib.ticker as mticker

import time

start = time.time()
ruta='/home/jesica/Documentos/clima-dinamica/Datos/MPI_ESM_LR/'

tas=[]
#elijo los archivos con los que voy a trabajar
tas.append(sorted(glob.glob(ruta+'tas_Amon_MPI-ESM-LR_historical_*_197601-200512_2.5_anu.nc')))
tas.append(sorted(glob.glob(ruta+'tas_Amon_MPI-ESM-LR_rcp26_*_202001-204912_2.5_anu.nc')))

evspsbl=[]
evspsbl.append(sorted(glob.glob(ruta+'evspsbl_Amon_MPI-ESM-LR_historical_*_197601-200512_2.5_anu.nc')))
evspsbl.append(sorted(glob.glob(ruta+'evspsbl_Amon_MPI-ESM-LR_rcp26_*_202001-204912_2.5_anu.nc')))


#este prepoceso es util cuando se tienen distintos archivoss con distintos periodos, se guardan los
#datos del tiempo en stime, y se renombra al tiempo como ntime (distintos periodos pero con una longitud fija)
def prepoc(datos):
    datos=datos.assign({'stime':(['time'],datos.time)}).drop('time').rename({'time':'ntime'})
    return datos

tas_set=[]
evspsbl_set=[]
for i in range(len(tas)):
 tas_set.append(xr.open_mfdataset(tas[i], concat_dim='miembros',preprocess=prepoc,combine='nested'))
 evspsbl_set.append(xr.open_mfdataset(evspsbl[i], concat_dim='miembros',preprocess=prepoc,combine='nested'))

tas_set=xr.concat(tas_set,dim='periodos')
evspsbl_set=xr.concat(evspsbl_set,dim='periodos')

tas_valores=tas_set['tas'].values
tiempos=tas_set.stime.dt.year.values
evspsbl_valores=evspsbl_set['evspsbl'].values
lons=tas_set['lon'].values
lats=tas_set['lat'].values

def correlation(x1, x2):

    return stats.pearsonr(x1, x2)[0] # to return a matriz correlation index


#en esta funcion le configuro la coordenada en la cual se va a hacer la correlacion
def wrapped_correlation(da, x, coord='ntime'):
    """Finds the correlation along a given dimension of a dataarray."""

    return xr.apply_ufunc(correlation, 
                          da, 
                          x,
                          input_core_dims=[[coord],[coord]] , 
                          output_core_dims=[[]],
                          vectorize=True,
                          dask='allowed',
                          output_dtypes=[float]
                          )

resultados=wrapped_correlation(tas_set.tas, evspsbl_set.evspsbl)

#calculo la media entre los miembros de cada periodo
media_periodo=resultados.mean('miembros')


end = time.time()
print(end - start)

fig = plt.figure(figsize=(15,13))
cmap='coolwarm'
levels=np.arange(-0.9,1,0.3)
ax = plt.axes(projection=ccrs.PlateCarree())

cf=plt.contourf(lons, lats, media_periodo[1,:,:], 
             transform=ccrs.PlateCarree(),cmap=cmap,levels=levels,extend='both')

#x,y=np.meshgrid(lons,lats)
#points=plt.scatter(x,y,p_significativo,'k',transform=ccrs.PlateCarree())
#gridlines = ax.gridlines()
gl=ax.gridlines(crs=ccrs.PlateCarree(),draw_labels=True,alpha=0.5,linestyle='--')
gl.xlabels_top=False
gl.ylabels_right=False
gl.xlocator=mticker.FixedLocator(np.arange(-180,180+1,60))
gl.ylocator=mticker.FixedLocator(np.arange(-90,90,30))
gl.xformatter=LONGITUDE_FORMATTER
gl.yformatter=LATITUDE_FORMATTER



ax.add_feature(cfeature.COASTLINE,linewidth=0.6)
ax.add_feature(cfeature.BORDERS,linewidth=0.5)
cb = plt.colorbar(cf, shrink=0.3)
cb.ax.tick_params(labelsize=8)
plt.show()
           
           