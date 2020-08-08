#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug  7 15:40:12 2020

@author: jesica
"""

import xarray as xr
import glob

import regionmask

import numpy as np

import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import matplotlib.ticker as mticker
import matplotlib.pyplot as plt



path='/home/jesica/Documentos/clima-dinamica/Datos/MPI_ESM_LR/'

#creo una lista vacia donde voy a guardar los nombres de los archivos del periodo
#2020-2049
tas_2049_archivos=[]

#archivos del periodo historico
tas_hist_archivos=sorted(glob.glob(path+'tas_Amon_MPI-ESM-LR_historical_*_197601-200512_2.5_anu.nc'))

#archivos del periodo 2020-2049
tas_2049_archivos.append(sorted(glob.glob(path+'tas_Amon_MPI-ESM-LR_rcp26*_202001-204912_2.5_anu.nc')))
tas_2049_archivos.append(sorted(glob.glob(path+'tas_Amon_MPI-ESM-LR_rcp85*_202001-204912_2.5_anu.nc')))

#abro todos los archivos del periodo historico, y los concateno en una nueva dimension llamda 'miembros'
tas_hist=xr.open_mfdataset(tas_hist_archivos, concat_dim='miembros',combine='nested' )


#abro los archivos en un dataset, y los concateno segun el orden en que estan guardados
#en la lista, tengo una lista de 2 elementos (2 escenarios) y cada escenario contiene una
#cierta cantidad de miembros, entonces la segunda dimension sera 'miembros.
#Las dimensiones de la variable entonces termina quedando con 2 dimensiones mas en las que las
#concatene
tas_2049=xr.open_mfdataset(tas_2049_archivos, concat_dim=['escenario','miembros'],combine='nested' )


#cargo los continentes,luego van a ser usados para enmascarar los datos
land = regionmask.defined_regions.natural_earth.land_110

#defino una función donde ingreso un dataset y los continentes. La funcion transforma las longitudes
#de 0-360 a -180-180, aplica la mascara y recorta en sudamerica
def recorte_sudamerica (dataset,land):
    #convierto las longitudes de 0-360 a -180-180
    dataset= dataset.assign_coords(lon=(((dataset.lon + 180) % 360) - 180)).sortby('lon')
    #creo una mascara donde le mando las longitudes y latitudes de mis datos
    mascara=land.mask(dataset.lon,dataset.lat)
    #aplico la mascara a los datos
    enmascarado=dataset.where(mascara==mascara)
    #recorto en sudamerica y solo me quedo con las dimensiones de la region de interes
    recorte= enmascarado.where((dataset.lat<=15) & (dataset.lat>=-60)
                  & (dataset.lon<=-32) & (dataset.lon>=-85),drop=True)
    return recorte


#llamo a la funcion recorte_sudamerica para los datos historicos   
region=recorte_sudamerica(tas_hist,land)


#grafico los datos para un tiempo y miembro en particular
fig = plt.figure(figsize=(12,10))
ax = plt.axes(projection=ccrs.PlateCarree())
p = region.tas.isel(miembros=0,time=5).plot(
        transform=ccrs.PlateCarree(),
        )
gl=ax.gridlines(crs=ccrs.PlateCarree(),draw_labels=True,alpha=0.5,linestyle='--')
gl.top_labels=False
gl.right_labels=False
gl.xlocator=mticker.FixedLocator(np.arange(-180,180+1,60))
gl.ylocator=mticker.FixedLocator(np.arange(-90,90,30))
gl.xformatter=LONGITUDE_FORMATTER
gl.yformatter=LATITUDE_FORMATTER
ax.add_feature(cfeature.COASTLINE,linewidth=0.6)
ax.add_feature(cfeature.BORDERS,linewidth=0.5)

region.tas[0,0,:,:].plot()