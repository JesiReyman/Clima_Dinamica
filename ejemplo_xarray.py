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



path='/home/jesica/Documentos/MPI/MPI-ESM1-2-LR/'

#creo una lista vacia donde voy a guardar los nombres de los archivos, se va a crear una lista de listas

pr_archivos=[]
#tas_archivos=[]
et_archivos=[]
#archivos del periodo historico
pr_archivos.append(sorted(glob.glob(path+'pr_Amon_MPI-ESM1-2-LR_historical_*_2.5_mes.nc')))

#archivos del periodo 2020-2049
pr_archivos.append(sorted(glob.glob(path+'pr_Amon_MPI-ESM1-2-LR_ssp126_*_2020-2049_2.5_mes.nc')))
pr_archivos.append(sorted(glob.glob(path+'pr_Amon_MPI-ESM1-2-LR_ssp585_*_2020-2049_2.5_mes.nc')))

#tas_archivos.append(sorted(glob.glob(path+'tas_Amon_MPI-ESM1-2-LR_historical_*_2.5.nc')))
#tas_archivos.append(sorted(glob.glob(path+'tas_Amon_MPI-ESM1-2-LR_ssp126_*_2020-2049_2.5.nc')))
#tas_archivos.append(sorted(glob.glob(path+'tas_Amon_MPI-ESM1-2-LR_ssp585_*_2020-2049_2.5.nc')))

et_archivos.append(sorted(glob.glob(path+'evspsbl_Amon_MPI-ESM1-2-LR_historical_*_2.5_mes.nc')))
et_archivos.append(sorted(glob.glob(path+'evspsbl_Amon_MPI-ESM1-2-LR_ssp126_*_2020-2049_2.5_mes.nc')))
et_archivos.append(sorted(glob.glob(path+'evspsbl_Amon_MPI-ESM1-2-LR_ssp585_*_2020-2049_2.5_mes.nc')))


#este proceso sirve si no interesa preservar las fechas, un ejemplo puede ser que se trabajen con datos anuales
#def preproceso(dataset):
 #   dataset=dataset.reset_index('time')
  #  return dataset

#defino una funcion en donde le mando una lista de una o varias variables, y para cada variable
#armo un datatset, que tendra agregado la dimension 'miembros' (si por ejemplo es un solo periodo con un solo escenario,historico)
#y si ademas tiene mas de un escenario para un mismo periodo, se agrega la dimension 'rcp'. 
#Luego de tener los datasets de cada variable, le aplico un merge para unificar los datos, segun el periodo

def set_datos(lista):
    
 variables=len(lista)  
 tipo=type(lista[0][0])

 #defino una lista vacia donde voy a guardar los datasets de cada variable
 dataset=[] 
 
 #para cada variable que esta en la lista, armo un dataset
 for variable in range(0,variables):
  #si dentro de la primer lista se encuentran solo strings, quiere decir que ya son los 
  #los nombres de los archivos, que serian los miembros,entonces los concateno
  #en esa nueva dimension   
  if (tipo==str):   
   dataset.append(xr.open_mfdataset(lista[variable], concat_dim='miembros',combine='nested'))
  #si en cambio dentro de la lista a su vez se encuentra otra lista, cada una de esa listas internas representan
  #distintos escenarios para un periodo, por lo tanto concanteno ademas de los miembros, los escenarios
  #en la dimension 'rcp'
  elif (tipo==list): 
   dataset.append(xr.open_mfdataset(lista[variable], concat_dim=['rcp','miembros'],combine='nested'))
 
 #unifico los datasets de todas las variables haciendo un merge   
 dataset=xr.merge(dataset)   
 return dataset


#llamo a la funcion set_datos para empezar a crear los datasets segun el periodo
historico=set_datos([pr_archivos[0],et_archivos[0]]) #las primeras listas son del historico
periodo_2049=set_datos([pr_archivos[1:3],et_archivos[1:3]]) #las listas 1 y 2 pertenecen al futuro cercano


pr_2049=xr.open_mfdataset(pr_archivos[1], concat_dim='miembros',combine='nested' )

pr=xr.combine_nested([pr_hist,pr_2049], concat_dim='Periodo')

#tas=xr.open_mfdataset(tas_archivos, concat_dim=['periodo','miembros'],combine='nested' )

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
pr_region=recorte_sudamerica(hist,land)
et_region=recorte_sudamerica(et,land)
pr_2049_region=recorte_sudamerica(pr_2049,land)

#a cada uno de los sets de datos, calculo la suma estacional (porque son pp y et, acumulables)

pr_estacional=pr_region.pr.resample(time='QS-DEC').sum('time')
pr_estacional=pr_2049_region.resample(time='QS-DEC').sum('time')
et_estacional=et_region.evspsbl.resample(time='QS-DEC').sum('time')

 
 suma_hist_verano=pr_estacional.sel(time=slice('1976-12-01','2005-09-01'),periodo=0)+tas_estacional.sel(time=slice('1976-12-01','2005-09-01'),periodo=0)
 verano_medio=suma_hist_verano.groupby('time.season').mean('time')
 return dataset_estacional

pr_2049_estacional=pr_2049.resample(time='QS-DEC').sum('time').sel(time=slice('2020-12-01','2049-09-01'))
tas_hist=pr_hist.resample(time='QS-DEC').sum('time').sel(time=slice('1976-12-01','2005-09-01'))
delta_verano=[]

for i in range(0,2):

 delta_verano=(pr_2049_estacional.pr.sel(time=pr_2049_estacional['time.season']=='DJF',rcp=[0])+pr_hist_estacional.pr.sel(time=pr_hist_estacional['time.season']=='DJF'))


pr_verano_hist=pr_hist_estacional.pr.sel(time=pr_estacional_hist['time.season']=='DJF')
#abro los archivos en un dataset, y los concateno segun el orden en que estan guardados
#en la lista, tengo una lista de 2 elementos (2 escenarios) y cada escenario contiene una
#cierta cantidad de miembros, entonces la segunda dimension sera 'miembros.
#Las dimensiones de la variable entonces termina quedando con 2 dimensiones mas en las que las
#concatene
tas_2049=xr.open_mfdataset(tas_2049_archivos, concat_dim=['escenario','miembros'],combine='nested' )






#grafico los datos para un tiempo y miembro en particular
fig = plt.figure(figsize=(12,10))
ax = plt.axes(projection=ccrs.PlateCarree())
p = periodo_2049.pr.isel(miembros=0,time=5,rcp=1).plot.pcolormesh(
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