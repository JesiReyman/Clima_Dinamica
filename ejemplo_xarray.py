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
from scipy import stats

import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import matplotlib.ticker as mticker
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()


path='/home/jesica/Documentos/clima-dinamica/Datos/MPI-ESM1-2-LR/'

#creo una lista vacia donde voy a guardar los nombres de los archivos, se va a crear una lista de listas

pr_archivos=[]
#tas_archivos=[]
et_archivos=[]

pr_archivos.append(sorted(glob.glob(path+'pr_Amon_MPI-ESM1-2-LR_historical_*_2.5_mes.nc')))
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
   datos=xr.open_mfdataset(lista[variable], concat_dim='miembros',combine='nested')
   #agrego una dimension, el historico no tiene escenarios pero para poder hacer cuentas 
   #con las proyecciones tiene que tener la misma dimension
   datos=datos.expand_dims(dim='rcp',axis=1)
   #guardo en la lista dataset
   dataset.append(datos)
   
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


#llamo a la funcion recorte_sudamerica para cada dataset 
historico_region=recorte_sudamerica(historico,land)
p2049_region=recorte_sudamerica(periodo_2049,land)


#a cada uno de los sets de datos, calculo la media estacional,pero esto cambia
#segun la variable,por ejemplo si es temperatura tiene que calcular la media,si es pp o et
#tiene que ser la suma acumulada estacional.
#Ademas selecciona el periodo segun indique
def media_estacional(dataset,variable):
    if (variable=='pr')|(variable=='evspsbl')|(variable=='etp'):
        
    #hay que tener cuidado en estas cuentas, si tengo los datos enmascarados entonces
    #los arreglos van a tener nan's, estos al aplicarle la suma van a reemplazarse por ceros
    #que luego sirven si por ejemplo se hace un calculo estadistico (como una correlacion).
    #Sin embargo, al graficar van a aparecer ceros sobre oceano. 
    # se debe poner skipna=False, para mantener los nan's
     
     data_estacional=dataset.resample(time='QS-DEC').sum('time',skipna=False)

    else:
        data_estacional=dataset.resample(time='QS-DEC').mean('time')
    
    #me quedo con los datos que estan entre diciembre del primer año (indice=4) y primavera del
    #ultimo año,si entonces hay 30 años de datos, quiero que se quede hasta el indice 119 
    data_estacional=data_estacional.isel(time=slice(4,120))
    
    return data_estacional

#llamo a la funcion media_estacional para cada dataset
hist_estacional=media_estacional(historico_region,'pr')
p2049_estacional=media_estacional(p2049_region,'pr')

#quiero calcular la serie temporal, es decir calcular los promedios areales.
#Para esto primero tengo que definir los pesos por latitud
pesos = np.cos(np.deg2rad(hist_estacional.lat))

#aplico los pesos a los datos
hist_estacional_weighted = hist_estacional.weighted(pesos)

#calculo el promedio areal de los datos ya pesados
hist_estacional_series=hist_estacional_weighted.mean(('lon','lat'))

hist_estacional_serie_sinpeso=hist_estacional.mean(('lon','lat'))

hist_estacional_series.pr.isel(miembros=0).plot(label='pesado')
hist_estacional_serie_sinpeso.pr.isel(miembros=0).plot(label='no pesado')
plt.legend()

#calculo la diferencia de p-e para cada estacion de cada año y luego calculo el promedio 
#de esa diferencia en el tiempo
 
dif_pe_hist=(hist_estacional.pr-hist_estacional.evspsbl).groupby('time.season').mean('time')
dif_pe_2049=(p2049_estacional.pr-p2049_estacional.evspsbl).groupby('time.season').mean('time')

#calculo el delta de p-e para cada estacion del año y luego hago el promedio entre los miembros
delta_pe=(dif_pe_2049.isel(rcp=[0,1])-dif_pe_hist.isel(rcp=0)).mean('miembros')


#correlacion entre p y e por estacion

def correlation(x1, x2):

    return stats.pearsonr(x1, x2)[0] # to return a matriz correlation index


#en esta funcion le configuro la coordenada en la cual se va a hacer la correlacion
def wrapped_correlation(da, x, coord='time'):
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

#para calcular las correlaciones primero reemplazo los nan's con ceros, luego los agrupo por estaciones para realizar la correlacion estacion por estacion entre
#ambas variables
correlaciones_estacionales=(wrapped_correlation(hist_estacional.pr.fillna(0).groupby('time.season'), hist_estacional.evspsbl.fillna(0).groupby('time.season'))).mean('miembros')

#xarray ya tiene una funcion de correlacion,que acepta 2 datarrays y la dimension en la que se realiza 
#(todavia falta para calcular por estacion la correlacion temporal)
correlaciones_xr=xr.corr(hist_estacional.pr.fillna(0), hist_estacional.evspsbl.fillna(0),dim='time')


#grafico los datos para un tiempo y miembro en particular
fig= plt.figure(figsize=(12,10))
ax = plt.axes(projection=ccrs.PlateCarree())
cmap='coolwarm_r'
levels=np.arange(-120,160,40)
p = delta_pe.isel(rcp=0,season=2).plot.contourf(
        transform=ccrs.PlateCarree(),cmap=cmap,levels=levels,extend='both',
        add_colorbar=False)
        
gl=ax.gridlines(crs=ccrs.PlateCarree(),draw_labels=True,alpha=0.5,linestyle='--')
gl.top_labels=False
gl.right_labels=False
gl.xlocator=mticker.FixedLocator(np.arange(-180,180+1,15))
gl.ylocator=mticker.FixedLocator(np.arange(-90,90,30))
gl.xformatter=LONGITUDE_FORMATTER
gl.yformatter=LATITUDE_FORMATTER
ax.add_feature(cfeature.COASTLINE,linewidth=0.6)
ax.add_feature(cfeature.BORDERS,linewidth=0.5)
cb = plt.colorbar(p,shrink=0.3)


