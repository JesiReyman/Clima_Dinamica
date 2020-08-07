# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

from cdo import *
cdo=Cdo()
import glob

import numpy as np
from scipy import stats

import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import matplotlib.ticker as mticker


ruta_cmip5='/home/jesica/Documentos/MPI/MPI_ESM_LR/'

#elijo los archivos con los que voy a trabajar
tas_5_historico='tas_Amon_MPI-ESM-LR_historical_*_197601-200512_2.5_anu.nc'
tas_5_2049_26='tas_Amon_MPI-ESM-LR_rcp26_*_202001-204912_2.5_anu.nc'
tas_5_2049_85='tas_Amon_MPI-ESM-LR_rcp85_*_202001-204912_2.5_anu.nc'
tas_5_7099_26='tas_Amon_MPI-ESM-LR_rcp26_*_207001-209912_2.5_anu.nc'
tas_5_7099_85='tas_Amon_MPI-ESM-LR_rcp85_*_207001-209912_2.5_anu.nc'

evspsbl_5_historico='evspsbl_Amon_MPI-ESM-LR_historical_*_197601-200512_2.5_anu.nc'
evspsbl_5_2049_26='evspsbl_Amon_MPI-ESM-LR_rcp26_*_202001-204912_2.5_anu.nc'
evspsbl_5_2049_85='evspsbl_Amon_MPI-ESM-LR_rcp85_*_202001-204912_2.5_anu.nc'
evspsbl_5_7099_26='evspsbl_Amon_MPI-ESM-LR_rcp26_*_207001-209912_2.5_anu.nc'
evspsbl_5_7099_85='evspsbl_Amon_MPI-ESM-LR_rcp85_*_207001-209912_2.5_anu.nc'


pr_5_historico='pr_Amon_MPI-ESM-LR_historical_*_197601-200512_2.5_anu.nc'
pr_5_2049_26='pr_Amon_MPI-ESM-LR_rcp26_*_202001-204912_2.5_anu.nc'
pr_5_2049_85='pr_Amon_MPI-ESM-LR_rcp85_*_202001-204912_2.5_anu.nc'
pr_5_7099_26='pr_Amon_MPI-ESM-LR_rcp26_*_207001-209912_2.5_anu.nc'
pr_5_7099_85='pr_Amon_MPI-ESM-LR_rcp85_*_207001-209912_2.5_anu.nc'

#etp_5_historico='etp_Amon_MPI-ESM-LR_historical_*_197601-200512_2.5_anu.nc'
#etp_5_2049_26='etp_Amon_MPI-ESM-LR_rcp26_*_202001-204912_2.5_anu.nc'
#etp_5_2049_85='etp_Amon_MPI-ESM-LR_rcp85_*_202001-204912_2.5_anu.nc'
#etp_5_7099_26='etp_Amon_MPI-ESM-LR_rcp26_*_207001-209912_2.5_anu.nc'
#etp_5_7099_85='etp_Amon_MPI-ESM-LR_rcp85_*_207001-209912_2.5_anu.nc'

#junto todas las listas de cada variable en una nueva lista
tas5=[]
tas5.append(glob.glob(ruta_cmip5+tas_5_historico))
tas5.append(glob.glob(ruta_cmip5+tas_5_2049_26))
tas5.append(glob.glob(ruta_cmip5+tas_5_2049_85))
tas5.append(glob.glob(ruta_cmip5+tas_5_7099_26))
tas5.append(glob.glob(ruta_cmip5+tas_5_7099_85))

evspsbl5=[]
evspsbl5.append(glob.glob(ruta_cmip5+evspsbl_5_historico))
evspsbl5.append(glob.glob(ruta_cmip5+evspsbl_5_2049_26))
evspsbl5.append(glob.glob(ruta_cmip5+evspsbl_5_2049_85))
evspsbl5.append(glob.glob(ruta_cmip5+evspsbl_5_7099_26))
evspsbl5.append(glob.glob(ruta_cmip5+evspsbl_5_7099_85))

pr5=[]
pr5.append(glob.glob(ruta_cmip5+pr_5_historico))
pr5.append(glob.glob(ruta_cmip5+pr_5_2049_26))
pr5.append(glob.glob(ruta_cmip5+pr_5_2049_85))
pr5.append(glob.glob(ruta_cmip5+pr_5_7099_26))
pr5.append(glob.glob(ruta_cmip5+pr_5_7099_85))

#etp5=[]
#etp5.append(glob.glob(ruta_cmip5+etp_5_historico))
#etp5.append(glob.glob(ruta_cmip5+etp_5_2049_26))
#etp5.append(glob.glob(ruta_cmip5+etp_5_2049_85))
#etp5.append(glob.glob(ruta_cmip5+etp_5_7099_26))
#etp5.append(glob.glob(ruta_cmip5+etp_5_7099_85))

#ordeno cada una de las listas
def orden(lista):
    lista_ordenada= lista.sort()
    return lista_ordenada


for i in range(0,5):
    orden(pr5[i])
    orden(tas5[i])
    orden(evspsbl5[i])
    
    
#calculo la correlacion entre cada miembro de T y E, P y E, y hago el promedio
#de ensamble

def cor_ensamble(lista1,lista2,var):
    
   #inicio lista donde se guardaran las correlaciones entre los miembros
    correlacion_miembros=[]
    #para cada miembro,calculo la correlacion,y lo guardo en la lista
    for i in range(len(lista1)):     
     temporal=cdo.timcor(input=lista1[i]+' '+lista2[i])
     correlacion_miembros.append(temporal)
     
    #hago un promedio de ensamble
    
    media_ensamble=cdo.ensmean(input=correlacion_miembros)
    #defino el dominio
    media_ensamble=cdo.sellonlatbox('-180,180,-90,90',input=media_ensamble,options='-f nc',returnCdf=True)
   
    ensamble=media_ensamble.variables[var][:]
    latitudes=media_ensamble.variables['lat'][:]
    longitudes=media_ensamble.variables['lon'][:]
   
    return ensamble,latitudes,longitudes

correlacion_T_E=np.empty((5,73,144),dtype='float32')

for i in range(len(evspsbl5)):
 correlacion_T_E[i],lats,lons=cor_ensamble(evspsbl5[i], tas5[i], 'evspsbl')


  

cor_sig=(abs(correlacion_T_E)>0.3118206902)

 
fig = plt.figure(figsize=(15,13))
cmap='coolwarm'
levels=np.arange(-0.9,1,0.3)
ax = plt.axes(projection=ccrs.PlateCarree())

cf=plt.contourf(lons, lats, correlacion_T_E[0], 
             transform=ccrs.PlateCarree(),cmap=cmap,levels=levels,extend='both')

x,y=np.meshgrid(lons,lats)
points=plt.scatter(x,y,cor_sig[0],'k',transform=ccrs.PlateCarree())
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