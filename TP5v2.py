#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 26 00:47:04 2020

@author: jesica
"""

import xarray as xr
import glob
import numpy as np
import numpy.ma as ma
from scipy import stats

import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
from cartopy.util import add_cyclic_point

import matplotlib.ticker as mticker
import time


start = time.time()
ruta='/home/jesica/Documentos/clima-dinamica/Datos/MPI_ESM_LR/'

#defino listas de cada variable donde voy a guardar las listas segun periodo y escenario
tas=[]
evspsbl=[]
pr=[]
etp=[]

etp.append(sorted(glob.glob(ruta+'etp_Amon_MPI-ESM-LR_historical_*_197601-200512_2.5_anu.nc')))
etp.append(sorted(glob.glob(ruta+'etp_Amon_MPI-ESM-LR_rcp26_*_202001-204912_2.5_anu.nc')))
etp.append(sorted(glob.glob(ruta+'etp_Amon_MPI-ESM-LR_rcp85_*_202001-204912_2.5_anu.nc')))
etp.append(sorted(glob.glob(ruta+'etp_Amon_MPI-ESM-LR_rcp26_*_207001-209912_2.5_anu.nc')))
etp.append(sorted(glob.glob(ruta+'etp_Amon_MPI-ESM-LR_rcp85_*_207001-209912_2.5_anu.nc')))

pr.append(sorted(glob.glob(ruta+'pr_Amon_MPI-ESM-LR_historical_*_197601-200512_2.5_anu.nc')))
pr.append(sorted(glob.glob(ruta+'pr_Amon_MPI-ESM-LR_rcp26_*_202001-204912_2.5_anu.nc')))
pr.append(sorted(glob.glob(ruta+'pr_Amon_MPI-ESM-LR_rcp85_*_202001-204912_2.5_anu.nc')))
pr.append(sorted(glob.glob(ruta+'pr_Amon_MPI-ESM-LR_rcp26_*_207001-209912_2.5_anu.nc')))
pr.append(sorted(glob.glob(ruta+'pr_Amon_MPI-ESM-LR_rcp85_*_207001-209912_2.5_anu.nc')))


##HISTORICO
evspsbl.append(sorted(glob.glob(ruta+'evspsbl_Amon_MPI-ESM-LR_historical_*_197601-200512_2.5_anu.nc')))
tas.append(sorted(glob.glob(ruta+'tas_Amon_MPI-ESM-LR_historical_*_197601-200512_2.5_anu.nc')))
pr.append(sorted(glob.glob(ruta+'pr_Amon_MPI-ESM-LR_historical_*_197601-200512_2.5_anu.nc')))

#2020-2049 2.6
evspsbl.append(sorted(glob.glob(ruta+'evspsbl_Amon_MPI-ESM-LR_rcp26_*_202001-204912_2.5_anu.nc')))
tas.append(sorted(glob.glob(ruta+'tas_Amon_MPI-ESM-LR_rcp26_*_202001-204912_2.5_anu.nc')))
pr.append(sorted(glob.glob(ruta+'pr_Amon_MPI-ESM-LR_rcp26_*_202001-204912_2.5_anu.nc')))

#2020-2049 8.5
evspsbl.append(sorted(glob.glob(ruta+'evspsbl_Amon_MPI-ESM-LR_rcp85_*_202001-204912_2.5_anu.nc')))
tas.append(sorted(glob.glob(ruta+'tas_Amon_MPI-ESM-LR_rcp85_*_202001-204912_2.5_anu.nc')))
pr.append(sorted(glob.glob(ruta+'pr_Amon_MPI-ESM-LR_rcp85_*_202001-204912_2.5_anu.nc')))

#2070-2099 2.6
evspsbl.append(sorted(glob.glob(ruta+'evspsbl_Amon_MPI-ESM-LR_rcp26_*_207001-209912_2.5_anu.nc')))
tas.append(sorted(glob.glob(ruta+'tas_Amon_MPI-ESM-LR_rcp26_*_207001-209912_2.5_anu.nc')))
pr.append(sorted(glob.glob(ruta+'pr_Amon_MPI-ESM-LR_rcp26_*_207001-209912_2.5_anu.nc')))

#2070-2099 8.5
evspsbl.append(sorted(glob.glob(ruta+'evspsbl_Amon_MPI-ESM-LR_rcp85_*_207001-209912_2.5_anu.nc')))
tas.append(sorted(glob.glob(ruta+'tas_Amon_MPI-ESM-LR_rcp85_*_207001-209912_2.5_anu.nc')))
pr.append(sorted(glob.glob(ruta+'pr_Amon_MPI-ESM-LR_rcp85_*_207001-209912_2.5_anu.nc')))

#defino una funcion que abra una lista de archivos en una sola,y extraiga 
#los datos

def extraccion_datos(lista,var):
 #uso xarray para abrir en un archivo una lista de miembros,los concatena y los guarda
 #en una nueva dimension llamada 'miembros',que pasa a ser la primer dimension
 datos=xr.open_mfdataset(lista, concat_dim='miembros',combine='nested')

 #extraigo datos segun de que variable se trate
 if var=='tas':
     variable=np.array(datos.tas-273.15)
 elif var=='evspsbl':
     variable=np.array(datos.evspsbl)
 elif var=='pr':
     variable=np.array(datos.pr)
     
 lons=np.array(datos.lon)
 lats=np.array(datos.lat)
 años=np.array(datos.time.dt.year)
 miembros=np.array(datos.miembros)
 
 return variable,años,lats,lons,miembros

#defino las matrices donde voy a guardar todos los datos de cada variable,las dimensiones son:
#(5,3,30,73,144)=(periodo-horizonte,miembros,tiempos,latitudes,longitudes)    
tas_matriz=np.zeros((5,3,30,73,144),dtype='float32')
evspsbl_matriz=np.zeros((5,3,30,73,144),dtype='float32')
pr_matriz=np.zeros((5,3,30,73,144),dtype='float32')
tiempo=np.zeros((5,30),dtype='float32')

#llamo a la funcion extraccion_datos para cada lista de cada variable
for i in range(len(tas)):
    tas_matriz[i],tiempo[i],lats,lons,miembros=extraccion_datos(tas[i],'tas')
    #para evspsbl y pr me quedo solo con la primer salida de la funcion (variable)
    #ya extraje las latitudes,longitudes y el tiempo que son las mismas que tas
    evspsbl_matriz[i]=extraccion_datos(evspsbl[i],'evspsbl')[0]
    pr_matriz[i]=extraccion_datos(pr[i],'pr')[0]
 
################################ CALCULOS ######################################    
#defino una funcion que calcula la correlacion entre cada uno de los miembros
#de las variables ingresadas y luego hace el promedio entre esos campos para obtener 
#la correlacion (y el p-value) del ensamble   
    
def correlacion(var1,var2,miembros): 
 #defino arreglos donde guardo las correlaciones y p-values de los miembros
 correlacion_miembros=np.zeros((len(miembros),73,144),dtype='float32')
 p_miembros=np.zeros((len(miembros),73,144),dtype='float32')

 #hago la correlacion para cada miembro del ensamble
 for i in range(len(miembros)):
    for j in range(0,73):
        for k in range(0,144):
            correlacion_miembros[i][j][k],p_miembros[i][j][k]=stats.pearsonr(var1[i,:,j,k],var2[i,:,j,k])

 #hago el promedio entre los campos de correlacion (ensamble)   
 correlacion_ensamble=np.mean(correlacion_miembros,axis=0)      
 p_ensamble=np.mean(p_miembros,axis=0)   
 p_ensamble=(p_ensamble<0.05)
 
 return correlacion_ensamble,p_ensamble

#defino las matrices donde se guardaran los campos de correlaciones y p-values del ensamble
correlaciones_T_E_ensamble=np.zeros((5,73,144),dtype='float32')
p_T_E_ensamble=np.zeros((5,73,144),dtype='bool')
correlaciones_P_E_ensamble=np.zeros((5,73,144),dtype='float32')
p_P_E_ensamble=np.zeros((5,73,144),dtype='bool')
    
#llamo a la funcion correlacion para evspsbl vs. tas y evspsbl vs. pr
for i in range(0,5):
     correlaciones_T_E_ensamble[i],p_T_E_ensamble[i]=correlacion(tas_matriz[i],evspsbl_matriz[i],miembros)
     correlaciones_P_E_ensamble[i],p_P_E_ensamble[i]=correlacion(pr_matriz[i],evspsbl_matriz[i],miembros)

#defino una funcion que calcula el desvio estandar para cada uno de los miembros y luego los promedia
def desvio(var,miembros): 
 #defino arreglos donde guardo los desvios de los miembros
 desvio_miembros=np.zeros((len(miembros),73,144),dtype='float32')

 #hago el desvio para cada miembro del ensamble
 for i in range(len(miembros)):
    for j in range(0,73):
        for k in range(0,144):
            desvio_miembros[i][j][k]=np.std(var[i,:,j,k])

 #hago el promedio entre los campos de correlacion (ensamble)   
 desvio_ensamble=np.mean(desvio_miembros,axis=0)   
  
 return desvio_ensamble

#defino las matrices donde se guardaran los campos de desvios del ensamble
desvio_E_ensamble=np.zeros((5,73,144),dtype='float32')
desvio_P_ensamble=np.zeros((5,73,144),dtype='float32')

#Llamo a la funcion desvio para cada periodo
for i in range(0,5):
     desvio_E_ensamble[i]=desvio(evspsbl_matriz[i],miembros)
     desvio_P_ensamble[i]=desvio(pr_matriz[i],miembros)

#me quiero quedar con los valores de desvios de P mayores de 73 mm
desvio_P_filtro=ma.masked_array(desvio_P_ensamble,(desvio_P_ensamble<73))

#calculo gamma para cada uno de los periodos
gamma=correlaciones_P_E_ensamble*desvio_E_ensamble/desvio_P_filtro

end = time.time()
print(end - start)

########################### GRAFICOS #############################################

def mapas(var,latitudes,longitudes,titulo):
 fig = plt.figure(figsize=(15,13))
 cmap='rainbow'
 levels=np.arange(-0.9,1,0.3)
 ax = plt.axes(projection=ccrs.PlateCarree())
 variable, lons_1 = add_cyclic_point(var, coord=longitudes,axis=1)

 cf=plt.contourf(lons_1, latitudes, variable, 
             transform=ccrs.PlateCarree(),cmap=cmap,levels=levels,extend='both')

 #x,y=np.meshgrid(longitudes,latitudes)
# points=plt.scatter(x,y,p,'k',transform=ccrs.PlateCarree())
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
 ax.set_title(titulo)
 plt.show()
 
 
######################################################
#Llamo a la funcion mapas para graficar el campo de correlaciones y lo puntos significativos
titulo=['Correlación P-E, (1976-2005,CMIP5)','Correlación P-E,(2020-2049,rcp26,CMIP5)','Correlación P-E, (2020-2049,rcp85,CMIP5)','Correlación P-E, (2070-2099,rcp26,CMIP5)','Correlación P-E, (2070-2099,rcp85,CMIP5)']
titulo1=['$\Gamma$ (1976-2005,CMIP5)','$\Gamma$ (2020-2049,rcp26,CMIP5)','$\Gamma$ (2020-2049,rcp85,CMIP5)','$\Gamma$ (2070-2099,rcp26,CMIP5)','$\Gamma$ (2070-2099,rcp85,CMIP5)']

for i in range(0,5):
    mapas(correlaciones_P_E_ensamble[i],p_P_E_ensamble[i],lats,lons,titulo1[i])

for i in range(0,5):
    mapas(gamma[i],lats,lons,titulo1[i])
