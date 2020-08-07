#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  2 19:47:10 2020

@author: jesica
"""


from cdo import *
cdo=Cdo()

import glob
import xarray as xr
import numpy as np
import numpy.ma as ma
import matplotlib
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
from cartopy.util import add_cyclic_point

import matplotlib.ticker as mticker
from matplotlib.patches import Rectangle
ruta='/home/jesica/Documentos/clima-dinamica/Datos/MPI_ESM_LR/'

#creo una lista vacia para cada variable donde luego se van a gusrdar las listas de achivos
evspsbl=[]
etp=[]
pr=[]

evspsbl.append(sorted(glob.glob(ruta+'evspsbl_Amon_MPI-ESM-LR_historical_*_197601-200512_2.5_anu.nc')))
evspsbl.append(sorted(glob.glob(ruta+'evspsbl_Amon_MPI-ESM-LR_rcp26_*_202001-204912_2.5_anu.nc')))
evspsbl.append(sorted(glob.glob(ruta+'evspsbl_Amon_MPI-ESM-LR_rcp85_*_202001-204912_2.5_anu.nc')))
evspsbl.append(sorted(glob.glob(ruta+'evspsbl_Amon_MPI-ESM-LR_rcp26_*_207001-209912_2.5_anu.nc')))
evspsbl.append(sorted(glob.glob(ruta+'evspsbl_Amon_MPI-ESM-LR_rcp85_*_207001-209912_2.5_anu.nc')))

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

#uso una mascara para sacar el oceano
mascara='/home/jesica/Documentos/clima-dinamica/Datos/seamask.nc'


#defino  una funcion donde ingreso una lista de archivos y para cada uno de ellos (realizaciones)
#lo recorto a un dominio mas chico,aplico la mascara y luego le calculo el campo medio climatologico del periodo

def recorte(archivo,var):
    
     enmascarado=cdo.mul(input=archivo+' '+mascara)
     recorte=cdo.sellonlatbox('-66,-52.5,-35,-20',input=enmascarado)
     media=cdo.timmean(input=recorte,options='-f nc',returnCdf=True)
     
     #extraigo las variables,y al campo medio lo relleno de nan's
     campo_medio=media.variables[var][:].filled(np.nan)
     
     lons=media.variables['lon'][:]
     lats=media.variables['lat'][:]

     return campo_medio,lons,lats

#creo un arreglo donde voy a guardar cada campo medio (periodo/escenario,miembros,lat,lon)
evspsbl_medio=np.zeros((5,3,7,6),dtype='float32')
etp_medio=np.zeros((5,3,7,6),dtype='float32')
pr_medio=np.zeros((5,3,7,6),dtype='float32')

#llamo a la funcion que recorta y calcula el campo medio de cada uno de los miembros de 
#cada variable y de cada periodo
for i in range(len(etp)):
    for j in range(len(etp[i])):
     etp_medio[i][j],lons,lats=recorte(etp[i][j],'etp')
     evspsbl_medio[i][j]=recorte(evspsbl[i][j],'evspsbl')[0]
     pr_medio[i][j]=recorte(pr[i][j],'pr')[0]


#para cada miembro y horizonte temporal, calculo la aridez y la disponibilidad de agua

aridez=etp_medio/pr_medio
disponibilidad=evspsbl_medio/pr_medio

############################## Budyko ###############################################
#quiero ajustar los datos con la curva de Budyko, con omega=2.6
x_budyko={'x_budyko':np.linspace(0,5,20)}
y_budyko='1+x_budyko-(1+(x_budyko)**2.6)**(1/2.6)'
curva_budyko=eval(y_budyko,x_budyko)

x_budyko_amazona={'x_budyko_amazona':np.linspace(0,5,20)}
y_budyko_amazona='1+x_budyko_amazona-(1+(x_budyko_amazona)**4)**(1/4)'
curva_budyko_ama=eval(y_budyko_amazona,x_budyko_amazona)

#voy a construir las rectas de limite energetico y limite de agua
x_energia={'a':np.linspace(0,1,5)}
recta='a'
energia_limitante=eval(recta,x_energia)
agua_limitante=np.ones(5)

titulos=['Histórico','RCP2.6, 2020-2049','RCP8.5, 2020-2049','RCP2.6, 2070-2099','RCP8.5, 2070-2099']
etiquetas=['r1','r2','r3']
#ploteo el espacio de Budyko para el historico
for j in range(0,5):
 fig = plt.figure(figsize=(10,7))
 
 for i in range(0,3):
  plt.scatter(aridez[j,i,:,:], disponibilidad[j,i,:,:],  alpha=0.5,label=etiquetas[i])
 plt.plot(np.linspace(0,1,5),energia_limitante,'k')
 plt.plot(np.linspace(1,5,5),agua_limitante,'k')
 plt.plot(np.linspace(0,5,20),curva_budyko,'r',label='$\omega=2.6$')
 plt.plot(np.linspace(0,5,20),curva_budyko_ama,'b',label='$\omega=4$')
 plt.ylim((0,1.2))
 plt.xlabel('$E_p/P$')
 plt.ylabel('$E/P$')
 plt.title(titulos[j])
 plt.legend(loc='lower right')

######################## CORRELACION P Y ETP ###############################

#quiero calcular la correlacion para cada realizacion entre p y etp,defino una 
#funcion donde voy a calcular esto

def correlacion(p,etp,var):
    #ingresa un archivo de p y de etp y calculo la correlacion en el tiempo
    tim_cor=cdo.timcor(input=p+' '+etp,returnArray=var)

    return tim_cor

#defino un arreglo donde voy a guardar cada campo de correlacion para cada una 
#de las realizaciones y cada periodo/escenario

correlaciones_PETP=np.zeros((5,3,73,144),dtype='float32')

for i in range(len(etp)):
    for j in range(len(etp[i])):
     correlaciones_PETP[i][j]=correlacion(pr[i][j],etp[i][j],'pr')
    
#calculo el promedio del ensamble
correlaciones_PETP_ensamble=np.mean(correlaciones_PETP,axis=1)

#quiero marcar la correlaciones significativas,es decir la que en valor absoluto
#esten por encima de 0.3610 (test t-student con 95% de confianza)

corr_PETP_significativas=(abs(correlaciones_PETP_ensamble)>0.3610)

############################## clasificacion climatica y proyecciones ###############
#aplico la mascara para todo el globo y luego calculo la media del periodo

def mask(archivo,var):
    
     enmascarado=cdo.div(input=archivo+' '+mascara)
     #recorte=cdo.sellonlatbox('-66,-52.5,-35,-20',input=enmascarado)
     media=cdo.timmean(input=enmascarado,options='-f nc',returnCdf=True)
     
     #extraigo las variables,y al campo medio lo relleno de nan's
     campo_medio=media.variables[var][:].filled(np.nan)
     
     lons=media.variables['lon'][:]
     lats=media.variables['lat'][:]

     return campo_medio,lons,lats
 
#creo un arreglo donde voy a guardar cada campo medio (periodo/escenario,miembros,lat,lon)
evspsbl_medio=np.zeros((5,3,73,144),dtype='float32')
etp_medio=np.zeros((5,3,73,144),dtype='float32')
pr_medio=np.zeros((5,3,73,144),dtype='float32')

#llamo a la funcion que recorta y calcula el campo medio de cada uno de los miembros de 
#cada variable y de cada periodo
for i in range(len(etp)):
    for j in range(len(etp[i])):
     etp_medio[i][j],lons,lats=mask(etp[i][j],'etp')
     evspsbl_medio[i][j]=mask(evspsbl[i][j],'evspsbl')[0]
     pr_medio[i][j]=mask(pr[i][j],'pr')[0]
   
#para cada miembro y horizonte temporal, calculo la aridez y la disponibilidad de agua

aridez_global=etp_medio/pr_medio
disponibilidad_global=evspsbl_medio/pr_medio

#calculo la media de ensamble,es decir, en el eje=1
aridez_ensamble=np.mean(aridez_global,axis=1)
disponibilidad_ensamble=np.mean(disponibilidad_global,axis=1)

#quiero los cambios proyectados de p,e y etp,entonces tomo los arreglos a partir del indice 1
#a 4 del eje 0, y se lo resto al hitorico, es decir indice=0 en el eje 0

delta_E=evspsbl_medio[1:5,:,:,:]-evspsbl_medio[0,:,:,:]
delta_ETP=etp_medio[1:5,:,:,:]-etp_medio[0,:,:,:]
delta_P=pr_medio[1:5,:,:,:]-pr_medio[0,:,:,:]

#calculo la diferencia entre los deltas

dif_EP=delta_E-delta_P
dif_ETPP=delta_ETP-delta_P

#hago el promedio de ensamble (axis=1)
dif_EP=np.mean(dif_EP,axis=1)
dif_ETPP=np.mean(dif_ETPP,axis=1)

#guardo en una nueva matriz los lugare donde se cumplen las distintas condiciones
#y les voy asignando un valor

#defino la matriz donde guardo las condiciones
condiciones=np.zeros((4,73,144),dtype='float32')

condiciones[(dif_EP>0)&(dif_EP<1.5)&(dif_ETPP>=1.5)]=1
condiciones[(dif_EP>=1.5)&(dif_ETPP>=1.5)]=2
condiciones[(dif_EP>=1.5)&(dif_ETPP<1.5)&(dif_ETPP>0)]=3

condiciones[(dif_EP<0)&(dif_EP>-1.5)&(dif_ETPP<=-1.5)]=4
condiciones[(dif_EP<=-1.5)&(dif_ETPP<=-1.5)]=5
condiciones[(dif_EP<=-1.5)&(dif_ETPP<0)&(dif_ETPP>-1.5)]=6


#reemplazo los ceros con nan
condiciones[condiciones==0]=np.nan

#calculo la diferencia de p-e y p-etp

dif_PE=pr_medio-evspsbl_medio
dif_PETP=pr_medio-etp_medio

#ahora calculo el delta de cada una de las diferencias (futuro-presente),es decir
#tomo los indices de 1 a 5 y de lo resto al indice 0 del eje 0 (periodos/escenarios)

delta_dif_PE=dif_PE[1:5,:,:,:]-dif_PE[0,:,:,:]
delta_dif_PETP=dif_PETP[1:5,:,:,:]-dif_PETP[0,:,:,:]

#calculo las medias de ensamble
delta_dif_PE=np.mean(delta_dif_PE,axis=1)
delta_dif_PETP=np.mean(delta_dif_PETP,axis=1)



#hago la clasificacion segun el indice de aridez
clas_climatica=np.zeros((5,73,144),dtype='float32')

clas_climatica[(aridez_ensamble>=0)&(aridez_ensamble<1.5)]=1
clas_climatica[(aridez_ensamble>=1.5)&(aridez_ensamble<2.5)]=2
clas_climatica[aridez_ensamble>=2.5]=3


#reemplazo los ceros con nan
clas_climatica[clas_climatica==0]=np.nan

titulos=['$\Delta$(P-ETP) RCP2.6 (2020/2049-Histórico)','$\Delta$(P-ETP) RCP8.5 (2020/2049-Histórico)','$\Delta$(P-ETP) RCP2.6 (2070/2099-Histórico)','$\Delta$(P-ETP) RCP8.5 (2070/2099-Histórico)']
titulo_cor=['Correlación P-ETP, (1976-2005,CMIP5)','Correlación P-ETP,(2020-2049,rcp26,CMIP5)','Correlación P-ETP, (2020-2049,rcp85,CMIP5)','Correlación P-ETP, (2070-2099,rcp26,CMIP5)','Correlación P-ETP, (2070-2099,rcp85,CMIP5)']
################################# GRAFICO #################################

for i in range(0,4):
 fig = plt.figure(figsize=(15,13))
#cmap = matplotlib.colors.ListedColormap(['b', 'g', 'y'])
#cmap = matplotlib.colors.ListedColormap(['pink', 'red', 'orange','lightblue','blue','cyan'])
#ticks = [1.3,2,2.7]
 cmap='RdYlBu'
 levels=np.arange(-100,125,25)
 ax = plt.axes(projection=ccrs.PlateCarree())
 variable, lons_1 = add_cyclic_point(delta_dif_PETP[i,:,:], coord=lons,axis=1)
#ax.set_extent([-68,-53.5,-34,-21],ccrs.PlateCarree())
 x,y=np.meshgrid(lons_1,lats)

 cf=plt.contourf(x, y, variable, 
             transform=ccrs.PlateCarree(),cmap=cmap,levels=levels,extend='both')

# x,y=np.meshgrid(lons,lats)
 #points=plt.scatter(x,y,p,'k',transform=ccrs.PlateCarree())
 #gridlines = ax.gridlines()
 gl=ax.gridlines(crs=ccrs.PlateCarree(),draw_labels=True,alpha=0.5,linestyle='--')
 gl.xlabels_top=False
 gl.ylabels_right=False
 gl.xlocator=mticker.FixedLocator(np.arange(-180,180+1,30))
 gl.ylocator=mticker.FixedLocator(np.arange(-90,90,30))
 gl.xformatter=LONGITUDE_FORMATTER
 gl.yformatter=LATITUDE_FORMATTER

 ax.add_feature(cfeature.COASTLINE,linewidth=0.6)
 ax.add_feature(cfeature.BORDERS,linewidth=0.5)
#cbar = fig.colorbar(cf, aspect = 20, ticks=ticks,shrink=0.3)
#cbar.set_ticklabels(['Humedo','Transicion','Arido'])
#b_patch = mpatches.Patch(color='b', label='Húmedo')
#g_patch = mpatches.Patch(color='g', label='Transición')
#y_patch = mpatches.Patch(color='y', label='Árido')
#plt.legend(handles=[b_patch,g_patch,y_patch],loc='lower left')
 cb = plt.colorbar(cf, shrink=0.3)
 cb.ax.tick_params(labelsize=8)
 ax.set_title(titulos[i])
 plt.show()