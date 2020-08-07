#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 18 19:43:15 2020

@author: jesica
"""

from cdo import *
cdo=Cdo()

import glob

import numpy as np
import numpy.ma as ma

#import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
#import seaborn as sns
#sns.set()
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER

import matplotlib.ticker as mticker

ruta='/home/jesica/Documentos/MPI/MPI_ESM_LR/'

#creo una lista vacia para cada variable donde luego se van a gusrdar las listas de achivos
evspsbl=[]
pr=[]

tas=[]
hus=[]

evspsbl.append(sorted(glob.glob(ruta+'evspsbl_Amon_MPI-ESM1-2-LR_historical_*_2.5_mes.nc')))
evspsbl.append(sorted(glob.glob(ruta+'evspsbl_Amon_MPI-ESM1-2-LR_ssp126_*_2020-2049_2.5_mes.nc')))
evspsbl.append(sorted(glob.glob(ruta+'evspsbl_Amon_MPI-ESM1-2-LR_ssp585_*_2020-2049_2.5_mes.nc')))
evspsbl.append(sorted(glob.glob(ruta+'evspsbl_Amon_MPI-ESM1-2-LR_ssp126_*_2070-2099_2.5_mes.nc')))
evspsbl.append(sorted(glob.glob(ruta+'evspsbl_Amon_MPI-ESM1-2-LR_ssp585_*_2070-2099_2.5_mes.nc')))

pr.append(sorted(glob.glob(ruta+'pr_Amon_MPI-ESM1-2-LR_historical_*_2.5_mes.nc')))
pr.append(sorted(glob.glob(ruta+'pr_Amon_MPI-ESM1-2-LR_ssp126_*_2020-2049_2.5_mes.nc')))
pr.append(sorted(glob.glob(ruta+'pr_Amon_MPI-ESM1-2-LR_ssp585_*_2020-2049_2.5_mes.nc')))
pr.append(sorted(glob.glob(ruta+'pr_Amon_MPI-ESM1-2-LR_ssp126_*_2070-2099_2.5_mes.nc')))
pr.append(sorted(glob.glob(ruta+'pr_Amon_MPI-ESM1-2-LR_ssp585_*_2070-2099_2.5_mes.nc')))

tas.append(sorted(glob.glob(ruta+'tas_Amon_MPI-ESM-LR_historical_r1_2.5.nc')))
tas.append(sorted(glob.glob(ruta+'tas_Amon_MPI-ESM-LR_rcp26_*_2020-2049_2.5.nc')))
tas.append(sorted(glob.glob(ruta+'tas_Amon_MPI-ESM-LR_rcp85_*_2020-2049_2.5.nc')))
tas.append(sorted(glob.glob(ruta+'tas_Amon_MPI-ESM-LR_rcp26_*_2070-2099_2.5.nc')))
tas.append(sorted(glob.glob(ruta+'tas_Amon_MPI-ESM-LR_rcp85_*_2070-2099_2.5.nc')))

hus.append(sorted(glob.glob(ruta+'hus_Amon_MPI-ESM-LR_historical_r1_197601-200512_1000-700hPa_2.5.nc')))
hus.append(sorted(glob.glob(ruta+'huss_Amon_MPI-ESM-LR_rcp26_r1_2020-2049_2.5.nc')))
hus.append(sorted(glob.glob(ruta+'huss_Amon_MPI-ESM-LR_rcp85_r1_2020-2049_2.5.nc')))
hus.append(sorted(glob.glob(ruta+'huss_Amon_MPI-ESM-LR_rcp26_r1_2070-2099_2.5.nc')))
hus.append(sorted(glob.glob(ruta+'huss_Amon_MPI-ESM-LR_rcp85_r1_2070-2099_2.5.nc')))



#defino una funcion en donde voy a ingresar una lista de archivos y el nombre de la variable,y calcula para
#esa variable las medias y las varianzas a lo largo de cada uno de los periodos

def estaciones(variable,var):
 periodo=['1977/2005','2021/2049','2021/2049','2071/2099','2071/2099']
 #defino arreglos donde se van a guardar cada una de las medias y varainzas para cada estacion
 media_DEF=np.zeros((5,10,31,22),dtype='float32')
 media_MAM=np.zeros((5,10,31,22),dtype='float32')
 media_JJA=np.zeros((5,10,31,22),dtype='float32')
 media_SON=np.zeros((5,10,31,22),dtype='float32')
 
 var_DEF=np.zeros((5,10,31,22),dtype='float32')
 var_MAM=np.zeros((5,10,31,22),dtype='float32')
 var_JJA=np.zeros((5,10,31,22),dtype='float32')
 var_SON=np.zeros((5,10,31,22),dtype='float32')
 #para cada archivo que ingrese, recorto en la region de interes, extraigo las latitudes y longitudes
 #calculo las sumatorias estacionales y luego separo por estaciones y calculo las medias y varianzas
 for i in range(len(variable)):
     for j in range(len(variable[i])):
      #recorto a cada uno de los archivos en Sudamerica
      sudamerica=cdo.sellonlatbox('-85,-32,-60,15',input=variable[i][j])
      lats=cdo.sellonlatbox('-85,-32,-60,15',input=variable[i][j],returnArray='lat')
      lons=cdo.sellonlatbox('-85,-32,-60,15',input=variable[i][j],returnArray='lon')
      #calculo la sumatoria por estaciones
      variable_estacional=cdo.seassum(input=sudamerica)
      #para cada estacion calculo las medias
      media_DEF[i,j]=cdo.timmean(input='-selyear,'+periodo[i]+' -select,month=1 '+variable_estacional,returnArray=var)
      media_MAM[i,j]=cdo.timmean(input='-select,month=4 '+variable_estacional,returnArray=var)
      media_JJA[i,j]=cdo.timmean(input='-select,month=7 '+variable_estacional,returnArray=var)
      media_SON[i,j]=cdo.timmean(input='-select,month=10 '+variable_estacional,returnArray=var)
 
      #ademas calculo las varianzas
      var_DEF[i,j]=cdo.timvar(input='-selyear,'+periodo[i]+' -select,month=1 '+variable_estacional,returnArray=var)
      var_MAM[i,j]=cdo.timvar(input='-select,month=4 '+variable_estacional,returnArray=var)
      var_JJA[i,j]=cdo.timvar(input='-select,month=7 '+variable_estacional,returnArray=var)
      var_SON[i,j]=cdo.timvar(input='-select,month=10 '+variable_estacional,returnArray=var)
 
#guardo los arreglos de las medias y las varianzas en un arreglo mas grande en donde en la 
#dimension 0 van a representar las 4 estaciones. Tamaño esperado: (4,5,31,22)
 medias=np.stack((media_DEF,media_MAM,media_JJA,media_SON),axis=0)
 varianzas=np.stack((var_DEF,var_MAM,var_JJA,var_SON),axis=0)

 return medias,varianzas,lats,lons


#para cada lista de variables llamo a la funcion estaciones, que me devuelve un arreglo de las 
#medias y varianzas para cada una de las estaciones y cada periodo. Ademas me devuelve las latitudes y longitudes
pr_medias,pr_varianzas,lats,lons=estaciones(pr,'pr')
evspsbl_medias,evspsbl_varianzas=estaciones(evspsbl,'evspsbl')[0:2]

#calculo la media de ensamble
pr_medias_ens=np.mean(pr_medias,axis=2)
evspsbl_medias_ens=np.mean(evspsbl_medias,axis=2)


#defino ahora una funcion test, que me devolvera los arreglos de los puntos significativos.
#testea si las diferencias entre los campos medios futuros e historico son significativas,devuelve matrices logicos (true o false)
def test(media_historico,media_futuro,var_historico,var_futuro,n):
   #primero calculo la varianza pesada
  sp=((n-1)*var_futuro+(n-1)*var_historico)/(n+n-2)
  #calculo el estadistico t con n+n-2 grados de libertad
  test=(media_futuro-media_historico)/np.sqrt(sp*(1/n+1/n))
  #defino los umbrales donde el estadistico t sera con un alpha=0.025 (a dos colas,5% de significancia)
  #para 29+29-2 (para def queda una estacion menos porque falta el diciembre del año anterior, y enero y febrero del ultimo año)
  #grados de libertad para def y para 58 grados de libertad para el resto de las estaciones
  #(son 30 años,entonces n=30)
 # t_def=2.0032
  #t_estaciones=2.0017
  
  #if n==29:
   #significativo=(abs(test)>t_def)
  #elif n==30:
   #   significativo=(abs(test)>t_estaciones)
      
  return test


#para cada una de las estaciones, voy a hacer el test, que consiste en ingresar los arreglos
#de las medias y varianzas del historico y del futuro, y ademas de la cantidad de años de cada estacion
n=[29,30,30,30]
#defino un arreglo vacio donde voy a guardar donde las diferencias entre el fututo y el historico es significativo
pr_significativo=np.zeros((4,4,10,31,22),dtype='float32')
evspsbl_significativo=np.zeros((4,4,10,31,22),dtype='float32')
for i in range(0,4):
 pr_significativo[i]=test(pr_medias[i,0,:,:,:], pr_medias[i,1:5,:,:,:], pr_varianzas[i,0,:,:,:], pr_varianzas[i,1:5,:,:,:], n[i])
 evspsbl_significativo[i]=test(evspsbl_medias[i,0,:,:,:], evspsbl_medias[i,1:5,:,:,:], evspsbl_varianzas[i,0,:,:,:], evspsbl_varianzas[i,1:5,:,:,:], n[i])

#calculo el promedio de ensamble, es decir, calculo la media en el eje=2 de los resultados del test
pr_significativo=np.mean(pr_significativo,axis=2)
evspsbl_significativo=np.mean(evspsbl_significativo,axis=2)

#le pongo el umbral a partir de donde las diferencias son significativas con 30+30-2 grados de libertad
#y 95% de nivel de significancia

pr_significativo=(abs(pr_significativo)>2.0032)
evspsbl_significativo=(abs(evspsbl_significativo)>2.0032)


#quiero calcular los cambios porcentuales fututo-presente
#primero defino un arreglo donde guardo los cambios porcentuales respecto al periodo
#historico para cada estacion
pr_delta_porcentual=np.zeros((4,4,10,31,22),dtype='float32')
evspsbl_delta_porcentual=np.zeros((4,4,10,31,22),dtype='float32')

for i in range(0,4):
 pr_delta_porcentual[i]=((pr_medias[i,1:5,:,:,:]-pr_medias[i,0,:,:,:])/pr_medias[i,0,:,:,:])*100
 evspsbl_delta_porcentual[i]=((evspsbl_medias[i,1:5,:,:,:]-evspsbl_medias[i,0,:,:,:])/evspsbl_medias[i,0,:,:,:])*100


#calculo la media de las diferencias porcentuales entre todos los miembros (axis=2)
pr_delta_porcentual=np.mean(pr_delta_porcentual,axis=2)
evspsbl_delta_porcentual=np.mean(evspsbl_delta_porcentual,axis=2)


#vuelvo a hacer los recortes y las sumas estacionales, separo en estaciones,y quiero
#todos los años por estacion
def estaciones1(variable,var,periodo):
    #recorto cada variable sobre sudamerica
    print(variable)
    if var=='hus':
        variable=cdo.sellevel('100000',input=variable)
    else:
        variable=variable
        
    sudamerica=cdo.sellonlatbox('-85,-32,-60,15',input=variable)
    lats=cdo.sellonlatbox('-85,-32,-60,15',input=variable,returnArray='lat')
    lons=cdo.sellonlatbox('-85,-32,-60,15',input=variable,returnArray='lon')
    
    #calculo las sumatorias por cada estacion
    estacional=cdo.seasmean(input=sudamerica)
    
    #separo por estacion
    DEF=cdo.selyear(periodo,input='-selmonth,1 '+estacional,returnArray=var)
    MAM=cdo.selmonth('4',input=estacional,returnArray=var)
    JJA=cdo.selmonth('7',input=estacional,returnArray=var)
    SON=cdo.selmonth('10',input=estacional,returnArray=var)

    return DEF,MAM,JJA,SON,lats,lons

periodo=['1977/2005','2021/2049','2021/2049','2071/2099','2071/2099']

#defino arreglos donde voy a guardar todos los años para cada estación
pr_DEF=np.zeros((5,10,29,31,22),dtype='float32')
pr_MAM=np.zeros((5,10,30,31,22),dtype='float32')
pr_JJA=np.zeros((5,10,30,31,22),dtype='float32')
pr_SON=np.zeros((5,10,30,31,22),dtype='float32')

evspsbl_DEF=np.zeros((5,10,29,31,22),dtype='float32')
evspsbl_MAM=np.zeros((5,10,30,31,22),dtype='float32')
evspsbl_JJA=np.zeros((5,10,30,31,22),dtype='float32')
evspsbl_SON=np.zeros((5,10,30,31,22),dtype='float32')

#para cada archivo llamo la funcion estaciones1
for i in range(len(pr)):
    for j in range(len(pr[0])):
     pr_DEF[i][j],pr_MAM[i][j],pr_JJA[i][j],pr_SON[i][j]=estaciones1(pr[i][j], 'pr', periodo[i])
     evspsbl_DEF[i][j],evspsbl_MAM[i][j],evspsbl_JJA[i][j],evspsbl_SON[i][j]=estaciones1(evspsbl[i][j], 'evspsbl', periodo[i])


#calculo las diferencias entre P y E para cada estacion
diferencia_def=pr_DEF-evspsbl_DEF
diferencia_mam=pr_MAM-evspsbl_MAM
diferencia_jja=pr_JJA-evspsbl_JJA
diferencia_son=pr_SON-evspsbl_SON

#hago el promedio de ensamble
diferencia_def_ens=np.mean(diferencia_def,axis=1)
diferencia_mam_ens=np.mean(diferencia_mam,axis=1)
diferencia_jja_ens=np.mean(diferencia_jja,axis=1)
diferencia_son_ens=np.mean(diferencia_son,axis=1)

#hago el promedio temporal de las diferencias 
diferencia_def_ens=np.mean(diferencia_def_ens,axis=1)
diferencia_mam_ens=np.mean(diferencia_mam_ens,axis=1)
diferencia_jja_ens=np.mean(diferencia_jja_ens,axis=1)
diferencia_son_ens=np.mean(diferencia_son_ens,axis=1)

#Junto los campos de las diferencias del periodo historico
diferencias_historico=np.stack((diferencia_def_ens[0,:,:],diferencia_mam_ens[0,:,:],diferencia_jja_ens[0,:,:],diferencia_son_ens[0,:,:]),axis=0)

#calculo los cambios de P-E, futuro-historico para cada una de las estaciones
cambio_pe_def=diferencia_def_ens[1:5,:,:]-diferencia_def_ens[0,:,:]
cambio_pe_mam=diferencia_mam_ens[1:5,:,:]-diferencia_mam_ens[0,:,:]
cambio_pe_jja=diferencia_jja_ens[1:5,:,:]-diferencia_jja_ens[0,:,:]
cambio_pe_son=diferencia_son_ens[1:5,:,:]-diferencia_son_ens[0,:,:]

########### BOWEN #################################
#defino arreglos donde voy a guardar todos los años para cada estación (para cmip6 uso las dos primeras
#realizaciones para tas y hus)
tas_DEF=np.zeros((5,29,31,22),dtype='float32')
tas_MAM=np.zeros((5,30,31,22),dtype='float32')
tas_JJA=np.zeros((5,30,31,22),dtype='float32')
tas_SON=np.zeros((5,30,31,22),dtype='float32')

hus_DEF=np.zeros((5,29,1,31,22),dtype='float32')
hus_MAM=np.zeros((5,30,1,31,22),dtype='float32')
hus_JJA=np.zeros((5,30,1,31,22),dtype='float32')
hus_SON=np.zeros((5,30,1,31,22),dtype='float32')

#llamo a la funcion estaciones1 para calcular por estaciones el promedio de tas y hus 
#HAY QUE CAMBIAR EN LA FUNCION ESTACIONES1 LA PARTE DONDE EN VEZ DE CALCULAR LA SUMATORIA
#ESTACIONAL, CAMBIARLA POR EL PROMEDIO ESTACIONAL!!
for i in range(len(tas)):
    #for j in range(len(tas[0])):
     tas_DEF[i],tas_MAM[i],tas_JJA[i],tas_SON[i],lats,lons=estaciones1(tas[i], 'tas', periodo[i])
     hus_DEF[i],hus_MAM[i],hus_JJA[i],hus_SON[i]=estaciones1(hus[i],'hus', periodo[i])[0:4]


hus_DEF=np.squeeze(hus_DEF)
hus_MAM=np.squeeze(hus_MAM)
hus_JJA=np.squeeze(hus_JJA)
hus_SON=np.squeeze(hus_SON)


#calculo el calor latente y el calor sensible para cada una de las estaciones
calor_sensible_DEF=1.005*tas_DEF
calor_sensible_MAM=1.005*tas_MAM
calor_sensible_JJA=1.005*tas_JJA
calor_sensible_SON=1.005*tas_SON

calor_latente_DEF=2500*hus_DEF
calor_latente_MAM=2500*hus_MAM
calor_latente_JJA=2500*hus_JJA
calor_latente_SON=2500*hus_SON

#calculo bowen para cada estacion
bowen_def=calor_sensible_DEF/calor_latente_DEF
bowen_mam=calor_sensible_MAM/calor_latente_MAM
bowen_jja=calor_sensible_JJA/calor_latente_JJA
bowen_son=calor_sensible_SON/calor_latente_SON

#hago el promedio de ensamble 
bowen_def_ens=np.mean(bowen_def,axis=1)
bowen_mam_ens=np.mean(bowen_mam,axis=1)
bowen_jja_ens=np.mean(bowen_jja,axis=1)
bowen_son_ens=np.mean(bowen_son,axis=1)

#hago el promedio temporal
bowen_def_ens=np.mean(bowen_def_ens,axis=1)
bowen_mam_ens=np.mean(bowen_mam_ens,axis=1)
bowen_jja_ens=np.mean(bowen_jja_ens,axis=1)
bowen_son_ens=np.mean(bowen_son_ens,axis=1)

#Junto los campos de bowen del periodo historico
bowen_historico=np.stack((bowen_def_ens[0,:,:],bowen_mam_ens[0,:,:],bowen_jja_ens[0,:,:],bowen_son_ens[0,:,:]),axis=0)

#calculo bowen futuro/bowen historico para cada estacion
bowen_fh_def=bowen_def_ens[1:5,:,:]/bowen_def_ens[0,:,:]
bowen_fh_mam=bowen_mam_ens[1:5,:,:]/bowen_mam_ens[0,:,:]
bowen_fh_jja=bowen_jja_ens[1:5,:,:]/bowen_jja_ens[0,:,:]
bowen_fh_son=bowen_son_ens[1:5,:,:]/bowen_son_ens[0,:,:]

#calculo el promedio de ensamble y temporal de los calores latente y sensible
cl_def_ens=np.mean(calor_latente_DEF,axis=1)
cl_mam_ens=np.mean(calor_latente_MAM,axis=1)
cl_jja_ens=np.mean(calor_latente_JJA,axis=1)
cl_son_ens=np.mean(calor_latente_SON,axis=1)

cs_def_ens=np.mean(calor_sensible_DEF,axis=1)
cs_mam_ens=np.mean(calor_sensible_MAM,axis=1)
cs_jja_ens=np.mean(calor_sensible_JJA,axis=1)
cs_son_ens=np.mean(calor_sensible_SON,axis=1)



#calculo el cociente entre el calor latente pasado y futuro
cl_fh_def=cl_def_ens[0,:,:]/cl_def_ens[1:5,:,:]
cl_fh_mam=cl_mam_ens[0,:,:]/cl_mam_ens[1:5,:,:]
cl_fh_jja=cl_jja_ens[0,:,:]/cl_jja_ens[1:5,:,:]
cl_fh_son=cl_son_ens[0,:,:]/cl_son_ens[1:5,:,:]

#calculo el cociente entre el calor sensible futuro y pasado
cs_fh_def=cs_def_ens[1:5,:,:]/cs_def_ens[0,:,:]
cs_fh_mam=cs_mam_ens[1:5,:,:]/cs_mam_ens[0,:,:]
cs_fh_jja=cs_jja_ens[1:5,:,:]/cs_jja_ens[0,:,:]
cs_fh_son=cs_son_ens[1:5,:,:]/cs_son_ens[0,:,:]



###################### GRAFICOS ################################################
titulo_estacion=['Calor Latente DEF ','Calor Latente MAM ','Calor Latente JJA ','Calor Latente SON ']
titulo_delta_periodo=['Histórico/Cercano (RCP26)','Histórico/Cercano (RCP85)','Histórico/Lejano (RCP26)','Histórico/Lejano (RCP85)']
historico=['Histórico (CMIP5)']

#for i in range(0,4):
 #fig = plt.figure(figsize=(15,13))
 #variable, lons_1 = add_cyclic_point(delta_dif_PETP[i,:,:], coord=lons,axis=1)
 #ax.set_extent([-85,-33,-60,10],ccrs.PlateCarree())
for j in range(0,4):
  fig = plt.figure(figsize=(15,13))
  cmap='PuOr_r'
  levels=np.arange(0.85,1.2,0.05)
  ax = plt.axes(projection=ccrs.PlateCarree())
  cf=plt.contourf(lons, lats, cl_fh_son[j,:,:], 
             transform=ccrs.PlateCarree(),cmap=cmap,levels=levels,extend='both')
  #x,y=np.meshgrid(lons,lats)
  #x1=np.ma.masked_array(x,~evspsbl_significativo[i,j,:,:]).filled(np.nan)
  #y1=np.ma.masked_array(y,~evspsbl_significativo[i,j,:,:]).filled(np.nan)
  #points=plt.scatter(x1,y1,s=10,c='k',transform=ccrs.PlateCarree())
  ax.add_feature(cfeature.OCEAN, zorder=110, edgecolor='k',facecolor='white')
 
# gridlines = ax.gridlines()
  gl=ax.gridlines(crs=ccrs.PlateCarree(),draw_labels=True,alpha=0.5,)
  gl.xlabels_top=False
  gl.ylabels_right=False
  gl.xlocator=mticker.FixedLocator(np.arange(-180,180+1,15))
  gl.ylocator=mticker.FixedLocator(np.arange(-90,90,15))
  gl.xformatter=LONGITUDE_FORMATTER
  gl.yformatter=LATITUDE_FORMATTER
  gl.xlabel_style = {'size': 15}
  gl.ylabel_style = {'size': 15}

  ax.add_feature(cfeature.COASTLINE,linewidth=0.6)
  ax.add_feature(cfeature.BORDERS,linewidth=0.5)

  cb = plt.colorbar(cf, shrink=0.3)
  cb.ax.tick_params(labelsize=15)
  #cb.ax.set_title('mm',fontsize=15)
  ax.set_title(titulo_estacion[3]+titulo_delta_periodo[j],fontsize=20)
  plt.savefig('/home/jesica/Documentos/tpfinal_figuras/lh_fut-hist_son_c5'+str(j)+'.png',bbox_inches='tight')


    
        
    
        
        
  
