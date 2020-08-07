#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  3 14:14:22 2020

@author: jesica
"""


from cdo import *
cdo=Cdo()

import glob

import numpy as np
import numpy.ma as ma
from netCDF4 import  num2date
#import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
#import seaborn as sns
#sns.set()
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER

import matplotlib.ticker as mticker

ruta='/home/jesica/Documentos/clima-dinamica/Datos/MPI_ESM_LR/'

evspsbl_7099=[]
evspsbl_hist=[]

pr_7099=[]
pr_hist=[]

evspsbl_7099.append(sorted(glob.glob(ruta+'evspsbl_Amon_MPI-ESM-LR_rcp85_*_207001-209912_2.5_anu.nc')))
evspsbl_hist.append(sorted(glob.glob(ruta+'evspsbl_Amon_MPI-ESM-LR_historical_*_197601-200512_2.5_anu.nc')))

pr_7099.append(sorted(glob.glob(ruta+'pr_Amon_MPI-ESM-LR_rcp85_*_207001-209912_2.5_anu.nc')))
pr_hist.append(sorted(glob.glob(ruta+'pr_Amon_MPI-ESM-LR_historical_*_197601-200512_2.5_anu.nc')))


def sudamerica(archivo,var):
    sudamerica=cdo.sellonlatbox('-85,-32,-60,15',input=archivo,options='-f nc',returnCdf=True)
    lat=cdo.sellonlatbox('-85,-32,-60,15',input=archivo,returnArray='lat')
    lon=cdo.sellonlatbox('-85,-32,-60,15',input=archivo,returnArray='lon')
    time=sudamerica.variables['time']
    dates=num2date(time[:],units=time.units,calendar=time.calendar)
    sudamerica=sudamerica.variables[var][:]
    
    return sudamerica,lat,lon,dates


lat,lon,time=sudamerica(evspsbl_hist[0][0],'evspsbl')[1:4]

evspsbl_historico_sud=np.zeros((len(evspsbl_hist[0]),len(time),len(lat),len(lon)),dtype='float32')
evspsbl_7099_sud=np.zeros((len(evspsbl_hist[0]),len(time),len(lat),len(lon)),dtype='float32')
pr_historico_sud=np.zeros((len(evspsbl_hist[0]),len(time),len(lat),len(lon)),dtype='float32')
pr_7099_sud=np.zeros((len(evspsbl_hist[0]),len(time),len(lat),len(lon)),dtype='float32')


for i in range(len(evspsbl_hist)):
    evspsbl_historico_sud[i]=sudamerica(evspsbl_hist[0][i],'evspsbl')[0]
    evspsbl_7099_sud[i]=sudamerica(evspsbl_7099[0][i],'evspsbl')[0]
    pr_historico_sud[i]=sudamerica(pr_hist[0][i],'pr')[0]
    pr_7099_sud[i]=sudamerica(pr_7099[0][i],'pr')[0]
    

pe_hist=pr_historico_sud-evspsbl_historico_sud
pe_7099=pr_7099_sud-evspsbl_7099_sud


#promedio temporal de cada campo
pe_hist_temp=np.mean(pe_hist,axis=1)
pe_7099_temp=np.mean(pe_7099,axis=1)

delta_pe=pe_7099_temp-pe_hist_temp

#media ensamble
delta_pe_ens=np.mean(delta_pe,axis=0)

fig = plt.figure(figsize=(15,13))
cmap='RdYlBu'
levels=np.arange(-100,125,25)
ax = plt.axes(projection=ccrs.PlateCarree())
cf=plt.contourf(lon, lat, delta_pe_ens, 
             transform=ccrs.PlateCarree(),cmap=cmap,levels=levels,extend='both')
  #x,y=np.meshgrid(lons,lats)
  #x1=np.ma.masked_array(x,~pr_significativo[i,j,:,:]).filled(np.nan)
  #y1=np.ma.masked_array(y,~pr_significativo[i,j,:,:]).filled(np.nan)
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
cb.ax.set_title('mm',fontsize=15)
ax.set_title('$\Delta$(P-E) Anual Lejano-Histórico (RCP8.5)',fontsize=20)