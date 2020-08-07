#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug  2 20:48:00 2020

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

ruta='/home/jesica/Documentos/clima-dinamica/Datos/MPI_ESM_LR/'

tas_historico=ruta+'tas_Amon_MPI-ESM-LR_historical_r1_2.5.nc'

tas_7099_85=ruta+'tas_Amon_MPI-ESM-LR_rcp85_r1_2070-2099_2.5.nc'

def estaciones(archivo,var,periodo):
    #recorto el archivo en sudamerica
    sudamerica=cdo.sellonlatbox('-85,-32,-60,15',input=archivo)
    lat=cdo.sellonlatbox('-85,-32,-60,15',input=archivo,returnArray='lat')
    lon=cdo.sellonlatbox('-85,-32,-60,15',input=archivo,returnArray='lon')
    
    #calculo la media estacional
    estacional=cdo.seasmean(input=sudamerica)
    info=print(cdo.sinfo(input=estacional))
    #separo en estaciones
    DEF=cdo.selyear(periodo,input='-selmonth,1 '+estacional,returnArray=var)
    MAM=cdo.selyear(periodo,input='-selmonth,4 '+estacional,returnArray=var)
    JJA=cdo.selyear(periodo,input='-selmonth,7 '+estacional,returnArray=var)
    SON=cdo.selyear(periodo,input='-selmonth,10 '+estacional,returnArray=var)
    
    return DEF,MAM,JJA,SON,lat,lon


def_hist,mam_hist,jja_hist,son_hist,lats,lons=estaciones(tas_historico,'tas','1977/2005')
def_7099,mam_7099,jja_7099,son_7099=estaciones(tas_7099_85,'tas','2071/2099')[0:4]


son_sensible_hist=1.005*son_hist
son_sensible_7099=1.005*son_7099

media_sensible_hist=np.mean(son_sensible_hist,axis=0)
media_sensible_7099=np.mean(son_sensible_7099,axis=0)

cociente_sensible_son=media_sensible_7099/media_sensible_hist


