#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug  7 15:40:12 2020

@author: jesica
"""

import xarray as xr
import glob

path='/home/jesica/Documentos/clima-dinamica/Datos/MPI_ESM_LR/'

tas_hist_archivos=sorted(glob.glob(path+'tas_Amon_MPI-ESM-LR_historical_*_197601-200512_2.5_anu.nc'))

tas_hist=xr.open_mfdataset(tas_hist_archivos, concat_dim='miembros',combine='nested' )

#convierto las longitudes de 0-360 a -180-180
tas_hist= tas_hist.assign_coords(lon=(((tas_hist.lon + 180) % 360) - 180)).sortby('lon')
