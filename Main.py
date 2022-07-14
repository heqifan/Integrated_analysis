# -*- coding: utf-8 -*-
"""
Created on Mon Jul  4 20:58:11 2022

@author: HYF
"""

from multiprocessing import cpu_count,Pool
#print(cpu_count())
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os,sys
from glob import glob as g
from osgeo import osr,ogr,gdal
from tkinter import _flatten
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
process_num = cpu_count()
import datetime
import copy
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

MuSyQ_inpath = r'F:\Integrated_analysis_data\Data\Geodata_1981_2018_1y'
GLASS_inpath = r'F:\Integrated_analysis_data\Data\GLASS_1982_2018_1y'
MODIS_path = r'F:\Integrated_analysis_data\Data\MODIS_2000_2020_1y'
CASA_path = r'F:\Integrated_analysis_data\Data\TPDC_2000_2017_1y'
W_path = r'F:\Integrated_analysis_data\Data\W_1980_2018_1y'


length = 5
styear = 2000
edyear = 2017
MuSyQ_datas,GLASS_datas,MODIS_datas,CASA_datas,W_datas = [],[],[],[],[]
MuSyQ_key,GLASS_key,MODIS_key,CASA_key,W_key =  'Sum_*.tif','Mul_*.tif','Mul_*.tif','Resample_*.tif','Resample_*.tif'
minxsize = 0
minysize = 0
'''Mean'''

def Mean(MuSyQ_datas,GLASS_datas,MODIS_datas,CASA_datas,W_datas,years,length):
    
    images_pixels = []
    for year in range(len(MuSyQ_datas)):
        images_pixels.append((MuSyQ_datas[year] + GLASS_data[year] + MODIS_data[year] + CASA_data[year] + W_data[year])/length)
    global minxsize
    minxsize = min([x.shape[0] for x in images_pixels])
    global minysize
    minysize  = min([x.shape[1] for x in images_pixels])
    print('x_min:',minxsize,'y_min:',minysize)
    R2 = copy.deepcopy(images_pixels[-1]) # 获取一个矩阵作为要写入的模板
    for i in range(minxsize):
        for j in range(minysize):
            mean_data = [] # 存放ij坐标下像元值的数组，以计算R2，RMSE
            for px in range(len(images_pixels)): # 遍历多个图像下ij坐标的像元值
                mean_data.append(images_pixels[px][i][j]) # 同一坐标的多点加入数组，以计算
            x_train,x_test,y_train,y_test = train_test_split(years,mean_data,train_size=0.8)
            
            forest_reg = RandomForestRegressor(random_state=41)
            forest_reg.fit(np.array(x_train).reshape(-1,1),np.array(y_train))
            y_predict = forest_reg.predict(np.array(x_test).reshape(-1,1))
            r2 = r2_score(y_test, y_predict)
            print("r2 score:", r2)
            print(f'x_train:{x_train} x_test:{x_test} y_train:{y_train} y_test:{y_test}')
            R2[i][j] = r2 # 写入该坐标下的变异系数
    return R2





years = [year for year in range(styear,edyear+1)]


for year in range(styear,edyear+1):
    
    MuSyQ_dir,GLASS_dir = MuSyQ_inpath + os.sep + str(year),GLASS_inpath + os.sep + str(year)
    MODIS_dir,CASA_dir = MODIS_path + os.sep + str(year),CASA_path + os.sep + str(year)
    W_dir = W_path + os.sep + str(year)
    
    MuSyQ_,GLASS_ = gdal.Open(g(MuSyQ_dir + os.sep + MuSyQ_key)[0]),gdal.Open(g(GLASS_dir + os.sep + GLASS_key)[0])
    MODIS_,CASA_ = gdal.Open(g(MODIS_dir + os.sep + MODIS_key)[0]),gdal.Open(g(CASA_dir + os.sep + CASA_key)[0])
    W_ = gdal.Open(g(W_dir + os.sep + W_key)[0])
    
    
    minxsize = min([MuSyQ_.RasterXSize,GLASS_.RasterXSize,MODIS_.RasterXSize,CASA_.RasterXSize,W_.RasterXSize])
    minysize = min([MuSyQ_.RasterYSize,GLASS_.RasterYSize,MODIS_.RasterYSize,CASA_.RasterYSize,W_.RasterYSize])
    
    print(f'{year}最小的x为：{minxsize}',f'最小的y为：{minysize}')
    
    MuSyQ_data = MuSyQ_.ReadAsArray(0, 0, minxsize, minysize)
    GLASS_data = GLASS_.ReadAsArray(0, 0, minxsize, minysize)
    MODIS_data = MODIS_.ReadAsArray(0, 0, minxsize, minysize)
    CASA_data = CASA_.ReadAsArray(0, 0, minxsize, minysize)
    W_data = W_.ReadAsArray(0, 0, minxsize, minysize)
    #mean = Mean(MuSyQ_data,GLASS_data,MODIS_data,CASA_data,W_data)
    MuSyQ_datas.append(MuSyQ_data),GLASS_datas.append(GLASS_data),MODIS_datas.append(MODIS_data),CASA_datas.append(CASA_data),W_datas.append(W_data)
    
    del MuSyQ_ 
    del GLASS_ 
    del MODIS_ 
    del CASA_
    
# pool = Pool(processes=process_num-5)    
# data = pool.apply_async(func = Mean,args = (MuSyQ_datas,GLASS_datas,MODIS_datas,CASA_datas,W_datas))
# pool.close()
# pool.join()
start = datetime.datetime.now()
Mean_R2 = Mean(MuSyQ_datas,GLASS_datas,MODIS_datas,CASA_datas,W_datas,years,length)
end = datetime.datetime.now()
print('totally time is ', end - start)
    

    
    
    