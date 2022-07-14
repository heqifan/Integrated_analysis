# -*- coding: utf-8 -*-
"""
Created on Mon Jul 11 12:42:15 2022

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
from sklearn.metrics import mean_absolute_error
from math import sqrt

from loguru import logger


Outpath = r'F:\Integrated_analysis_data\Data\Out'
MuSyQ_inpath = r'F:\Integrated_analysis_data\Data\Geodata_1981_2018_1y'
GLASS_inpath = r'F:\Integrated_analysis_data\Data\GLASS_1982_2018_1y'
MODIS_path = r'F:\Integrated_analysis_data\Data\MODIS_2000_2020_1y'
CASA_path = r'F:\Integrated_analysis_data\Data\TPDC_2000_2017_1y'
W_path = r'F:\Integrated_analysis_data\Data\W_1980_2018_1y'
MuSyQ_key,GLASS_key,MODIS_key,CASA_key,W_key =  'Sum_*.tif','Mul_*.tif','Mul_*.tif','Resample_*.tif','Resample_*.tif'
nodatakey = ['>100000','<-100000','<-100000','>60000','<-100000']


length = 5
styear = 2000
edyear = 2017
MuSyQ_datas,GLASS_datas,MODIS_datas,CASA_datas,W_datas = [],[],[],[],[]
processes = 7

years = [year for year in range(styear,edyear+1)]


var = ['R2','RMSE','MSE','MAE']   


class Startwork():
    
    def  SetNodata(self,Datas,nodatakey):
        i=0
        datas_ = []
        for data in Datas:
            data_ = []
            for da in data:
                da = pd.DataFrame(da)
                da.fillna(0,inplace = True)
                da = np.array(da) 
                symbol = nodatakey[i][0]
                value = int(nodatakey[i][1:])
                # print(value)
                if symbol == '>':
                    # print('>')
                    da[da>=value] = 0
                    da[da<0] = 0
                    # if da.all() < value and da.all() >= 0:
                    #     #print(f"数组中所有元素都大于等于0且小于{value}")
                    # elif  da.any() >= value:
                    #     #print(f"数组中所有元素还存在大于等于{value}")
                    # elif da.any()<0:
                    #     #print(f"数组中所有元素还存在小于0的值")
                    # #da = np.where(da>float(nodatakey[i][1:]),np.nan,da)
                if symbol == '<':
                    # print('<')
                    da[da<=value] = 0
                    da[da<0] = 0
                    # if da.all() > value and da.all()>=0:
                    #     #print(f"数组中所有元素都大于等于0且大于{value}")
                    # elif  da.any() <= value:
                    #     #print(f"数组中所有元素还存在小于等于{value}的值")
                    # elif da.any()<0:
                    #     #print(f"数组中所有元素还存在小于0的值")
                data_.append(da)
                    #da = np.where(da<float(nodatakey[i][1:]),np.nan,da)
            i+=1
            datas_.append(data_)
        
        return datas_
        
        
        
        
    def  SetDatatype(self,Datas):
        datas_ = []
        for data in Datas:
            data_ = []
            for da in data:
                #print('data:',data)
                #da.astype('uint16')
                print(f'原始的数据类型为{da.dtype}')
                da.dtype = np.uint32
                print(f'数据类型更改为 {da.dtype}')
                data_.append(da)
            datas_.append(data_)
        return datas_
        
        
        
    def  WriteArray(self,datalist,Outpath,minxsize,minysize,var,Type):
    
        sample_tif = r'F:\Integrated_analysis_data\Data\TPDC_2000_2017_1y\2000\Resample_2000_npp.tif'                          # 需要读取的tif文件所在的文件夹的所在文件夹的路径    
        ds = gdal.Open(sample_tif)                             # 打开文件
        im_width = minxsize                           # 获取栅格矩阵的列数
        print(f'im_width: {im_width}')
        im_height = minysize                         # 获取栅格矩阵的行数
        print(f'im_height: {im_height}')
        im_bands = ds.RasterCount                           # 获取栅格矩阵的波段数
        #print(f'im_bands: {im_bands}')
        band1 = ds.GetRasterBand(1)                         # 波段的indice起始为1，不为0
        img_datatype = band1.DataType                       # 数据类型
        
        outdir = Outpath + os.sep + Type
        print(f'-------输出文件夹为 {outdir}---------')
        if not os.path.exists(outdir):       #判断原始文件路劲是否存在,如果不存在就直接退出
            os.makedirs(outdir)
        for j in range(0,len(datalist)):    
            out_ds = gdal.GetDriverByName('GTiff').Create(
                  outdir + os.sep + var[j] + '.tif',                   # tif文件所保存的路径
                  im_width,                                     # 获取栅格矩阵的列数
                  im_height,                                     # 获取栅格矩阵的行数
                  ds.RasterCount,                                     # 获取栅格矩阵的波段数
                  img_datatype)                                       # 获取第一波段的数据类型
            out_ds.SetProjection(ds.GetProjection())                # 投影信息
            out_ds.SetGeoTransform(ds.GetGeoTransform())            # 仿射信息
            for i in range(1, ds.RasterCount + 1):                  # 循环逐波段写入
                out_band = out_ds.GetRasterBand(i)
                out_band.WriteArray(datalist[j])                           # 写入数据 (why)
                out_ds.FlushCache()  #(刷新缓存)
                del out_ds #删除 
        
                print(f' {outdir + os.sep + var[j] + ".tif"} is  ok   !!!!!!!!')
        del ds
        
        
        
        
        
    def RR(self,mean_data):
        logger.info('%s run' %os.getpid())
        print('%s run' %os.getpid())
        y_data = mean_data
        years2 = years
        x_train,x_test,y_train,y_test = train_test_split(years2,y_data,train_size=0.8)
        print(f'x_train:{x_train}')
        print(f'x_test:{x_test}')
        print(f'y_train:{y_train}')
        print(f'y_test:{y_test}')
        forest_reg = RandomForestRegressor(random_state=41)
        forest_reg.fit(np.array(x_train).reshape(-1,1),np.array(y_train))
        y_predict = forest_reg.predict(np.array(x_test).reshape(-1,1))
        r2 = r2_score(y_test, y_predict)
        mse =  mean_squared_error(y_test, y_predict)
        mae = mean_absolute_error(y_test, y_predict)
        rmse = sqrt(mse)
        
        print(f"r2 score: {r2}")
        print(f"rmse: {rmse}")
        print(f"mean_squared_error: {mse}" )
        print(f"mean_absolute_error: {mae}")
        print(f'x_train: {x_train} x_test: {x_test} y_train: {y_train} y_test: {y_test}')
        
        return r2,rmse,mse,mae
            
    




if __name__ == "__main__": 
    print('-----------------Start----------------------')
    minxsizes_list  = []
    minysizes_list  = []
    
    for year in range(styear,edyear+1):
        MuSyQ_dir,GLASS_dir = MuSyQ_inpath + os.sep + str(year),GLASS_inpath + os.sep + str(year)
        MODIS_dir,CASA_dir = MODIS_path + os.sep + str(year),CASA_path + os.sep + str(year)
        W_dir = W_path + os.sep + str(year)
        
        MuSyQ_,GLASS_ = gdal.Open(g(MuSyQ_dir + os.sep + MuSyQ_key)[0],gdal.GA_ReadOnly),gdal.Open(g(GLASS_dir + os.sep + GLASS_key)[0],gdal.GA_ReadOnly)
        MODIS_,CASA_ = gdal.Open(g(MODIS_dir + os.sep + MODIS_key)[0],gdal.GA_ReadOnly),gdal.Open(g(CASA_dir + os.sep + CASA_key)[0],gdal.GA_ReadOnly)
        W_ = gdal.Open(g(W_dir + os.sep + W_key)[0])
        
        minxsize = min([MuSyQ_.RasterXSize,GLASS_.RasterXSize,MODIS_.RasterXSize,CASA_.RasterXSize,W_.RasterXSize])
        minysize = min([MuSyQ_.RasterYSize,GLASS_.RasterYSize,MODIS_.RasterYSize,CASA_.RasterYSize,W_.RasterYSize])
        
        minxsizes_list.append(minxsize)
        minysizes_list.append(minysize)
        
        del MuSyQ_ 
        del GLASS_ 
        del MODIS_ 
        del CASA_
        
    minx_minx = min(minxsizes_list)
    miny_miny = min(minysizes_list)
        
    
    print(f'minx_minx: {minx_minx}')
    print(f'miny_miny: {miny_miny}')
    
    
    for year in range(styear,edyear+1):
        
        MuSyQ_dir,GLASS_dir = MuSyQ_inpath + os.sep + str(year),GLASS_inpath + os.sep + str(year)
        MODIS_dir,CASA_dir = MODIS_path + os.sep + str(year),CASA_path + os.sep + str(year)
        W_dir = W_path + os.sep + str(year)
        
        MuSyQ_,GLASS_ = gdal.Open(g(MuSyQ_dir + os.sep + MuSyQ_key)[0],gdal.GA_ReadOnly),gdal.Open(g(GLASS_dir + os.sep + GLASS_key)[0],gdal.GA_ReadOnly)
        MODIS_,CASA_ = gdal.Open(g(MODIS_dir + os.sep + MODIS_key)[0],gdal.GA_ReadOnly),gdal.Open(g(CASA_dir + os.sep + CASA_key)[0],gdal.GA_ReadOnly)
        W_ = gdal.Open(g(W_dir + os.sep + W_key)[0])
        
        
        MuSyQ_data = MuSyQ_.ReadAsArray(0, 0, minx_minx, miny_miny)
        GLASS_data = GLASS_.ReadAsArray(0, 0, minx_minx, miny_miny)
        MODIS_data = MODIS_.ReadAsArray(0, 0, minx_minx, miny_miny)
        CASA_data = CASA_.ReadAsArray(0, 0, minx_minx, miny_miny)
        W_data = W_.ReadAsArray(0, 0, minx_minx, miny_miny)
        MuSyQ_datas.append(MuSyQ_data),GLASS_datas.append(GLASS_data),MODIS_datas.append(MODIS_data),CASA_datas.append(CASA_data),W_datas.append(W_data)
        
        del MuSyQ_ 
        del GLASS_ 
        del MODIS_ 
        del CASA_
         
    Datas = [MuSyQ_datas,GLASS_datas,MODIS_datas,CASA_datas,W_datas]
    
    print('开始设置无效值')
    Setnodata_Datas = Startwork().SetNodata(Datas,nodatakey)
    # Setnodata_Datas = SetNodata(Datas,nodatakey)
    
    print('开始设置数据类型')
    #SetDatatypes = SetDatatype(setnodata_Datas)
    
    start = datetime.datetime.now()
    print(f'开始时间为：{start}')
    
    print('开始计算Mean_R2,Mean_RMSE,Meam_MSE,Mean_MAE')
    # Mean(Setnodata_Datas,Outpath,var,minx_minx,miny_miny)
    # def Mean(datas,Outpath,var,minx_minx,miny_miny):
    images_pixels = []
    #datas = SetDatatype(datas)
    for year in range(len(years)):
        images_pixels.append((Setnodata_Datas[0][year].reshape(miny_miny,minx_minx) + Setnodata_Datas[1][year].reshape(miny_miny,minx_minx) + Setnodata_Datas[2][year].reshape(miny_miny,minx_minx) + Setnodata_Datas[3][year].reshape(miny_miny,minx_minx) + Setnodata_Datas[4][year].reshape(miny_miny,minx_minx))/len(years))
    
    images = []
    for image in images_pixels:
        #print(image)
        images.append(list(image.flatten()))
    images_array = pd.DataFrame(images).T
    print(images_array.shape)
    #images_pixels_2 = [list(x.flatten()) for x in images_pixels]
    #print(images_pixels_2.shape)
    print('-------------开启进程池---------------')
    with Pool(processes) as  pool:
        All_data = pool.map(Startwork().RR(), images_array)
    print('------------关闭进程池----------------')
    datalist = [pd.DataFrame(All_data)[:0],pd.DataFrame(All_data)[:1],pd.DataFrame(All_data)[:2],pd.DataFrame(All_data)[:3]]
    Startwork().WriteArray(datalist,Outpath,minx_minx,miny_miny,var,'Mean')
    print('结束')
    
    end = datetime.datetime.now()
    print('totally time is ', end - start)
    

    
    
    