# -*- coding: utf-8 -*-
"""
Created on Mon Jul  4 20:58:11 2022

@author: HYF
"""
from tqdm import tqdm
import PySimpleGUI as sg
import pandas as pd
import numpy as np
import os
from glob import glob as g
from osgeo import gdal
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import datetime
from sklearn.metrics import mean_squared_error,r2_score,mean_absolute_error
from math import sqrt
import time
import random
import logging
from sklearn import linear_model
from sklearn.linear_model import LinearRegression   #引入多元线性回归算法模块进行相应的训练


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [*] %(processName)s %(message)s"
)



Outpath = r'F:\Integrated_analysis_data\Data\Out'
MuSyQ_inpath = r'F:\Integrated_analysis_data\Data\1Y\Geodata_2000_2017_1y'
GLASS_inpath = r'F:\Integrated_analysis_data\Data\1Y\GLASS_2000_2017_1y'
MODIS_path = r'F:\Integrated_analysis_data\Data\1Y\MODIS_2000_2017_1y'
CASA_path = r'F:\Integrated_analysis_data\Data\1Y\TPDC_2000_2017_1y'
W_path = r'F:\Integrated_analysis_data\Data\1Y\W_2000_2017_1y'
LAI_path = r'F:\Integrated_analysis_data\Data\1Y\LAI_2003_2017_1y'

MuSyQ_R2 = r'F:\Integrated_analysis_data\Data\Out\R2_Geodata\R2.tif'
GLASS_R2 = r'F:\Integrated_analysis_data\Data\Out\R2_GLASS\R2.tif'
MODIS_R2 = r'F:\Integrated_analysis_data\Data\Out\R2_MODIS\R2.tif'
CASA_R2 = r'F:\Integrated_analysis_data\Data\Out\R2_TPDC\R2.tif'
W_R2 = r'F:\Integrated_analysis_data\Data\Out\R2_W\R2.tif'

MuSyQ_key,GLASS_key,MODIS_key,CASA_key,W_key,LAI_key =  'Mul_*.tif','Mul_*.tif','Mul_*.tif','Reproject_*.tif','Resample_*.tif','Reproject_*.tif'

# MuSyQ_key,GLASS_key,CASA_key,W_key =  'Mul_*.tif','Mul_*.tif','Reproject_*.tif','Resample_*.tif'
nodatakey = ['<-1000','<-1000','<-1000','<-1000','<-1000','<-1000']

na_me = ['Geodata','GLASS','MODIS','TPDC','W']

length = 5
styear = 2003
edyear = 2017

minx_minx = 2671   #最小的列数
miny_miny =  2101  #最小的行数

years = [x for x in range(styear,edyear+1)]

MuSyQ_datas,GLASS_datas,MODIS_datas,CASA_datas,W_datas,LAI_datas = [],[],[],[],[],[]
# MuSyQ_datas,GLASS_datas,CASA_datas,W_datas = [],[],[],[]
var = ['R2','RMSE','MSE','MAE']

'''预处理函数'''
def SetNodata(Datas,nodatakey):
    i=0
    datas_ = []
    for data in Datas:
        data_ = []
        for da in data:
            symbol = nodatakey[i][0]
            value = int(nodatakey[i][1:])
            if symbol == '>':
                da[da>=value] = np.nan
                da[da<0] = np.nan
            else:
                da[da<=value] = np.nan
                da[da<0] = np.nan
            data_.append(da)
        i+=1
        datas_.append(data_)
    return datas_

def R2_SetNodata(Datas):
    data_ = []
    for da in Datas:
        da[da<0] = np.nan
        data_.append(da)
    data_.append(data_)
    return data_

def SetDatatype(Datas): 
    datas_ = []
    for data in Datas:
        data_ = []
        for da in data:
            print(f'原始的数据类型为{da.dtype}')
            da.dtype = np.uint32
            print(f'数据类型更改为 {da.dtype}')
            data_.append(da)
        datas_.append(data_)
        
    return datas_

def normalization(Datas):

    datas_ = []
    for data in Datas:
        data_ = []
        for da in data:
            max_value = np.nanmax(da)
            min_value = np.nanmin(da)
            da = (da-min_value)/(max_value-min_value)
            data_.append(da)
        datas_.append(data_)
    return datas_

    
'''Write'''
def WriteArray(datalist,Name):
    sample_tif = r'F:\Integrated_analysis_data\Data\Sample\Reproject_Mul_2002.tif'                          # 需要读取的tif文件所在的文件夹的所在文件夹的路径    
    ds = gdal.Open(sample_tif)                             # 打开文件
    im_width = minx_minx                          # 获取栅格矩阵的列数
    logging.info(f'im_width: {im_width}')
    im_height = miny_miny                         # 获取栅格矩阵的行数
    logging.info(f'im_height: {im_height}')
    # im_bands = ds.RasterCount                     # 获取栅格矩阵的波段数
    #print(f'im_bands: {im_bands}')
    # band1 = ds.GetRasterBand(1)                         # 波段的indice起始为1，不为0
    img_datatype = gdal.GDT_Float32                       # 数据类型
    outdir = Outpath + os.sep + Name
    logging.info(f'-------输出文件夹为 {outdir}---------')
    if not os.path.exists(outdir):       #判断原始文件路劲是否存在,如果不存在就直接退出
        os.makedirs(outdir)
    for j in range(0,len(datalist)):    
        out_ds = gdal.GetDriverByName('GTiff').Create(
              outdir + os.sep + Name + "_" + str(var[j]) + '.tif',                   # tif文件所保存的路径
              im_width,                                     # 获取栅格矩阵的列数
              im_height,                                     # 获取栅格矩阵的行数
              ds.RasterCount,                                     # 获取栅格矩阵的波段数
              img_datatype)                                       # 获取第一波段的数据类型
        out_ds.SetProjection(ds.GetProjection())                # 投影信息
        out_ds.SetGeoTransform(ds.GetGeoTransform())            # 仿射信息
        out_band = out_ds.GetRasterBand(1)
        out_band.WriteArray(np.array(datalist[j]).reshape(miny_miny,minx_minx))    # 写入数据 (why)
        out_ds.FlushCache()  #(刷新缓存)
        del out_ds #删除 
        logging.info(f' {outdir + os.sep + Name + "_" + str(var[j]) + ".tif"} is  ok   !!!!!!!!')
    del ds    
def WriteArray_year(datalist,Name):
    sample_tif = r'F:\Integrated_analysis_data\Data\Sample\Reproject_Mul_2002.tif'                          # 需要读取的tif文件所在的文件夹的所在文件夹的路径    
    ds = gdal.Open(sample_tif)                             # 打开文件
    im_width = minx_minx                          # 获取栅格矩阵的列数
    logging.info(f'im_width: {im_width}')
    im_height = miny_miny                         # 获取栅格矩阵的行数
    logging.info(f'im_height: {im_height}')
    # im_bands = ds.RasterCount                     # 获取栅格矩阵的波段数
    #print(f'im_bands: {im_bands}')
    # band1 = ds.GetRasterBand(1)                         # 波段的indice起始为1，不为0
    img_datatype = gdal.GDT_Float32                       # 数据类型
    outdir = Outpath + os.sep + Name
    logging.info(f'-------输出文件夹为 {outdir}---------')
    if not os.path.exists(outdir):       #判断原始文件路劲是否存在,如果不存在就直接退出
        os.makedirs(outdir)
    for j in range(0,len(datalist)):    
        out_ds = gdal.GetDriverByName('GTiff').Create(
              outdir + os.sep + Name + "_" + str(years[j]) + '.tif',                   # tif文件所保存的路径
              im_width,                                     # 获取栅格矩阵的列数
              im_height,                                     # 获取栅格矩阵的行数
              ds.RasterCount,                                     # 获取栅格矩阵的波段数
              img_datatype)                                       # 获取第一波段的数据类型
        out_ds.SetProjection(ds.GetProjection())                # 投影信息
        out_ds.SetGeoTransform(ds.GetGeoTransform())            # 仿射信息
        out_band = out_ds.GetRasterBand(1)
        out_band.WriteArray(np.array(datalist[j]).reshape(miny_miny,minx_minx))    # 写入数据 (why)
        out_ds.FlushCache()  #(刷新缓存)
        del out_ds #删除 
        logging.info(f' {outdir + os.sep + Name + "_" + str(years[j]) + ".tif"} is  ok   !!!!!!!!')
    del ds 
def WriteArray_RR(images_pixels,type_,y_datas):
    images_list = np.array(images_pixels).T.tolist()
    y_list = np.array(y_datas).T.tolist()
    mean_results = []
    for images,y in tqdm(zip(images_list,y_list),desc = type_):
        result = RR_LinearRegression(images,y)
        mean_results.append(result)
    mean_results = np.array(mean_results)
    WriteArray([mean_results[:,0],mean_results[:,1],mean_results[:,2],mean_results[:,3]],type_)        
def WriteArray_R2(images_pixels,y_datas,type_):
    # logging.info(f'整理好的二维列表的行数为：{len(images_list)},列数为：{len(images_list[0])}')
    mean_results = []
    for images,y in tqdm(zip(np.array(images_pixels).T.tolist(),np.array(y_datas).T.tolist()),desc= type_):
        result = R2_LinearRegression(images,y)
        mean_results.append(result)
    WriteArray([pd.DataFrame(mean_results).iloc[:,0]],type_) 
# def RR_RF(images_pixels,Outpath,type_):
#     images_list = np.array(images_pixels).T.tolist()
#     mean_results = []
#     for images in tqdm(images_list):
#         result = RR_RandomForest(images)
#         mean_results.append(result)
#     mean_results = pd.DataFrame(mean_results)
#     datalist = [mean_results.iloc[:,0],mean_results.iloc[:,1],mean_results.iloc[:,2],mean_results.iloc[:,3]]
#     WriteArray(datalist,type_)

def RR_Multiply_Regression(mean_data,y_data):
    if np.isnan(np.array(mean_data)).any() or np.isnan(np.array(y_data)).any():
        return [np.nan,np.nan,np.nan,np.nan]
    model = LinearRegression()
    model.fit(np.array(mean_data),np.array(y_data).reshape(-1, 1))
    y_predict = model.predict(mean_data)
    r2 = r2_score(np.array(y_data).reshape(-1, 1), y_predict)
    mse =  mean_squared_error(np.array(y_data).reshape(-1, 1), y_predict)
    mae = mean_absolute_error(np.array(y_data).reshape(-1, 1), y_predict)
    rmse = sqrt(mse) 
    # print('r2:   ',r2)
    # print(na_me[np.argsort(model.coef_)])  #输出各个特征按照影响系数从小到大的顺序
    return [r2,rmse,mse,mae]

def R2_LinearRegression(mean_data,y_data):
    start = datetime.datetime.now()
    model = linear_model.LinearRegression()
    if np.isnan(np.array(mean_data)).any() or np.isnan(np.array(y_data)).any():
        return [np.nan]
    model.fit(np.array(mean_data).reshape(-1, 1), np.array(y_data).reshape(-1, 1))
    y_predict = model.predict(np.array(mean_data).reshape(-1, 1))
    r2 = r2_score(np.array(y_data).reshape(-1, 1), np.array(y_predict))
    end = datetime.datetime.now()
    sg.popup_notify(f'Mean Task done! Spend-time: {end-start}',display_duration_in_ms = 10000,fade_in_duration = 10000)
    return [r2]

def RR_LinearRegression(mean_data,y_data):
    start = datetime.datetime.now()
    model = linear_model.LinearRegression()
    if np.isnan(np.array(mean_data)).any() or np.isnan(np.array(y_data)).any():
        return [np.nan,np.nan,np.nan,np.nan]
    model.fit(np.array(mean_data).reshape(-1, 1), np.array(y_data).reshape(-1, 1))
    y_predict = model.predict(np.array(mean_data).reshape(-1, 1))
    r2 = r2_score(np.array(y_data).reshape(-1, 1), np.array(y_predict))
    mse =  mean_squared_error(np.array(y_data).reshape(-1, 1), y_predict)
    mae = mean_absolute_error(np.array(y_data).reshape(-1, 1), y_predict)
    rmse = sqrt(mse) 
    end = datetime.datetime.now()
    sg.popup_notify(f'Mean Task done! Spend-time: {end-start}',display_duration_in_ms = 10000,fade_in_duration = 10000)  
    return [r2,mse,mae,rmse]
  
def Cal_R2(Setnodata_datas):
    start = datetime.datetime.now()
    for name,da in tqdm(enumerate(Setnodata_datas[:-1]),desc = 'Cal_R2'):
        images_pixels1 = [] 
        images_pixels5 = []
        for year in tqdm(range(len(years)),desc = 'Year'):
            images_pixels3 = []
            images_pixels6 = []
            for y in range(miny_miny):
                for x in range(minx_minx):
                    images_pixels3.append(da[year][y][x])
                    images_pixels6.append(Setnodata_datas[-1][year][y][x])
            images_pixels1.append(images_pixels3)
            images_pixels5.append(images_pixels6)
        WriteArray_R2(images_pixels1,images_pixels5,'R2_' + na_me[name])
        sg.popup_notify(f'R2_{na_me[name]}  Task done!!!! Spend-time: {datetime.datetime.now()-start}')
    end = datetime.datetime.now()
    sg.popup_notify(f' Cal_R2     ALL  Task done!!!! Spend-time: {end-start}')
    
    
'''total'''
def Mean_Median(Setnodata_datas,nn_mean,nn_median):
    print('——————————————Mean,Median——————————————————')
    start = datetime.datetime.now()
    images_pixels1 = []  
    images_pixels2 = []
    images_pixels5 = []
    for year in tqdm(range(len(years)),desc = 'Mean,Median'):
        images_pixels3 = []
        images_pixels4 = []
        images_pixels6 = []
        for y in range(miny_miny):
            for x in range(minx_minx):
                a = np.array([Setnodata_datas[0][year][y][x] , Setnodata_datas[1][year][y][x], Setnodata_datas[2][year][y][x] , Setnodata_datas[3][year][y][x],Setnodata_datas[4][year][y][x]])
                if len(a[~np.isnan(a)])<length:
                    images_pixels3.append(np.nan)
                    images_pixels4.append(np.nan)
                else:
                    images_pixels3.append(np.nanmean(a))
                    images_pixels4.append(np.nanmedian(a))
                images_pixels6.append(Setnodata_datas[-1][year][y][x])
        images_pixels1.append(images_pixels3)
        images_pixels2.append(images_pixels4)
        images_pixels5.append(images_pixels6)
    WriteArray_RR(images_pixels1,nn_mean,images_pixels5)
    WriteArray_RR(images_pixels2,nn_median,images_pixels5)
    sg.popup_notify(f'Mean-Median Task done! Spend-time: {datetime.datetime.now()-start}')
    
def Mean_Median_Year(Setnodata_datas,nn_mean,nn_median):
    print('——————————————Mean,Median——————————————————')
    start = datetime.datetime.now()
    images_pixels1 = []  
    images_pixels2 = []
    for year in tqdm(range(len(years)),desc = 'Mean,Median Year'):
        images_pixels3 = []
        images_pixels4 = []
        for y in range(miny_miny):
            for x in range(minx_minx):
                a = np.array([Setnodata_datas[0][year][y][x] , Setnodata_datas[1][year][y][x], Setnodata_datas[2][year][y][x] , Setnodata_datas[3][year][y][x],Setnodata_datas[4][year][y][x]])
                if len(a[~np.isnan(a)])<length:
                    images_pixels3.append(np.nan)
                    images_pixels4.append(np.nan)
                else:
                    images_pixels3.append(np.nanmean(a))
                    images_pixels4.append(np.nanmedian(a))
        images_pixels1.append(images_pixels3)
        images_pixels2.append(images_pixels4)
    WriteArray_year(images_pixels1,nn_mean)
    WriteArray_year(images_pixels2,nn_median)
    sg.popup_notify(f'Mean-Median_Year Task done! Spend-time: {datetime.datetime.now()-start}')   

def Weight(Setnodata_datas,R2_SetNodata,nn):
    print('——————————————Weight——————————————————')
    start = datetime.datetime.now()
    images_pixels1 = []  #用于存放所有年，每年五种数据mean的值，一年一个列表
    images_pixels5 = []
    for year in tqdm(range(len(years)),desc = 'Weight'):
        images_pixels3 = []
        images_pixels6 = []
        for y in range(miny_miny):
            for x in range(minx_minx):
                a = np.array([Setnodata_datas[0][year][y][x] , Setnodata_datas[1][year][y][x], Setnodata_datas[2][year][y][x] , Setnodata_datas[3][year][y][x],Setnodata_datas[4][year][y][x]])
                if len(a[~np.isnan(a)])<length:
                    images_pixels3.append(np.nan)
                else:
                    images_pixels3.append((Setnodata_datas[0][year][y][x] * R2_SetNodata[0][y][x] + Setnodata_datas[1][year][y][x] * R2_SetNodata[1][y][x] + 
                                          Setnodata_datas[2][year][y][x] * R2_SetNodata[2][y][x] + Setnodata_datas[3][year][y][x] * R2_SetNodata[3][y][x] +
                                          Setnodata_datas[4][year][y][x] * R2_SetNodata[4][y][x])/(R2_SetNodata[0][y][x] + R2_SetNodata[1][y][x] + 
                                          R2_SetNodata[2][y][x] + R2_SetNodata[3][y][x] + R2_SetNodata[4][y][x]))
                images_pixels6.append(Setnodata_datas[-1][year][y][x])
        images_pixels1.append(images_pixels3)
        images_pixels5.append(images_pixels6)
    WriteArray_RR(images_pixels1,nn,images_pixels5)
    sg.popup_notify(f'Weight Task done! Spend-time: {datetime.datetime.now()-start}',display_duration_in_ms = 10000,fade_in_duration = 10000)

def Weight_Year(Setnodata_datas,R2_SetNodata,nn):
    print('——————————————Weight Year——————————————————')
    start = datetime.datetime.now()
    images_pixels1 = []  #用于存放所有年，每年五种数据mean的值，一年一个列表
    for year in tqdm(range(len(years)),desc = 'Weight Year'):
        images_pixels3 = []
        for y in range(miny_miny):
            for x in range(minx_minx):
                a = np.array([Setnodata_datas[0][year][y][x] , Setnodata_datas[1][year][y][x], Setnodata_datas[2][year][y][x] , Setnodata_datas[3][year][y][x],Setnodata_datas[4][year][y][x]])
                if len(a[~np.isnan(a)])<length:
                    images_pixels3.append(np.nan)
                else:
                    images_pixels3.append((Setnodata_datas[0][year][y][x] * R2_SetNodata[0][y][x] + Setnodata_datas[1][year][y][x] * R2_SetNodata[1][y][x] + 
                                          Setnodata_datas[2][year][y][x] * R2_SetNodata[2][y][x] + Setnodata_datas[3][year][y][x] * R2_SetNodata[3][y][x] +
                                          Setnodata_datas[4][year][y][x] * R2_SetNodata[4][y][x])/(R2_SetNodata[0][y][x] + R2_SetNodata[1][y][x] + 
                                          R2_SetNodata[2][y][x] + R2_SetNodata[3][y][x] + R2_SetNodata[4][y][x]))
        images_pixels1.append(images_pixels3)
    WriteArray_year(images_pixels1,nn)
    sg.popup_notify(f'Multiply_Regression Task done! Spend-time: {datetime.datetime.now()-start}',display_duration_in_ms = 10000,fade_in_duration = 10000)

def Multiply_Regression(Setnodata_datas,nn):
    print('——————————————Multiply_Regression——————————————————')
    start = datetime.datetime.now()
    images_pixels1 = []  #用于存放所有年，每年五种数据mean的值，一年一个列表
    images_pixels5 = []
    for y in tqdm(range(miny_miny),desc = 'Multiply_Regression'):
        for x in range(minx_minx):
            images_pixels3 = []
            images_pixels6 = []
            for year in range(len(years)):
                images_pixels3.append([Setnodata_datas[0][year][y][x],Setnodata_datas[1][year][y][x],Setnodata_datas[2][year][y][x],Setnodata_datas[3][year][y][x],Setnodata_datas[4][year][y][x]])
                images_pixels6.append([Setnodata_datas[-1][year][y][x]])
            images_pixels1.append(images_pixels3)
            images_pixels5.append(images_pixels6)
    mean_results = []
    for images,y in tqdm(zip(images_pixels1,images_pixels5)):
        mean_results.append(RR_Multiply_Regression(images,y))
    mean_results = np.array(mean_results)
    WriteArray([mean_results[:,0],mean_results[:,1],mean_results[:,2],mean_results[:,3]],nn)
    sg.popup_notify(f'Multiply_Regression Task done! Spend-time: {datetime.datetime.now()-start}',display_duration_in_ms = 10000,fade_in_duration = 10000)
 
def Multiply_Regression_Year(Setnodata_datas,nn):
    print('——————————————Multiply_Regression  Year——————————————————')
    start = datetime.datetime.now()
    images_pixels1 = []  #用于存放所有年，每年五种数据mean的值，一年一个列表
    for y in tqdm(range(miny_miny),desc = 'Multiply_Regression_Year'):
        for x in range(minx_minx):
            images_pixels3 = []
            for year in range(len(years)):
                images_pixels3.append([Setnodata_datas[0][year][y][x],Setnodata_datas[1][year][y][x],Setnodata_datas[2][year][y][x],Setnodata_datas[3][year][y][x],Setnodata_datas[4][year][y][x]])
            images_pixels1.append(images_pixels3)
    WriteArray_year(np.array(images_pixels1).T.tolist(),nn)
    sg.popup_notify(f'Multiply_Regression_Year Task done! Spend-time: {datetime.datetime.now()-start}',display_duration_in_ms = 10000,fade_in_duration = 10000)
    
if __name__ == "__main__": 
    
    print('-----------------Start----------------------')
    
    # minxsizes_list  = []
    # minysizes_list  = []
    
    # for year in range(styear,edyear+1):
        
    #     MuSyQ_dir = MuSyQ_inpath + os.sep + str(year)
    #     GLASS_dir = GLASS_inpath + os.sep + str(year)
    #     MODIS_dir = MODIS_path + os.sep + str(year)
    #     CASA_dir = CASA_path + os.sep + str(year)
    #     W_dir = W_path + os.sep + str(year)
    #     LAI_dir = LAI_path + os.sep + str(year)
        
    #     MuSyQ_ = gdal.Open(g(MuSyQ_dir + os.sep + MuSyQ_key)[0],gdal.GA_ReadOnly)
    #     GLASS_ = gdal.Open(g(GLASS_dir + os.sep + GLASS_key)[0],gdal.GA_ReadOnly)
    #     MODIS_ = gdal.Open(g(MODIS_dir + os.sep + MODIS_key)[0],gdal.GA_ReadOnly)
    #     CASA_ = gdal.Open(g(CASA_dir + os.sep + CASA_key)[0],gdal.GA_ReadOnly)
    #     W_ = gdal.Open(g(W_dir + os.sep + W_key)[0])
    #     LAI_ = gdal.Open(g(LAI_dir + os.sep + LAI_key)[0])
        
    #     # im_geotrans = dataset1.GetGeoTransform()
    #     # im_proj = dataset1.GetProjection()
    #     print(f'Geodata.RasterX,YSize---{year}: ',MuSyQ_.RasterXSize,MuSyQ_.RasterYSize)
    #     print(f'GLASS_.RasterX,YSize---{year}: ',GLASS_.RasterXSize,GLASS_.RasterYSize)
    #     print(f'MODIS_.RasterX,YSize---{year}: ',MODIS_.RasterXSize,MODIS_.RasterYSize)
    #     print(f'TPDC_.RasterX,YSize---{year}: ',CASA_.RasterXSize,CASA_.RasterYSize)
    #     print(f'W_.RasterX,YSize---{year}: ',W_.RasterXSize,W_.RasterYSize)
    #     print(f'W_.RasterX,YSize---{year}: ',LAI_.RasterXSize,LAI_.RasterYSize)
        
    #     minxsize = min([MuSyQ_.RasterXSize,GLASS_.RasterXSize,MODIS_.RasterXSize,CASA_.RasterXSize,W_.RasterXSize,LAI_.RasterXSize])
    #     # minxsize = min([MuSyQ_.RasterXSize,GLASS_.RasterXSize,CASA_.RasterXSize,W_.RasterXSize])

    #     minysize = min([MuSyQ_.RasterYSize,GLASS_.RasterYSize,MODIS_.RasterYSize,CASA_.RasterYSize,W_.RasterYSize,LAI_.RasterYSize])
    #     # minysize = min([MuSyQ_.RasterYSize,GLASS_.RasterYSize,CASA_.RasterYSize,W_.RasterYSize])
        
    #     minxsizes_list.append(minxsize)
    #     minysizes_list.append(minysize)
        
    #     del MuSyQ_ 
    #     del GLASS_ 
    #     del MODIS_ 
    #     del CASA_
    #     del W_
    #     del LAI_
        
    # minx_minx = min(minxsizes_list)
    # miny_miny = min(minysizes_list)
        
    print(f'minx_minx: {minx_minx}')
    print(f'miny_miny: {miny_miny}')
    
    for year in tqdm(range(styear,edyear+1),desc = 'Year'):
        MuSyQ_dir,GLASS_dir = MuSyQ_inpath + os.sep + str(year),GLASS_inpath + os.sep + str(year)
        MODIS_dir = MODIS_path + os.sep + str(year)
        CASA_dir = CASA_path + os.sep + str(year)
        W_dir = W_path + os.sep + str(year)
        LAI_dir = LAI_path + os.sep + str(year)

        MuSyQ_datas.append(gdal.Open(g(MuSyQ_dir + os.sep + MuSyQ_key)[0],gdal.GA_ReadOnly).ReadAsArray(0, 0, minx_minx, miny_miny))
        GLASS_datas.append(gdal.Open(g(GLASS_dir + os.sep + GLASS_key)[0],gdal.GA_ReadOnly).ReadAsArray(0, 0, minx_minx, miny_miny) )
        MODIS_datas.append(gdal.Open(g(MODIS_dir + os.sep + MODIS_key)[0],gdal.GA_ReadOnly).ReadAsArray(0, 0, minx_minx, miny_miny))
        CASA_datas.append(gdal.Open(g(CASA_dir + os.sep + CASA_key)[0],gdal.GA_ReadOnly).ReadAsArray(0, 0, minx_minx, miny_miny))
        W_datas.append(gdal.Open(g(W_dir + os.sep + W_key)[0]).ReadAsArray(0, 0, minx_minx, miny_miny))
        LAI_datas.append(gdal.Open(g(LAI_dir + os.sep + LAI_key)[0]).ReadAsArray(0, 0, minx_minx, miny_miny))

        
    MuSyQ_r2 = gdal.Open(MuSyQ_R2,gdal.GA_ReadOnly).ReadAsArray(0, 0, minx_minx, miny_miny)
    GLASS_r2 = gdal.Open(GLASS_R2,gdal.GA_ReadOnly).ReadAsArray(0, 0, minx_minx, miny_miny)
    MODIS_r2 = gdal.Open(MODIS_R2,gdal.GA_ReadOnly).ReadAsArray(0, 0, minx_minx, miny_miny)
    CASA_r2 = gdal.Open(CASA_R2,gdal.GA_ReadOnly).ReadAsArray(0, 0, minx_minx, miny_miny)
    W_r2 = gdal.Open(W_R2,gdal.GA_ReadOnly).ReadAsArray(0, 0, minx_minx, miny_miny)
     
    
    Mean_Median_Year(normalization(SetNodata([MuSyQ_datas,GLASS_datas,MODIS_datas,CASA_datas,W_datas,LAI_datas],nodatakey)),'Normal_Mean_Year','Normal_Median_Year')
    Mean_Median_Year(SetNodata([MuSyQ_datas,GLASS_datas,MODIS_datas,CASA_datas,W_datas,LAI_datas],nodatakey),'Mean_Year','Median_Year')
    Weight_Year(normalization(SetNodata([MuSyQ_datas,GLASS_datas,MODIS_datas,CASA_datas,W_datas,LAI_datas],nodatakey)),R2_SetNodata([MuSyQ_r2,GLASS_r2,MODIS_r2,CASA_r2,W_r2]),'Normal_Weight_Year')
    Weight_Year(SetNodata([MuSyQ_datas,GLASS_datas,MODIS_datas,CASA_datas,W_datas,LAI_datas],nodatakey),R2_SetNodata([MuSyQ_r2,GLASS_r2,MODIS_r2,CASA_r2,W_r2]),'Weight_Year')
    Multiply_Regression_Year(normalization(SetNodata([MuSyQ_datas,GLASS_datas,MODIS_datas,CASA_datas,W_datas,LAI_datas],nodatakey)),'Normal_Multiply_Regression_Year')
    Multiply_Regression_Year(SetNodata([MuSyQ_datas,GLASS_datas,MODIS_datas,CASA_datas,W_datas,LAI_datas],nodatakey),'Multiply_Regression_Year')
    
    # Multiply_Regression(normalization(SetNodata([MuSyQ_datas,GLASS_datas,MODIS_datas,CASA_datas,W_datas,LAI_datas],nodatakey)))
    # Mean_Median(normalization(SetNodata([MuSyQ_datas,GLASS_datas,MODIS_datas,CASA_datas,W_datas,LAI_datas],nodatakey)))
    # sg.popup_notify(title = 'Task done!',display_duration_in_ms = 10000,fade_in_duration = 10000)
    
    


    
    
    