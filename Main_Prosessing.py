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
from multiprocessing.dummy import Pool as ThreadPool
from pathos.multiprocessing import ProcessingPool as newPool

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [*] %(processName)s %(message)s"
)



Outpath = r'K:\HeQiFan\Out'   #输出路径

MuSyQ_inpath = r'K:\HeQiFan\1Y\Geodata_2000_2017_1y'   
GLASS_inpath = r'K:\HeQiFan\1Y\GLASS_2000_2017_1y'
MODIS_path = r'K:\HeQiFan\1Y\MODIS_2000_2017_1y'
CASA_path = r'K:\HeQiFan\1Y\TPDC_2000_2017_1y'
W_path = r'K:\HeQiFan\1Y\W_2000_2017_1y'
LAI_path = r'K:\HeQiFan\1Y\LAI_2003_2017_1y'

MuSyQ_R2 = r'K:\HeQiFan\Out\R2_Geodata\R2_Geodata_.tif'    #每种模型的R2，weight中要用
GLASS_R2 = r'K:\HeQiFan\Out\R2_GLASS\R2_GLASS_.tif'
MODIS_R2 = r'K:\HeQiFan\Out\R2_MODIS\R2_MODIS_.tif'
CASA_R2  = r'K:\HeQiFan\Out\R2_TPDC\R2_TPDC_.tif'
W_R2     = r'K:\HeQiFan\Out\R2_W\R2_W_.tif'

MuSyQ_key,GLASS_key,MODIS_key,CASA_key,W_key,LAI_key =  'Mask_*.tif','Mask_*.tif','Mask_*.tif','Mask_*.tif','Resample_*.tif','Mask_*.tif'

nodatakey = [['<-1000'],['<-1000'],['<-1000'],['<-1000','>1700'],['<-1000','>3200'],['<-1000']]  #每种模型的无效值

na_me = ['Geodata','GLASS','MODIS','TPDC','W']   #每种模型的名称
na_me2 = ['Geodata','GLASS','MODIS','TPDC','W','LAI'] 

Pools = 8
length = 5      #模型的数量
styear = 2003   #开始年份
edyear = 2017   #结束年份

minx_minx = 2671   #列数
miny_miny =  2101  #行数

years = [x for x in range(styear,edyear+1)]  #年份的列表

MuSyQ_datas,GLASS_datas,MODIS_datas,CASA_datas,W_datas,LAI_datas = [],[],[],[],[],[]  #定义空的列表，存放每年的数据
var = ['R2','RMSE','MSE','MAE']   

'''预处理函数'''
def SetNodata(Datas,nodatakey):
    '''
    设置无效值
    '''
    datas_ = []
    for data,key in zip(Datas,nodatakey):
        data_ = []
        for da in data:
            for k in key:
                symbol = k[0]  #获取符号
                value = int(k[1:])  #获取数组
                if symbol == '>':
                    da[da>=value] = np.nan
                    da[da<0] = np.nan
                else:
                    da[da<=value] = np.nan
                    da[da<0] = np.nan
                data_.append(da)
        datas_.append(data_)
    return datas_

def R2_SetNodata(Datas):
    '''
    设置无效值
    '''
    data_ = []
    for da in Datas:
        da[da<0] = np.nan
        data_.append(da)
    data_.append(data_)
    return data_

def SetDatatype(Datas): 
    '''
    设置数据类型
    '''
    datas_ = []
    for data in Datas:
        data_ = []
        for da in data:
            # print(f'原始的数据类型为{da.dtype}')
            da.dtype = np.uint32
            # print(f'数据类型更改为 {da.dtype}')
            data_.append(da)
        datas_.append(data_)
    return datas_

def normalization(Datas):
    '''
    归一化
    '''
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
def A_WriteArray(datalist,Name,var_list):
    '''
    写出数据
    '''
    sample_tif = r'K:\HeQiFan\Sample\Mask_Mul_2009.tif'                            # 需要读取的tif文件所在的文件夹的所在文件夹的路径
    ds = gdal.Open(sample_tif)                             # 打开文件
    im_width = minx_minx                          # 获取栅格矩阵的列数
    # logging.info(f'im_width: {im_width}')
    im_height = miny_miny                         # 获取栅格矩阵的行数
    # logging.info(f'im_height: {im_height}')
    # im_bands = ds.RasterCount                     # 获取栅格矩阵的波段数
    #print(f'im_bands: {im_bands}')
    # band1 = ds.GetRasterBand(1)                         # 波段的indice起始为1，不为0
    img_datatype = gdal.GDT_Float32                    # 数据类型
    outdir = Outpath + os.sep + Name
    logging.info(f'-------输出文件夹为 {outdir}---------')
    if not os.path.exists(outdir):       #判断原始文件路劲是否存在,如果不存在就直接退出
        os.makedirs(outdir)
    for j in range(0,len(datalist)):    
        out_ds = gdal.GetDriverByName('GTiff').Create(
              outdir + os.sep + Name + "_" + str(var_list[j]) + '.tif',                   # tif文件所保存的路径
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
        logging.info(f' {outdir + os.sep + Name + "_" + str(var_list[j]) + ".tif"} is  ok   !!!!!!!!')
    del ds    

def M_R_P(mean_data,y_data,r_name):
    '''Get Multiply_Regression_RR or Get Multiply_Regression Predicted Data'''
    mean_data = np.array(mean_data)
    y_data = np.array(y_data)
    if np.isnan(mean_data).any() or np.isnan(y_data).any():
        if r_name == 'RR':
            return [np.nan,np.nan,np.nan,np.nan]
        elif r_name == 'Predicted':
            return [np.nan]*len(years)
    else:
        model = LinearRegression()
        model.fit(mean_data,y_data.reshape(-1, 1))
        y_predict = model.predict(mean_data)
        r2 = r2_score(y_data.reshape(-1, 1), y_predict.reshape(-1, 1))
        mse =  mean_squared_error(y_data.reshape(-1, 1), y_predict.reshape(-1, 1))
        mae = mean_absolute_error(y_data.reshape(-1, 1), y_predict.reshape(-1, 1))
        rmse = sqrt(mse) 
        y_predict_data = model.predict(mean_data).flatten().tolist()
        if r_name == 'RR':
            return [r2,rmse,mse,mae]
        elif r_name == 'Predicted':
            return y_predict_data
        
def L_R(mean_data,y_data,r_name):
    '''Get liner_Regression_R2 or get liner_Regression_RR'''
    mean_data = np.array(mean_data)
    y_data = np.array(y_data)
    if np.isnan(mean_data).any() or np.isnan(y_data).any():
        if r_name == 'R2':
            return [np.nan]
        elif r_name == 'RR':
            return [np.nan,np.nan,np.nan,np.nan]
    else:
        model = linear_model.LinearRegression()
        model.fit(mean_data.reshape(-1, 1), y_data.reshape(-1, 1))
        y_predict = model.predict(mean_data.reshape(-1, 1))
        r2 = r2_score(y_data.reshape(-1, 1), np.array(y_predict))
        mse =  mean_squared_error(y_data.reshape(-1, 1), y_predict.reshape(-1, 1))
        mae = mean_absolute_error(y_data.reshape(-1, 1), y_predict.reshape(-1, 1))
        rmse = sqrt(mse) 
        if r_name == 'R2':
            return [r2]
        elif r_name == 'RR':
            return [r2,mse,mae,rmse]
 
def Cal_R2(Setnodata_datas):
    start = datetime.datetime.now()
    for name,da in tqdm(enumerate(Setnodata_datas[:-1]),desc = 'Cal_R2'):
        images_pixels1 = [] 
        images_pixels5 = []
        for year in range(len(years)):
            images_pixels1.append(da[year].flatten().tolist())
            images_pixels5.append(Setnodata_datas[-1][year].flatten().tolist())
        # mean_results = []
        # for images,y in tqdm(zip(np.array(images_pixels1).T.tolist(),np.array(images_pixels5).T.tolist()),desc= 'R2_' + na_me[name]):
        #     result = L_R(images,y,'R2')
        #     mean_results.append(result)
        name_list = ['R2']*len(np.array(images_pixels1).T.tolist())
        print('————————————————————————————————')
        print('———————————R2 Pool Start—————————————————————')
        try:
            mean_results = pool.map(L_R,np.array(images_pixels1).T.tolist(),np.array(images_pixels5).T.tolist(),name_list)
            # mean_results = list((tqdm(pool.imap(L_R,np.array(images_pixels1).T.tolist(),np.array(images_pixels5).T.tolist(),name_list),desc='Cal_R2 Pool')))
            pool.close()
            pool.join()
        except:
            pool.restart()
            mean_results = pool.map(L_R,np.array(images_pixels1).T.tolist(),np.array(images_pixels5).T.tolist(),name_list)
            # mean_results = list((tqdm(pool.imap(L_R,np.array(images_pixels1).T.tolist(),np.array(images_pixels5).T.tolist(),name_list),desc='Cal_R2 Pool')))
            pool.close()
            pool.join()
        print('————————————————————————————————')
        print('———————————R2 Pool End—————————————————————')
        A_WriteArray([np.array(mean_results)[:,0]],'R2_' + na_me[name],['']) 
    end = datetime.datetime.now()
    sg.popup_notify(f' Cal_R2     ALL  Task done!!!! Spend-time: {end-start}')
      
'''total'''
def Mean_Median_RR(Setnodata_datas,nn_mean,nn_median):
    print('——————————————Mean,Median——————————————————')
    start = datetime.datetime.now()
    # images_pixels1 = []  
    # images_pixels2 = []
    # images_pixels5 = []
    # for year in tqdm(range(len(years)),desc = 'Mean,Median'):
    #     images_pixels3 = []
    #     images_pixels4 = []
    #     images_pixels6 = []
    #     for y in range(miny_miny):
    #         for x in range(minx_minx):
    #             a = np.array([Setnodata_datas[0][year][y][x] , Setnodata_datas[1][year][y][x], Setnodata_datas[2][year][y][x] , Setnodata_datas[3][year][y][x],Setnodata_datas[4][year][y][x]])
    #             if len(a[~np.isnan(a)])<length:
    #                 images_pixels3.append(np.nan)
    #                 images_pixels4.append(np.nan)
    #             else:
    #                 images_pixels3.append(np.nanmean(a))
    #                 images_pixels4.append(np.nanmedian(a))
    #             images_pixels6.append(Setnodata_datas[-1][year][y][x])
    #     images_pixels1.append(images_pixels3)
    #     images_pixels2.append(images_pixels4)
    #     images_pixels5.append(images_pixels6)
    images_pixels1 = []  
    images_pixels2 = []
    images_pixels5 = []
    for year in tqdm(range(len(years)),desc = 'Mean,Median'):
        a = np.array([Setnodata_datas[0][year] , Setnodata_datas[1][year], Setnodata_datas[2][year] , Setnodata_datas[3][year],Setnodata_datas[4][year]])
        images_pixels1.append(np.nanmean(a,axis=0).flatten().tolist())
        images_pixels2.append(np.nanmedian(a,axis=0).flatten().tolist())
        images_pixels5.append(Setnodata_datas[-1][year].flatten().tolist())
    # # mean_results = []
    # # for images,y in tqdm(zip(np.array(images_pixels1).T.tolist(),np.array(images_pixels5).T.tolist()),desc = nn_mean):
    # #     result = L_R(images,y,'RR')
    # #     mean_results.append(result)
    # print('————————————————————————————————')
    # print('———————————Mean Pool Start—————————————————————')
    # name_list = ['RR']*len(np.array(images_pixels1).T.tolist())
    # try:
    #     mean_results = pool.map(L_R,np.array(images_pixels1).T.tolist(),np.array(images_pixels5).T.tolist(),name_list)
    #     # mean_results = list((tqdm(pool.imap(L_R,np.array(images_pixels1).T.tolist(),np.array(images_pixels5).T.tolist(),name_list),desc='Mean Pool')))
    #     pool.close()
    #     pool.join()
    # except:
    #     pool.restart()
    #     mean_results = pool.map(L_R,np.array(images_pixels1).T.tolist(),np.array(images_pixels5).T.tolist(),name_list)
    #     # mean_results = list((tqdm(pool.imap(L_R,np.array(images_pixels1).T.tolist(),np.array(images_pixels5).T.tolist(),name_list),desc='Mean Pool')))
    #     pool.close()
    #     pool.join()
    # print('———————————Mean Pool End—————————————————————')
    # print('————————————————————————————————')
    # mean_results = np.array(mean_results)
    # A_WriteArray([mean_results[:,0],mean_results[:,1],mean_results[:,2],mean_results[:,3]],nn_mean,var)
    # mean_results = []
    # for images,y in tqdm(zip(np.array(images_pixels2).T.tolist(),np.array(images_pixels5).T.tolist()),desc = nn_mean):
    #     result = L_R(images,y,'RR')
    #     mean_results.append(result)
    print('————————————————————————————————')
    print('———————————Median Pool Start—————————————————————')
    name_list = ['RR']*len(np.array(images_pixels2).T.tolist())
    try:
        mean_results = pool.map(L_R,np.array(images_pixels2).T.tolist(),np.array(images_pixels5).T.tolist(),name_list)
        # mean_results = list((tqdm(pool.imap(L_R,np.array(images_pixels2).T.tolist(),np.array(images_pixels5).T.tolist(),name_list),desc='Median Pool')))
        pool.close()
        pool.join()
    except:
        pool.restart()
        mean_results = pool.map(L_R,np.array(images_pixels2).T.tolist(),np.array(images_pixels5).T.tolist(),name_list)
        # mean_results = list((tqdm(pool.imap(L_R,np.array(images_pixels2).T.tolist(),np.array(images_pixels5).T.tolist(),name_list),desc='Median Pool')))
        pool.close()
        pool.join()
    print('———————————Median Pool End—————————————————————')
    print('————————————————————————————————')
    mean_results = np.array(mean_results)
    A_WriteArray([mean_results[:,0],mean_results[:,1],mean_results[:,2],mean_results[:,3]],nn_median,var)      
    sg.popup_notify(f'Mean-Median Task done! Spend-time: {datetime.datetime.now()-start}')
    
def Mean_Median_Year(Setnodata_datas,nn_mean,nn_median):
    print('——————————————Mean,Median——————————————————')
    start = datetime.datetime.now()
    # images_pixels1 = []  
    # images_pixels2 = []
    # for year in tqdm(range(len(years)),desc = 'Mean,Median Year'):
    #     images_pixels3 = []
    #     images_pixels4 = []
    #     for y in range(miny_miny):
    #         for x in range(minx_minx):
    #             a = np.array([Setnodata_datas[0][year][y][x] , Setnodata_datas[1][year][y][x], Setnodata_datas[2][year][y][x] , Setnodata_datas[3][year][y][x],Setnodata_datas[4][year][y][x]])
    #             if len(a[~np.isnan(a)])<length:
    #                 images_pixels3.append(np.nan)
    #                 images_pixels4.append(np.nan)
    #             else:
    #                 images_pixels3.append(np.nanmean(a))
    #                 images_pixels4.append(np.nanmedian(a))
    #     images_pixels1.append(images_pixels3)
    #     images_pixels2.append(images_pixels4)
    images_pixels1 = []  
    images_pixels2 = []
    for year in tqdm(range(len(years)),desc = 'Mean,Median'):
        a = np.array([Setnodata_datas[0][year] , Setnodata_datas[1][year], Setnodata_datas[2][year] , Setnodata_datas[3][year],Setnodata_datas[4][year]])
        images_pixels1.append(np.nanmean(a,axis=0))
        images_pixels2.append(np.nanmedian(a,axis=0))
    A_WriteArray(images_pixels1,nn_mean,years)
    A_WriteArray(images_pixels2,nn_median,years)
    sg.popup_notify(f'Mean-Median_Year Task done! Spend-time: {datetime.datetime.now()-start}')   

def Weight_RR(Setnodata_datas,R2_SetNodata,nn):
    print('——————————————Weight——————————————————')
    start = datetime.datetime.now()
    images_pixels1 = []  #用于存放所有年，每年五种数据mean的值，一年一个列表
    images_pixels5 = []
    for year in tqdm(range(len(years)),desc = 'Weight'):
        a = np.array([(Setnodata_datas[0][year]* R2_SetNodata[0]+
                      Setnodata_datas[1][year]* R2_SetNodata[1]+
                      Setnodata_datas[2][year]* R2_SetNodata[2]+ 
                      Setnodata_datas[3][year]* R2_SetNodata[3]+
                      Setnodata_datas[4][year]* R2_SetNodata[4])/
                      (R2_SetNodata[0] + R2_SetNodata[1] + 
                      R2_SetNodata[2] + R2_SetNodata[3] + R2_SetNodata[4])
                      ])
        images_pixels1.append(a.flatten().tolist())
        images_pixels5.append(Setnodata_datas[-1][year].flatten().tolist())
    # mean_results = []
    # for images,y in tqdm(zip(np.array(images_pixels1).T.tolist(),np.array(images_pixels5).T.tolist()),desc = nn):
    #     result = L_R(images,y,'R2')
    #     mean_results.append(result)
    print('————————————————————————————————')
    print('———————————Weight Pool Start—————————————————————')
    name_list = ['RR']*len(np.array(images_pixels1).T.tolist())
    
    try:
        mean_results = pool.map(L_R,np.array(images_pixels1).T.tolist(),np.array(images_pixels5).T.tolist(),name_list)
        # mean_results = list((tqdm(pool.imap(L_R,np.array(images_pixels1).T.tolist(),np.array(images_pixels5).T.tolist(),name_list),desc='Weight Pool')))
        pool.close()
        pool.join()
    except:
        pool.restart()
        mean_results = pool.map(L_R,np.array(images_pixels1).T.tolist(),np.array(images_pixels5).T.tolist(),name_list)
        # mean_results = list((tqdm(pool.imap(L_R,np.array(images_pixels1).T.tolist(),np.array(images_pixels5).T.tolist(),name_list),desc='Weight Pool')))
        pool.close()
        pool.join()
    print('————————————————————————————————')
    print('———————————Weight Pool End—————————————————————')
    mean_results = np.array(mean_results)
    A_WriteArray([mean_results[:,0],mean_results[:,1],mean_results[:,2],mean_results[:,3]],nn,var)      
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
    A_WriteArray(images_pixels1,nn,years)
    sg.popup_notify(f'Weight_Year Task done! Spend-time: {datetime.datetime.now()-start}',display_duration_in_ms = 10000,fade_in_duration = 10000)

def Multiply_Regression_RR(Setnodata_datas,nn):
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
                images_pixels6.append(Setnodata_datas[-1][year][y][x])
            images_pixels1.append(images_pixels3)
            images_pixels5.append(images_pixels6)
    # mean_results = []
    # for images,y in tqdm(zip(images_pixels1,images_pixels5)):
    #     mean_results.append(M_R_P(images,y,'RR'))
    print('————————————————————————————————')
    print('———————————Multiply_Regression_RR Pool Start—————————————————————')
    name_list = ['RR']*len(images_pixels1)
    try:
        mean_results = pool.map(M_R_P,images_pixels1,images_pixels5,name_list)
        # mean_results = list((tqdm(pool.imap(L_R,images_pixels1,images_pixels5,name_list),desc='Multiply_Regression_RR Pool')))
        pool.close()
        pool.join()
    except:
        pool.restart()
        mean_results = pool.map(M_R_P,images_pixels1,images_pixels5,name_list)
        # mean_results = list((tqdm(pool.imap(L_R,images_pixels1,images_pixels5,name_list),desc='Multiply_Regression_RR Pool')))
        pool.close()
        pool.join()
    print('————————————————————————————————')
    print('———————————Multiply_Regression_RR Pool End—————————————————————')
    mean_results = np.array(mean_results)
    A_WriteArray([mean_results[:,0],mean_results[:,1],mean_results[:,2],mean_results[:,3]],nn,var)
    sg.popup_notify(f'Multiply_Regression Task done! Spend-time: {datetime.datetime.now()-start}',display_duration_in_ms = 10000,fade_in_duration = 10000)
 
def Multiply_Regression_Year(Setnodata_datas,nn):
    print('——————————————Multiply_Regression  Year——————————————————')
    start = datetime.datetime.now()
    images_pixels1 = []  #用于存放所有年，每年五种数据mean的值，一年一个列表
    images_pixels5 = []
    for y in tqdm(range(miny_miny),desc = 'Multiply_Regression_Year'):
        for x in range(minx_minx):
            images_pixels3 = []
            images_pixels6 = []
            for year in range(len(years)):
                images_pixels3.append([Setnodata_datas[0][year][y][x],Setnodata_datas[1][year][y][x],Setnodata_datas[2][year][y][x],Setnodata_datas[3][year][y][x],Setnodata_datas[4][year][y][x]])
                images_pixels6.append(Setnodata_datas[-1][year][y][x])
            images_pixels1.append(images_pixels3)
            images_pixels5.append(images_pixels6)
    # mean_results = []
    # for images,y in tqdm(zip(images_pixels1,images_pixels5)):
    #     mean_results.append(M_R_P(images,y,'Predicted'))
    print('————————————————————————————————')
    print('———————————Multiply_Regression_Year Pool Start—————————————————————')
    name_list = ['Predicted']*len(images_pixels1)

    try:
        mean_results = pool.map(M_R_P,images_pixels1,images_pixels5,name_list)
        # mean_results = list((tqdm(pool.imap(L_R,images_pixels1,images_pixels5,name_list),desc='Multiply_Regression_Year Pool')))
        pool.close()
        pool.join()
    except:
        pool.restart()
        mean_results = pool.map(M_R_P,images_pixels1,images_pixels5,name_list)
        # mean_results = list((tqdm(pool.imap(L_R,images_pixels1,images_pixels5,name_list),desc='Multiply_Regression_Year Pool')))
        pool.close()
        pool.join()
    print('————————————————————————————————')
    print('———————————Multiply_Regression_Year Pool End—————————————————————')
    A_WriteArray(np.array(mean_results).T.tolist(),nn,years)
    sg.popup_notify(f'Multiply_Regression_Year Task done! Spend-time: {datetime.datetime.now()-start}',display_duration_in_ms = 10000,fade_in_duration = 10000)
       
def normalization_Writearray_Spatial(Datas):
    '''
    归一化（空间）
    '''
    sample_tif = r'K:\HeQiFan\Sample\Mask_Mul_2009.tif'       # 需要读取的tif文件所在的文件夹的所在文件夹的路径
    ds = gdal.Open(sample_tif)                             # 打开文件
    for data,na in zip(Datas,na_me2):
        for da,year in zip(data,years):
            max_value = np.nanmax(da)
            min_value = np.nanmin(da)
            da = (da-min_value)/(max_value-min_value)
            im_width = minx_minx                          # 获取栅格矩阵的列数
            # logging.info(f'im_width: {im_width}')
            im_height = miny_miny                         # 获取栅格矩阵的行数
            # logging.info(f'im_height: {im_height}')
            # im_bands = ds.RasterCount                     # 获取栅格矩阵的波段数
            #print(f'im_bands: {im_bands}')
            band1 = ds.GetRasterBand(1)                         # 波段的indice起始为1，不为0
            img_datatype = gdal.GDT_Float32                       # 数据类型
            outdir = Outpath + os.sep + 'Normal_Spatial_' + na
            logging.info(f'-------输出文件夹为 {outdir}---------')
            if not os.path.exists(outdir):       #判断原始文件路劲是否存在,如果不存在就直接退出
                os.makedirs(outdir)
            # for j in range(0,len(datalist)):    
            out_ds = gdal.GetDriverByName('GTiff').Create(
                  outdir + os.sep + 'Normal_Spatial_' + na + "_" + str(year) + '.tif',                   # tif文件所保存的路径
                  im_width,                                         # 获取栅格矩阵的列数
                  im_height,                                        # 获取栅格矩阵的行数
                  ds.RasterCount,                                   # 获取栅格矩阵的波段数
                  img_datatype)                                     # 获取第一波段的数据类型
            out_ds.SetProjection(ds.GetProjection())                # 投影信息
            out_ds.SetGeoTransform(ds.GetGeoTransform())            # 仿射信息
            out_band = out_ds.GetRasterBand(1)
            out_band.WriteArray(da.reshape(miny_miny,minx_minx))    # 写入数据 (why)
            out_ds.FlushCache()  #(刷新缓存)
            del out_ds #删除 
            logging.info(f' {outdir + os.sep + "Normal_Spatial_" + na + "_" + str(year) + ".tif"} is  ok   !!!!!!!!')
    del ds 
    
    
def normalization_Writearray_Spatial_time(Datas):
    '''
    归一化（空间和时间）
    '''
    sample_tif = r'K:\HeQiFan\Sample\Mask_Mul_2009.tif'                          # 需要读取的tif文件所在的文件夹的所在文件夹的路径
    ds = gdal.Open(sample_tif)                             # 打开文件
    MuSyQ_min_list,GLASS_min_list,MODIS_min_list,CASA_min_list,W_min_list,LAI_min_list = [],[],[],[],[],[]
    MuSyQ_max_list,GLASS_max_list,MODIS_max_list,CASA_max_list,W_max_list,LAI_max_list = [],[],[],[],[],[]
    for year in range(len(years)):
        MuSyQ_min_list.append(np.nanmin(Datas[0][year]))
        GLASS_min_list.append(np.nanmin(Datas[1][year]))
        MODIS_min_list.append(np.nanmin(Datas[2][year]))
        CASA_min_list.append(np.nanmin(Datas[3][year]))
        W_min_list.append(np.nanmin(Datas[4][year]))
        LAI_min_list.append(np.nanmin(Datas[5][year]))
        
        MuSyQ_max_list.append(np.nanmax(Datas[0][year]))
        GLASS_max_list.append(np.nanmax(Datas[1][year]))
        MODIS_max_list.append(np.nanmax(Datas[2][year]))
        CASA_max_list.append(np.nanmax(Datas[3][year]))
        W_max_list.append(np.nanmax(Datas[4][year]))
        LAI_max_list.append(np.nanmax(Datas[5][year]))
    MuSyQ_min,GLASS_min,MODIS_min,CASA_min,W_min,LAI_min = min(MuSyQ_min_list),min(GLASS_min_list),min(MODIS_min_list),min(CASA_min_list),min(W_min_list),min(LAI_min_list)
    MuSyQ_max,GLASS_max,MODIS_max,CASA_max,W_max,LAI_max = max(MuSyQ_max_list),max(GLASS_max_list),max(MODIS_max_list),max(CASA_max_list),max(W_max_list),max(LAI_max_list)
    print('min:',MuSyQ_min,GLASS_min,MODIS_min,CASA_min,W_min,LAI_min)
    print('max:',MuSyQ_max,GLASS_max,MODIS_max,CASA_max,W_max,LAI_max)
    min_max = {'Geodata': {'min':MuSyQ_min,'max':MuSyQ_max},'GLASS':{'min':GLASS_min,'max':GLASS_max},'MODIS':{'min':MODIS_min,'max':MODIS_max},'TPDC':{'min':CASA_min,'max':CASA_max},'W':{'min':W_min,'max':W_max},'LAI':{'min':LAI_min,'max':LAI_max}}

    for data,na in zip(Datas,na_me2):
        for da,year in zip(data,years):  
            max_value = np.nanmax(min_max[na]['max'])
            min_value = np.nanmin(min_max[na]['min'])
            mean_value = np.nanmean(da)
            da = (da-mean_value)/(max_value-min_value)
            im_width = minx_minx                          # 获取栅格矩阵的列数
            # logging.info(f'im_width: {im_width}')
            im_height = miny_miny                         # 获取栅格矩阵的行数
            # logging.info(f'im_height: {im_height}')
            # im_bands = ds.RasterCount                     # 获取栅格矩阵的波段数
            #print(f'im_bands: {im_bands}')
            # band1 = ds.GetRasterBand(1)                         # 波段的indice起始为1，不为0
            img_datatype = gdal.GDT_Float32                       # 数据类型
            outdir = Outpath + os.sep + 'Normal_' + na
            logging.info(f'-------输出文件夹为 {outdir}---------')
            if not os.path.exists(outdir):       #判断原始文件路劲是否存在,如果不存在就直接退出
                os.makedirs(outdir)
            # for j in range(0,len(datalist)):    
            out_ds = gdal.GetDriverByName('GTiff').Create(
                  outdir + os.sep + 'Normal_' + na + "_" + str(year) + '.tif',                   # tif文件所保存的路径
                  im_width,                                         # 获取栅格矩阵的列数
                  im_height,                                        # 获取栅格矩阵的行数
                  ds.RasterCount,                                   # 获取栅格矩阵的波段数
                  img_datatype)                                     # 获取第一波段的数据类型
            out_ds.SetProjection(ds.GetProjection())                # 投影信息
            out_ds.SetGeoTransform(ds.GetGeoTransform())            # 仿射信息
            out_band = out_ds.GetRasterBand(1)
            out_band.WriteArray(da.reshape(miny_miny,minx_minx))    # 写入数据 (why)
            out_ds.FlushCache()  #(刷新缓存)
            del out_ds #删除 
            logging.info(f' {outdir + os.sep + "Normal_" + na + "_" + str(year) + ".tif"} is  ok   !!!!!!!!')
    del ds 
    
if __name__ == "__main__": 
    # __spec__ = "ModuleSpec(name='builtins', loader=<class '_frozen_importlib.BuiltinImporter'>)"
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
        GLASS_datas.append(gdal.Open(g(GLASS_dir + os.sep + GLASS_key)[0],gdal.GA_ReadOnly).ReadAsArray(0, 0, minx_minx, miny_miny))
        MODIS_datas.append(gdal.Open(g(MODIS_dir + os.sep + MODIS_key)[0],gdal.GA_ReadOnly).ReadAsArray(0, 0, minx_minx, miny_miny))
        CASA_datas.append(gdal.Open(g(CASA_dir + os.sep + CASA_key)[0],gdal.GA_ReadOnly).ReadAsArray(0, 0, minx_minx, miny_miny))
        W_datas.append(gdal.Open(g(W_dir + os.sep + W_key)[0],gdal.GA_ReadOnly).ReadAsArray(0, 0, minx_minx, miny_miny))
        LAI_datas.append(gdal.Open(g(LAI_dir + os.sep + LAI_key)[0],gdal.GA_ReadOnly).ReadAsArray(0, 0, minx_minx, miny_miny))
        
    

     
    pool = newPool(Pools)
    '''先计算平均值，中值的RR'''
    # Mean_Median_RR(normalization(SetNodata([MuSyQ_datas,GLASS_datas,MODIS_datas,CASA_datas,W_datas,LAI_datas],nodatakey)),'Liner_Mean','Liner_Median')
    '''然后计算每个模型的R2'''
    # Cal_R2(normalization(SetNodata([MuSyQ_datas,GLASS_datas,MODIS_datas,CASA_datas,W_datas,LAI_datas],nodatakey)))
    
    # MuSyQ_r2 = gdal.Open(MuSyQ_R2,gdal.GA_ReadOnly).ReadAsArray(0, 0, minx_minx, miny_miny)
    # GLASS_r2 = gdal.Open(GLASS_R2,gdal.GA_ReadOnly).ReadAsArray(0, 0, minx_minx, miny_miny)
    # MODIS_r2 = gdal.Open(MODIS_R2,gdal.GA_ReadOnly).ReadAsArray(0, 0, minx_minx, miny_miny)
    # CASA_r2 = gdal.Open(CASA_R2,gdal.GA_ReadOnly).ReadAsArray(0, 0, minx_minx, miny_miny)
    # W_r2 = gdal.Open(W_R2,gdal.GA_ReadOnly).ReadAsArray(0, 0, minx_minx, miny_miny)

    '''再根据每个模型的R2计算权重的RR'''
    # Weight_RR(normalization(SetNodata([MuSyQ_datas,GLASS_datas,MODIS_datas,CASA_datas,W_datas,LAI_datas],nodatakey)),R2_SetNodata([MuSyQ_r2,GLASS_r2,MODIS_r2,CASA_r2,W_r2]),'Liner_Weight')
    # '''再计算多元回归的RR'''
    # Multiply_Regression_RR(normalization(SetNodata([MuSyQ_datas,GLASS_datas,MODIS_datas,CASA_datas,W_datas,LAI_datas],nodatakey)),'Liner_Mul')
    
    '''再计算每种方法的每年的值（归一化和没有归一化的）'''
    # Mean_Median_Year(normalization(SetNodata([MuSyQ_datas,GLASS_datas,MODIS_datas,CASA_datas,W_datas,LAI_datas],nodatakey)),'Normal_Mean_Year','Normal_Median_Year')
    # Mean_Median_Year(SetNodata([MuSyQ_datas,GLASS_datas,MODIS_datas,CASA_datas,W_datas,LAI_datas],nodatakey),'Mean_Year','Median_Year')
    # Weight_Year(normalization(SetNodata([MuSyQ_datas,GLASS_datas,MODIS_datas,CASA_datas,W_datas,LAI_datas],nodatakey)),R2_SetNodata([MuSyQ_r2,GLASS_r2,MODIS_r2,CASA_r2,W_r2]),'Normal_Weight_Year')
    # Weight_Year(SetNodata([MuSyQ_datas,GLASS_datas,MODIS_datas,CASA_datas,W_datas,LAI_datas],nodatakey),R2_SetNodata([MuSyQ_r2,GLASS_r2,MODIS_r2,CASA_r2,W_r2]),'Weight_Year')
    # Multiply_Regression_Year(normalization(SetNodata([MuSyQ_datas,GLASS_datas,MODIS_datas,CASA_datas,W_datas,LAI_datas],nodatakey)),'Normal_Multiply_Regression_Year')
    # Multiply_Regression_Year(SetNodata([MuSyQ_datas,GLASS_datas,MODIS_datas,CASA_datas,W_datas,LAI_datas],nodatakey),'Multiply_Regression_Year')

    
    # normalization_Writearray_Spatial(SetNodata([MuSyQ_datas,GLASS_datas,MODIS_datas,CASA_datas,W_datas,LAI_datas],nodatakey))
    normalization_Writearray_Spatial_time(SetNodata([MuSyQ_datas,GLASS_datas,MODIS_datas,CASA_datas,W_datas,LAI_datas],nodatakey))
    sg.popup_notify(title = 'Task done!',display_duration_in_ms = 10000,fade_in_duration = 10000)

    
    
    