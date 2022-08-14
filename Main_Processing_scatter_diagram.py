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
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from math import sqrt
import time
import random
import logging
from sklearn import linear_model
from sklearn.linear_model import LinearRegression  # 引入多元线性回归算法模块进行相应的训练
from multiprocessing.dummy import Pool as ThreadPool
from pathos.multiprocessing import ProcessingPool as newPool

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [*] %(processName)s %(message)s"
)

Outpath = r'K:\HeQiFan\Out'  # 输出路径

MuSyQ_inpath = r'K:\HeQiFan\Out\Normal_Geodata\Mean\Mean_Normal_Geodata.tif'
GLASS_inpath = r'K:\HeQiFan\Out\Normal_GLASS\Mean\Mean_Normal_GLASS.tif'
MODIS_path = r'K:\HeQiFan\Out\Normal_MODIS\Mean\Mean_Normal_MODIS.tif'
CASA_path = r'K:\HeQiFan\Out\Normal_TPDC\Mean\Mean_Normal_TPDC.tif'
W_path = r'K:\HeQiFan\Out\Normal_W\Mean\Mean_Normal_W.tif'
LAI_path = r'K:\HeQiFan\Out\Normal_LAI\Mean\Mean_Normal_LAI.tif'
nodatakey = ['<-1000','<-1000','<-1000','<-1000','<-1000','<-1000']
na_me2 = ['Geodata', 'GLASS', 'MODIS', 'TPDC', 'W', 'LAI']

length = 5  # 模型的数量

minx_minx = 2671  # 列数
miny_miny = 2101  # 行数

MuSyQ_datas, GLASS_datas, MODIS_datas, CASA_datas, W_datas, LAI_datas = [], [], [], [], [], []  # 定义空的列表，存放每年的数据

'''预处理函数'''
def SetNodata(Datas, nodatakey):
    '''
    设置无效值
    '''
    datas_ = []
    for data, key in zip(Datas, nodatakey):
        symbol = key[0]  # 获取符号
        value = int(key[1:])  # 获取数组
        if symbol == '>':
            data[data >= value] = np.nan
        else:
            data[data <= value] = np.nan
        datas_.append(data)
    return datas_

def A_WriteArray(datalist, Name, var_list):
    '''
    写出数据
    '''
    sample_tif = r'K:\HeQiFan\Sample\Reproject_Mul_2002.tif'  # 需要读取的tif文件所在的文件夹的所在文件夹的路径
    ds = gdal.Open(sample_tif)  # 打开文件
    im_width = minx_minx  # 获取栅格矩阵的列数
    # logging.info(f'im_width: {im_width}')
    im_height = miny_miny  # 获取栅格矩阵的行数
    # logging.info(f'im_height: {im_height}')
    # im_bands = ds.RasterCount                     # 获取栅格矩阵的波段数
    # print(f'im_bands: {im_bands}')
    # band1 = ds.GetRasterBand(1)                         # 波段的indice起始为1，不为0
    img_datatype = gdal.GDT_Float32  # 数据类型
    outdir = Outpath + os.sep + Name
    logging.info(f'-------输出文件夹为 {outdir}---------')
    if not os.path.exists(outdir):  # 判断原始文件路劲是否存在,如果不存在就直接退出
        os.makedirs(outdir)
    for j in range(0, len(datalist)):
        out_ds = gdal.GetDriverByName('GTiff').Create(
            outdir + os.sep + Name + "_" + str(var_list[j]) + '.tif',  # tif文件所保存的路径
            im_width,  # 获取栅格矩阵的列数
            im_height,  # 获取栅格矩阵的行数
            ds.RasterCount,  # 获取栅格矩阵的波段数
            img_datatype)  # 获取第一波段的数据类型
        out_ds.SetProjection(ds.GetProjection())  # 投影信息
        out_ds.SetGeoTransform(ds.GetGeoTransform())  # 仿射信息
        out_band = out_ds.GetRasterBand(1)
        out_band.WriteArray(np.array(datalist[j]).reshape(miny_miny, minx_minx))  # 写入数据 (why)
        out_ds.FlushCache()  # (刷新缓存)
        del out_ds  # 删除
        logging.info(f' {outdir + os.sep + Name + "_" + str(var_list[j]) + ".tif"} is  ok   !!!!!!!!')
    del ds

def L_R(mean_data, y_data, r_name):
    '''Get liner_Regression_R2 or get liner_Regression_RR'''

    if np.isnan(np.array(mean_data)).any() or np.isnan(np.array(y_data)).any():
        if r_name == 'R2':
            return [np.nan]
        elif r_name == 'RR':
            return [np.nan, np.nan, np.nan, np.nan]
    else:
        model = linear_model.LinearRegression()
        model.fit(np.array(mean_data).reshape(-1, 1), np.array(y_data).reshape(-1, 1))
        y_predict = model.predict(np.array(mean_data).reshape(-1, 1))
        r2 = r2_score(np.array(y_data).reshape(-1, 1), np.array(y_predict))
        mse = mean_squared_error(np.array(y_data).reshape(-1, 1), np.array(y_predict).reshape(-1, 1))
        mae = mean_absolute_error(np.array(y_data).reshape(-1, 1), np.array(y_predict).reshape(-1, 1))
        rmse = sqrt(mse)
        if r_name == 'R2':
            return [r2]
        elif r_name == 'RR':
            return [r2, mse, mae, rmse]


if __name__ == "__main__":
    # __spec__ = "ModuleSpec(name='builtins', loader=<class '_frozen_importlib.BuiltinImporter'>)"
    print('-----------------Start----------------------')
    print(f'minx_minx: {minx_minx}')
    print(f'miny_miny: {miny_miny}')
    MuSyQ_ = gdal.Open(MuSyQ_inpath, gdal.GA_ReadOnly).ReadAsArray(0, 0, minx_minx, miny_miny)
    GLASS_ = gdal.Open(GLASS_inpath, gdal.GA_ReadOnly).ReadAsArray(0, 0, minx_minx, miny_miny)
    MODIS_ = gdal.Open(MODIS_path, gdal.GA_ReadOnly).ReadAsArray(0, 0, minx_minx, miny_miny)
    CASA_ = gdal.Open(CASA_path, gdal.GA_ReadOnly).ReadAsArray(0, 0, minx_minx, miny_miny)
    W_ = gdal.Open(W_path, gdal.GA_ReadOnly).ReadAsArray(0, 0, minx_minx, miny_miny)
    LAI_ = gdal.Open(LAI_path, gdal.GA_ReadOnly).ReadAsArray(0, 0, minx_minx, miny_miny)

    result = SetNodata([MuSyQ_, GLASS_, MODIS_, CASA_, W_, LAI_], nodatakey)
    new_result = [x.flatten().tolist() for x in result ]
    new_result = np.array(new_result).T
    new_result = new_result[~np.isnan(new_result).any(axis=1), :]
    row_rand_array = np.arange(new_result.shape[0])
    np.random.shuffle(row_rand_array)
    row_rand = new_result[row_rand_array[0:2]]
    sg.popup_notify(title='Task done!', display_duration_in_ms=1000, fade_in_duration=1000)

