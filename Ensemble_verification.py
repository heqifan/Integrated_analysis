# -- coding: utf-8 --
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import HalvingGridSearchCV
from sklearn.model_selection import HalvingRandomSearchCV
from lce import LCERegressor
import smtplib
from email.mime.text import MIMEText
from email.header import Header
import gc
from sklearn.ensemble import VotingRegressor
from sklearn.linear_model import RidgeCV
from sklearn.svm import LinearSVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import StackingRegressor
from sklearn import model_selection
from sklearn import datasets, ensemble
from sklearn.inspection import permutation_importance
from sklearn.ensemble import BaggingRegressor
from sklearn import metrics
from tqdm import tqdm
import PySimpleGUI as sg
from scipy.stats import randint
import os
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import AdaBoostRegressor
import sklearn
from glob import glob as g
from osgeo import gdal
import datetime
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from math import sqrt
from sklearn.ensemble import GradientBoostingRegressor
import logging
from sklearn import linear_model
from sklearn.linear_model import LinearRegression
from multiprocessing.dummy import Pool as ThreadPool
from pathos.multiprocessing import ProcessingPool as newPool
from sklearn.model_selection import HalvingGridSearchCV
import os,sys,glob,rasterio
from osgeo import osr,ogr
import pandas as pd
from osgeo import gdal, gdalconst
import numpy as np
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [*] %(processName)s %(message)s"
)

vertify_xlsx = r'E:\Integrated_analysis_data\Data\NPP验证数据\NPP验证数据.xlsx'

Outpath = r'E:\Integrated_analysis_data\Data\Vertify_out'

Sample_tif = r'K:\HeQiFan\Sample\Mask_Mul_2009.tif'

MuSyQ_inpath = r'K:\HeQiFan\1Y\Geodata_2000_2017_1y'
GLASS_inpath = r'K:\HeQiFan\1Y\GLASS_2000_2017_1y'
MODIS_path = r'K:\HeQiFan\1Y\MODIS_2000_2017_1y'
CASA_path = r'K:\HeQiFan\1Y\TPDC_2000_2017_1y'
W_path = r'K:\HeQiFan\1Y\W_2000_2017_1y'


MuSyQ_key, GLASS_key, MODIS_key, CASA_key, W_key = 'Mask_*.tif', 'Mask_*.tif', 'Mask_*.tif', 'Mask_*.tif', 'Resample_*.tif'  # 关键字

nodatakey = [['<-1000'], ['<-1000'], ['<-1000'], ['<-1000'], ['<-1000'], ['<-1000']]  # 每种模型的无效值

na_me = ['Geodata', 'GLASS', 'MODIS', 'TPDC', 'W']

Pools = 10
length = 5  # 模型的数量
styear = 2003  # 开始年份
edyear = 2017  # 结束年份
minx_minx = 2671   #列数
miny_miny =  2101  #行数

years = [x for x in range(styear, edyear + 1)]  # 年份的列表
Sample_tif = r'E:\Integrated_analysis_data\Data\Sample\Resample_Mask_2007.tif'
MuSyQ_datas, GLASS_datas, MODIS_datas, CASA_datas, W_datas, LAI_datas = [], [], [], [], [], []  # 定义空的列表，存放每年的数据

vertify_npp = pd.read_excel(vertify_xlsx)
vertify_npp["NPP (t ha-1 yr-1)"] = vertify_npp["NPP (t ha-1 yr-1)"]*50
vertify_groupby = vertify_npp.groupby("Forest").mean()


def SetNodata(Datas,nodatakey):
    '''
    设置无效值
    '''
    for data,key in zip(Datas,nodatakey):
        for da in data:
            for k in key:
                symbol = k[0]       #获取符号
                value = int(k[1:])  #获取数组
                if symbol == '>':
                    da[da>=value] = np.nan
                    da[da<0] = np.nan
                else:
                    da[da<=value] = np.nan
                    da[da<0] = np.nan
    return Datas

def getSRSPair(dataset):
    '''
    得到给定数据的投影参考系和地理参考系
    :param dataset: GDAL地理数据
    :return: 投影参考系和地理参考系
    '''
    prosrs = osr.SpatialReference()
    prosrs.ImportFromWkt(dataset.GetProjection())
    geosrs = prosrs.CloneGeogCS()
    return prosrs, geosrs

def Get_lon_lat(Sample_tif):
    '''
    Args:
    input:
        Sample_tif:  输入的tif数据的路径
    return:
        arr_x:    tif数据中像元对应的经度的数组
        arr_y:    tif数据中像元对应的纬度的数组
        im_data:  tif数据
    '''

    dataset = gdal.Open(Sample_tif)  # 打开tif

    adfGeoTransform = dataset.GetGeoTransform()  # 读取地理信息

    # 左上角地理坐标
    print('左上角x地理坐标：', adfGeoTransform[0])
    print('左上角y地理坐标：', adfGeoTransform[3])

    nXSize = dataset.RasterXSize  # 列数
    nYSize = dataset.RasterYSize  # 行数
    print('列数为：', nXSize, '行数为：', nYSize)

    prosrs, geosrs = getSRSPair(dataset)
    ct = osr.CoordinateTransformation(prosrs, geosrs)
    arr_lon = []  # 用于存储每个像素的（X，Y）坐标
    arr_lat = []
    for i in range(nYSize):
        row_lon = []
        row_lat = []
        for j in range(nXSize):
            px = adfGeoTransform[0] + j * adfGeoTransform[1] + i * adfGeoTransform[2]
            py = adfGeoTransform[3] + j * adfGeoTransform[4] + i * adfGeoTransform[5]
            coords = ct.TransformPoint(px, py)
            row_lon.append(coords[1])
            row_lat.append(coords[0])
            # row_lat.append(px)
            # row_lon.append(py)
        arr_lon.append(row_lon)
        arr_lat.append(row_lat)
    del dataset
    return np.array(arr_lon) , np.array(arr_lat)

def Get_imagedata(Sample_tif):
    '''
    Args:
    input:
        Sample_tif:  输入的tif数据的路径
    return:
        arr_x:    tif数据中像元对应的经度的数组
        arr_y:    tif数据中像元对应的纬度的数组
        im_data:  tif数据
    '''

    dataset = gdal.Open(Sample_tif)  # 打开tif
    nXSize = dataset.RasterXSize  # 列数
    nYSize = dataset.RasterYSize  # 行数
    im_data = dataset.ReadAsArray(0, 0, nXSize, nYSize)  # 将数据写成数组，对应栅格矩阵
    del dataset
    return im_data


if __name__ == "__main__":
    print('-----------------Start----------------------')
    print(f'minx_minx: {minx_minx}')
    print(f'miny_miny: {miny_miny}')
    sample_lon, sample_lat = Get_lon_lat(Sample_tif)
    sample_lon = np.round(sample_lon, decimals=2)
    sample_lat = np.round(sample_lat, decimals=2)
    for year in tqdm(range(styear, edyear + 1), desc='Year'):
        MuSyQ_data  = Get_imagedata(g(MuSyQ_inpath + os.sep + str(year) + os.sep + MuSyQ_key)[0])
        GLASS_data = Get_imagedata(g(GLASS_inpath + os.sep + str(year) + os.sep + GLASS_key)[0])
        MODIS_data = Get_imagedata(g(MODIS_path + os.sep + str(year) + os.sep + MODIS_key)[0])
        CASA_data = Get_imagedata(g(CASA_path + os.sep + str(year) + os.sep + CASA_key)[0])
        W_data = Get_imagedata(g(W_path + os.sep + str(year) + os.sep + W_key)[0])

    # MuSyQ_datas = np.array(MuSyQ_datas).astype('float32')
    # GLASS_datas = np.array(GLASS_datas).astype('float32')
    # MODIS_datas = np.array(MODIS_datas).astype('float32')
    # CASA_datas = np.array(CASA_datas).astype('float32')
    # W_datas = np.array(W_datas).astype('float32')
    #
    # pool = newPool(Pools)
    #
    # set = SetNodata([MuSyQ_datas, GLASS_datas, MODIS_datas, CASA_datas, W_datas, LAI_datas], nodatakey)
    # Mean_Median_RR(nor,'Liner_Mean','Liner_Median')
    # Cal_R2(nor)

    # MuSyQ_r2 = gdal.Open(MuSyQ_R2,gdal.GA_ReadOnly).ReadAsArray(0, 0, minx_minx, miny_miny)
    # GLASS_r2 = gdal.Open(GLASS_R2,gdal.GA_ReadOnly).ReadAsArray(0, 0, minx_minx, miny_miny)
    # MODIS_r2 = gdal.Open(MODIS_R2,gdal.GA_ReadOnly).ReadAsArray(0, 0, minx_minx, miny_miny)
    # CASA_r2 = gdal.Open(CASA_R2,gdal.GA_ReadOnly).ReadAsArray(0, 0, minx_minx, miny_miny)
    # W_r2 = gdal.Open(W_R2,gdal.GA_ReadOnly).ReadAsArray(0, 0, minx_minx, miny_miny)
    # all_R2 = np.array([MuSyQ_r2,GLASS_r2,MODIS_r2,CASA_r2,W_r2])

    # Weight_RR(nor,R2_SetNodata(all_R2),'Liner_Weight')

    # Multiply_Regression_RR(nor,'Liner_Mul')
    # Bagging_RR(nor, 'Liner_Bagging')
    # Ada_RR(nor, 'Liner_AdaBoost')
    # Gra_RR(nor, 'Liner_Gradient')
    # Sta_RR(nor, 'Liner_Stacking')
    # RF_RR(nor, 'Liner_RandomForestRegressor')
    # LCE_RR(nor, 'Liner_LCERegressor')
    # Vote_RR(nor, 'Liner_Vote')
    # '''再计算每种方法的每年的值（归一化和没有归一化的）'''
    # Mean_Median_Year(nor,'Normal_Mean_Year','Normal_Median_Year')
    # Mean_Median_Year(set,'Mean_Year','Median_Year')
    # Weight_Year(nor,R2_SetNodata([MuSyQ_r2,GLASS_r2,MODIS_r2,CASA_r2,W_r2]),'Normal_Weight_Year')
    # Weight_Year(set,R2_SetNodata([MuSyQ_r2,GLASS_r2,MODIS_r2,CASA_r2,W_r2]),'Weight_Year')
    # Multiply_Regression_Year(nor,'Normal_Multiply_Regression_Year')
    # Multiply_Regression_Year(set,'Multiply_Regression_Year')
    # Bagging_Year(nor, 'Normal_Bagging_Year')
    # Bagging_Year(set, 'Bagging_Year')
    # # Ada_Year(nor, 'Normal_AdaBoost_Year')
    # # Ada_Year(set, 'AdaBoost_Year')
    # # Gra_Year(nor, 'Normal_Gradient_Year')
    # # Gra_Year(set, 'Gradient_Year')
    # # Sta_Year(nor, 'Normal_Stacking_Year')
    # # Sta_Year(set, 'Stacking_Year')
    # # RF_Year(nor, 'Normal_RandomForestRegressor_Year')
    # # RF_Year(set, 'RandomForestRegressor_Year')
    # # LCE_Year(nor, 'Normal_LCERegressor_Year')
    # # LCE_Year(set, 'LCERegressor_Year')
    # # Vote_Year(nor, 'Normal_VoteRegressor_Year')
    # # Vote_Year(set, 'VoteRegressor_Year')
    # normalization_Writearray_Spatial(set)
    # normalization_Writearray_Spatial_time(set)
    # del nor
    # del set
    # gc.collect()
    # sg.popup_notify(title='Task done!', display_duration_in_ms=1000, fade_in_duration=1000)
