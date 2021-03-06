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
from sklearn.metrics import mean_absolute_error
from math import sqrt
import time
import traceback
import random

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
processes = 6
MuSyQ_datas,GLASS_datas,MODIS_datas,CASA_datas,W_datas = [],[],[],[],[]


years = [year for year in range(styear,edyear+1)]


var = ['R2','RMSE','MSE','MAE']   



def  SetNodata(Datas,nodatakey):
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
                #     #print(f"????????????????????????????????????0?????????{value}")
                # elif  da.any() >= value:
                #     #print(f"??????????????????????????????????????????{value}")
                # elif da.any()<0:
                #     #print(f"????????????????????????????????????0??????")
                # #da = np.where(da>float(nodatakey[i][1:]),np.nan,da)
            if symbol == '<':
                # print('<')
                da[da<=value] = 0
                da[da<0] = 0
                # if da.all() > value and da.all()>=0:
                #     #print(f"????????????????????????????????????0?????????{value}")
                # elif  da.any() <= value:
                #     #print(f"??????????????????????????????????????????{value}??????")
                # elif da.any()<0:
                #     #print(f"????????????????????????????????????0??????")
            data_.append(da)
                #da = np.where(da<float(nodatakey[i][1:]),np.nan,da)
        i+=1
        datas_.append(data_)

    return datas_




def  SetDatatype(Datas):
    datas_ = []
    for data in Datas:
        data_ = []
        for da in data:
            #print('data:',data)
            #da.astype('uint16')
            print(f'????????????????????????{da.dtype}')
            da.dtype = np.uint32
            print(f'????????????????????? {da.dtype}')
            data_.append(da)
        datas_.append(data_)
    return datas_



def  WriteArray(datalist,Outpath,minxsize,minysize,var,Type):
    
    sample_tif = r'F:\Integrated_analysis_data\Data\TPDC_2000_2017_1y\2000\Resample_2000_npp.tif'                          # ???????????????tif???????????????????????????????????????????????????    
    ds = gdal.Open(sample_tif)                             # ????????????
    im_width = minxsize                           # ???????????????????????????
    print(f'im_width: {im_width}')
    im_height = minysize                          # ???????????????????????????
    print(f'im_height: {im_height}')
    im_bands = ds.RasterCount                     # ??????????????????????????????
    #print(f'im_bands: {im_bands}')
    band1 = ds.GetRasterBand(1)                         # ?????????indice?????????1?????????0
    img_datatype = band1.DataType                       # ????????????

    outdir = Outpath + os.sep + Type
    print(f'-------?????????????????? {outdir}---------')
    if not os.path.exists(outdir):       #????????????????????????????????????,??????????????????????????????
        os.makedirs(outdir)
    for j in range(0,len(datalist)):    
        out_ds = gdal.GetDriverByName('GTiff').Create(
              outdir + os.sep + var[j] + '.tif',                   # tif????????????????????????
              im_width,                                     # ???????????????????????????
              im_height,                                     # ???????????????????????????
              ds.RasterCount,                                     # ??????????????????????????????
              img_datatype)                                       # ?????????????????????????????????
        out_ds.SetProjection(ds.GetProjection())                # ????????????
        out_ds.SetGeoTransform(ds.GetGeoTransform())            # ????????????
        for i in range(1, ds.RasterCount + 1):                  # ?????????????????????
            out_band = out_ds.GetRasterBand(i)
            out_band.WriteArray(datalist[j])                           # ???????????? (why)
            out_ds.FlushCache()  #(????????????)
            del out_ds #?????? 
    
            print(f' {outdir + os.sep + var[j] + ".tif"} is  ok   !!!!!!!!')
    del ds
    
    
    
    

def RR(mean_data,years):
    print('%s run' %os.getpid())
    time.sleep(random.random())
    y_data = mean_data
    x_train,x_test,y_train,y_test = train_test_split(years,y_data,train_size=0.8)
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


def workMulti(province,years):
    try:
        RR(province,years)
    except Exception as e:
        print('Error: %s' % (province), traceback.print_exc())



def median(data):
    data.sort()
    half = len(data) // 2
    return (data[half] + data[~half])/2

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
    
    print('?????????????????????')
    Setnodata_Datas = SetNodata(Datas,nodatakey)
    
    print('????????????????????????')
    #SetDatatypes = SetDatatype(setnodata_Datas)
    
    '''Mean'''
    start = datetime.datetime.now()
    print(f'Mean??????????????????{start}')
    # Mean(Setnodata_Datas,Outpath,var,minx_minx,miny_miny)
    images_pixels = []
    #datas = SetDatatype(datas)
    for year in range(len(years)):
        images_pixels.append(list(((Setnodata_Datas[0][year].reshape(miny_miny,minx_minx) + Setnodata_Datas[1][year].reshape(miny_miny,minx_minx) + Setnodata_Datas[2][year].reshape(miny_miny,minx_minx) + Setnodata_Datas[3][year].reshape(miny_miny,minx_minx) + Setnodata_Datas[4][year].reshape(miny_miny,minx_minx))/len(years)).flatten()))

    # images = []
    # for image in images_pixels:
    #     #print(image)
    #     images.append(list(image.flatten()))
    # images_array = pd.DataFrame(images).T
    images_array = np.array(images_pixels).T.tolist()
    #images_pixels_2 = [list(x.flatten()) for x in images_pixels]
    #print(images_pixels_2.shape)
    print('-------------???????????????:Mean---------------')
    pool = Pool(processes)         # ???????????????
    All_data=[]
    for images in images_array:
        print("Add task:", images)
        # ??????????????????workMulti?????????????????????province,arr???
        # ??????????????????????????????????????????????????????
        res = pool.apply_async(workMulti, args=(images,years,))
        All_data.append(res)
    print('------------???????????????----------------')
    pool.close()
    pool.join() #????????????????????????????????????????????????????????????????????????
    print('----------????????????????????????---------------')
    # nums=[]
    # for res in All_data:
    #     print(res.get())
    #     nums.append(res.get()) #??????????????????
    # datalist = [pd.DataFrame(nums)[:0],pd.DataFrame(nums)[:1],pd.DataFrame(nums)[:2],pd.DataFrame(nums)[:3]]
    # WriteArray(datalist,Outpath,minx_minx,miny_miny,var,'Mean')
    # print('??????')
    # end = datetime.datetime.now()
    # print('totally time is ', end - start)
    
    
    
    # '''Median'''
    # start = datetime.datetime.now()
    # print(f'Median??????????????????{start}')
    # images_pixels2 = []
    # for year in range(len(years)):
    #     images_pixels3 = []
    #     for y in range(miny_miny):
    #         for x in range(minx_minx):
    #             images_pixels3.append(median([Setnodata_Datas[0][y][x],Setnodata_Datas[1][y][x], Setnodata_Datas[2][y][x] , Setnodata_Datas[3][y][x] ,Setnodata_Datas[4][y][x]]))
    #     images_pixels2.append(images_pixels3)
    
    # images_array = np.array(images_pixels).T.tolist()
    
    # pool = Pool(processes)         # ???????????????
    # All_data=[]
    # for images in images_array:
    #     print("Add task:", images)
    #     # ??????????????????workMulti?????????????????????province,arr???
    #     # ??????????????????????????????????????????????????????
    #     res = pool.apply_async(workMulti, args=(images,years,))
    #     All_data.append(res)
    # print('------------???????????????----------------')
    # pool.close()
    # pool.join() #????????????????????????????????????????????????????????????????????????
    # print('----------????????????????????????---------------')
    # nums=[]
    # for res in All_data:
    #     print(res.get())
    #     nums.append(res.get()) #??????????????????
    # datalist = [pd.DataFrame(nums)[:0],pd.DataFrame(nums)[:1],pd.DataFrame(nums)[:2],pd.DataFrame(nums)[:3]]
    # WriteArray(datalist,Outpath,minx_minx,miny_miny,var,'Median')
    # print('??????')
    # end = datetime.datetime.now()
    # print('totally time is ', end - start)
    

    
    
    