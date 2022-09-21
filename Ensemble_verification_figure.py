# -- coding: utf-8 --
# -- coding: utf-8 --
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import HalvingGridSearchCV
from sklearn.model_selection import HalvingRandomSearchCV
from lce import LCERegressor
import smtplib
import PySimpleGUI
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
import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import AdaBoostRegressor
import matplotlib; matplotlib.use('TkAgg')
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
import matplotlib as mpl
from osgeo import gdal, gdalconst
import numpy as np
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [*] %(processName)s %(message)s"
)
plt.rcParams['font.sans-serif'] = ['SimHei']
mpl.rcParams['axes.unicode_minus'] = False

MuSyQ_path = r'E:\Integrated_analysis_data\Data\1Y\MuSyQ_1981_2018_1y_chinese'
GLASS_path = r'E:\Integrated_analysis_data\Data\1Y\GLASS_1982_2018_1y_chinese'
MODIS_path = r'E:\Integrated_analysis_data\Data\1Y\MODIS_2000_2017_1y_chinese'
# CASA_path = r'K:\HeQiFan\1Y\TPDC_2000_2017_1y'
W_path = r'E:\Integrated_analysis_data\Data\1Y\GLOPEM-CEVSA_1980_2020_1y_chinese'
# LAI_path = r'K:\HeQiFan\1Y\LAI_2003_2017_1y'
vertify_xlsx = r'E:\Integrated_analysis_data\Data\NPP验证数据\Value.xlsx'
Sample_tif = r'E:\Integrated_analysis_data\Data\Sample\Mask_Mul_2009.tif'
Outpath = r'E:\Integrated_analysis_data\Data\Vertify_out'


MODIS_key,MuSyQ_key,W_key,GLASS_key = 'Mask_*.tif', 'Mask_*.tif', 'RNPP_*.flt','Mask_*.tif'
nodatakey = [['<-1000','<0'], ['<-1000','<0'], ['<-1000','<0'], ['<-1000','<0']]  # 每种模型的无效值
model_name = ['MODIS', 'MuSyQ', 'GLOPE', 'GLASS']



styear = 2014  # 开始年份
edyear = 2017  # 结束年份

# length = 5  # 模型的数量

minx_minx = 4998   #列数 4998
miny_miny = 4088  #行数  4088

MuSyQ_datas, GLASS_datas, MODIS_datas,W_datas = [], [], [], [] # 定义空的列表，存放每年的数据

years = [x for x in range(styear, edyear + 1)]  # 年份的列表
Pools = 10



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
                else:
                    da[da<=value] = np.nan
    return Datas


def Get_Model(x_data,y_data,forest_type):
    x_data = x_data.T.tolist()  # 转为一行一个模型的数据
    Index = ['Forest_type', 'R2', 'RMSE'] + forest_type
    model_all = [Index]
    all_r2 = []
    for num, x in enumerate(x_data):
        x = np.array(x)
        r2 = r2_score(y_data, x)
        rmse = sqrt(mean_squared_error(y_data, x))
        logging.info(f'-------{model_name[num]}的r2为: {r2}---------')
        logging.info(f'-------{model_name[num]}的rmse为: {rmse}---------')
        #sg.popup_ok(f'——————{model_name[num]}的r2，rmse 运行完毕————————')
        # Plot outputs
        get_plot(r2, rmse, x, y_data, model_name[num])
        all_r2.append(r2)
        model_all.append([model_name[num],r2,rmse] + list(x))
    y_ = ['Station', np.nan, np.nan] + y_data.flatten().tolist()
    model_all.append(y_)
    result_data = pd.DataFrame(transposition(model_all))
    result_data.to_excel(Outpath + os.sep + 'Model.xlsx',header=None,index = False)
    #sg.popup_ok(f'——————Model 运行完毕————————')
    return all_r2
def get_plot(r2,rmse,y_predict,y_data,name):
    plt.scatter(y_data, y_predict, color="black")
    # plt.plot(y_predict, y_data, color="blue", linewidth=3)
    # plt.plot(y_predict, y_data, 'ko')
    parameter = np.polyfit(y_data, y_predict, 1)  # n=1为一次函数，返回函数参数
    f = np.poly1d(parameter)  # 拼接方程
    plt.plot(y_data, f(y_data), "r--")
    plt.ylabel(name + '_Predict')
    plt.xlabel('NPP_Station')
    plt.text(x = 1,y = 1,s = f"R2: {r2}", fontsize=20)
    plt.text(x = 1,y = 0.9,s = f"RMSE: {rmse}", fontsize=20)
    plt.ion()
    plt.savefig(Outpath + os.sep + name + '_plot.jpg', dpi=600)
    plt.pause(1)
    plt.close()
def Weight(x_data,y_data):
    x_data  = x_data.T.tolist()   #转为一行一个模型的数据
    r2_total = 0
    y_predict = []
    for num,x in enumerate(x_data):
        x = np.array(x)
        r2 = r2_score(y_data, x)
        rmse = sqrt(mean_squared_error(y_data, x))
        logging.info(f'-------{model_name[num]}的r2为: {r2}---------')
        logging.info(f'-------{model_name[num]}的rmse为: {rmse}---------')
        # Plot outputs
        # get_plot(r2,rmse,x,y_data,model_name[num])
        y_predict.append(x*r2)
        r2_total += r2
    y_predict = np.array(y_predict).T.sum(axis = 1)/r2_total
    r2 = r2_score(y_data, y_predict)
    rmse = sqrt(mean_squared_error(y_data, y_predict))
    logging.info(f'-------Weight的r2为: {r2}---------')
    logging.info(f'-------Weight的rmse为: {rmse}---------')
    # Plot outputs
    get_plot(r2,rmse,y_predict,y_data,'Weight')
    y_predict_data = y_predict.flatten().tolist()
    #sg.popup_ok(f'——————Weight 运行完毕————————')
    return ['Weight'] + [r2, rmse] + y_predict_data
def Mean(x_data,y_data):
    y_predict = np.mean(x_data,axis =1)
    r2 = r2_score(y_data, y_predict)
    rmse = sqrt(mean_squared_error(y_data, y_predict))
    logging.info(f'-------Mean的r2为: {r2}---------')
    logging.info(f'-------Mean的rmse为: {rmse}---------')
    get_plot(r2, rmse, y_predict, y_data,'Mean')
    y_predict_data = y_predict.flatten().tolist()
    #sg.popup_ok(f'——————Mean 运行完毕————————')
    return ['Mean'] + [r2,rmse] + y_predict_data
def Median(x_data,y_data):
    y_predict = np.median(x_data,axis =1)
    r2 = r2_score(y_data, y_predict)
    rmse = sqrt(mean_squared_error(y_data, y_predict))
    logging.info(f'-------Median的r2为: {r2}---------')
    logging.info(f'-------Median的rmse为: {rmse}---------')
    get_plot(r2, rmse, y_predict, y_data,'Median')
    y_predict_data = y_predict.flatten().tolist()
    #sg.popup_ok(f'——————Median 运行完毕————————')
    return ['Median'] + [r2,rmse] + y_predict_data
def Mul(x_data,y_data,patten):
    '''Get Multiply_Regression_RR or Get Multiply_Regression Predicted Data'''
    model = LinearRegression(n_jobs = -1).fit(x_data, y_data.ravel())
    if patten == 'fit_tif':
        #sg.popup_ok(f'——————Mul model 运行完毕————————')
        return model
    else:
        y_predict = model.predict(x_data).astype('float32')
        r2 = r2_score(y_data, y_predict)
        rmse = sqrt(mean_squared_error(y_data, y_predict))
        logging.info(f'-------Mul的r2为: {r2}---------')
        logging.info(f'-------Mul的rmse为: {rmse}---------')
        get_plot(r2, rmse, y_predict, y_data,'Multiply_Regression')
        y_predict_data = y_predict.flatten().tolist()
        #sg.popup_ok(f'——————Mul 运行完毕————————')
        return ['Mul'] + [r2,rmse] + y_predict_data
def Bagging(x_data, y_data,patten):
    '''Get Bagging_RR or Get Bagging Predicted Data'''
    model = BaggingRegressor(random_state=123)
    param_distributions = {"n_estimators": range(1, 20)}
    model = HalvingGridSearchCV(model, param_distributions,random_state=123,factor=2,n_jobs = -1).fit(x_data, y_data.ravel())
    if patten == 'fit_tif':
        #sg.popup_ok(f'——————Bagging model 运行完毕————————')
        return model
    else:
        logging.info(f'-------Ba最佳参数为: {model.best_params_}-------')
        # logging.info(f'——————————Ba最佳得分为: {model.best_score_}——————————')
        # logging.info(f'——————————Ba最佳模型为: {model.best_estimator_}——————————')
        # logging.info(f'——————————Ba交叉验证拆分（折叠/迭代）的次数为: {model.n_splits_}——————————')
        # logging.info(f'——————————Ba实际运行的迭代次数为: {model.n_iterations_}——————————')
        y_predict = model.predict(x_data).astype('float32')
        r2 = r2_score(y_data, y_predict)
        logging.info(f'-------Ba的r2为: {r2}---------')
        rmse = sqrt(mean_squared_error(y_data, y_predict))
        logging.info(f'-------Ba的rmse为: {rmse}---------')
        get_plot(r2, rmse, y_predict, y_data, 'Bagging')
        y_predict_data = y_predict.flatten().tolist()
        #sg.popup_ok(f'——————Bagging 运行完毕————————')
        return ['Bagging'] + [r2,rmse] + y_predict_data
def AdaBoosting(x_data, y_data,patten):
    '''Get AdaBoost_RR or Get AdaBoost Predicted Data'''
    rng = np.random.RandomState(1)
    model = AdaBoostRegressor(DecisionTreeRegressor(max_depth=4),
                                random_state=rng)
    param_distributions = {"n_estimators": range(1, 10),
                            "loss":["linear", "square", "exponential"]
                            }
    model = HalvingGridSearchCV(model, param_distributions,
                                    random_state=rng,factor=2,n_jobs = -1).fit(x_data, y_data.ravel())
    if patten == 'fit_tif':
        #sg.popup_ok(f'——————Adaboost model 运行完毕————————')
        return model
    else:
        logging.info(f'-------Ada最佳参数为: {model.best_params_}---------')
        # logging.info(f'-------Ada最佳度量值为: {model.best_score_}---------')
        # logging.info(f'-------Ada最佳模型为: {model.best_estimator_}---------')
        # logging.info(f'-------Ada交叉验证拆分（折叠/迭代）的次数为: {model.n_splits_}---------')
        # logging.info(f'-------Ada实际运行的迭代次数为: {model.n_iterations_}---------')
        y_predict = model.predict(x_data).astype('float32')
        r2 = r2_score(y_data, y_predict)
        logging.info(f'-------Ada的r2为: {r2}---------')
        rmse = sqrt(mean_squared_error(y_data, y_predict))
        logging.info(f'-------Ada的rmse为: {rmse}---------')
        get_plot(r2, rmse, y_predict, y_data, 'AdaBoosting')
        y_predict_data = y_predict.flatten().tolist()
        #sg.popup_ok(f'——————Adaboost 运行完毕————————')
        return ['AdaBoosting'] + [r2, rmse] + y_predict_data
def GradientBoosting(x_data, y_data,patten):
    '''Get Gradient_RR or Get Gradient Predicted Data'''
    model = ensemble.GradientBoostingRegressor(random_state = 123)
    param_distributions = {"n_estimators": range(1, 20),
                            "max_depth": range(1, 20),
                            "loss":["squared_error", "absolute_error", "huber", "quantile"],
                            "criterion":["squared_error"]
                            }
    model = HalvingGridSearchCV(model, param_distributions,
                                    random_state=123,factor=2).fit(x_data, y_data.ravel())
    if patten == 'fit_tif':
        #sg.popup_ok(f'——————GradienBoosting model 运行完毕————————')
        return model
    else:
        logging.info(f'-------GradientBoosting最佳参数为: {model.best_params_}---------')
        # logging.info(f'-------Gra最佳度量值为: {model.best_score_}---------')
        # logging.info(f'-------Gra最佳模型为: {model.best_estimator_}---------')
        # logging.info(f'-------Gra交叉验证拆分（折叠/迭代）的次数为: {model.n_splits_}---------')
        # logging.info(f'-------Gra实际运行的迭代次数为: {model.n_iterations_}---------')
        y_predict = model.predict(x_data).astype('float32')
        r2 = r2_score(y_data, y_predict)
        logging.info(f'-------GradientBoosting的r2为: {r2}---------')
        rmse = sqrt(mean_squared_error(y_data, y_predict))
        logging.info(f'-------GradientBoosting的rmse为: {rmse}---------')
        get_plot(r2, rmse, y_predict, y_data, 'GradientBoosting')
        y_predict_data = y_predict.flatten().tolist()
        #sg.popup_ok(f'——————GradienBoosting 运行完毕————————')
        return ['GradientBoosting'] + [r2, rmse] + y_predict_data
def Stacking(x_data, y_data,patten):
    '''Get Stacking_RR or Get Stacking Predicted Data'''
    estimators = [('lr', RidgeCV()), ('svr', LinearSVR(random_state=42,max_iter=10000))]
    model = StackingRegressor(estimators=estimators,
                                final_estimator=RandomForestRegressor(random_state=42)).fit(x_data, y_data.ravel())
    if patten == 'fit_tif':
        #sg.popup_ok(f'——————Stacking moddel 运行完毕————————')
        return model
    else:
        y_predict = model.predict(x_data).astype('float32')
        r2 = r2_score(y_data, y_predict)
        logging.info(f'-------Sta的r2为: {r2}---------')
        rmse = sqrt(mean_squared_error(y_data, y_predict))
        logging.info(f'-------Sta的rmse为: {rmse}---------')
        get_plot(r2, rmse, y_predict, y_data, 'Stacking')
        y_predict_data = y_predict.flatten().tolist()
        #sg.popup_ok(f'——————Stacking 运行完毕————————')
        return ['Stacking'] + [r2, rmse] + y_predict_data
def RF(x_data, y_data,patten):
    '''Get RandomForestRegressor_RR or Get RandomForestRegressor Predicted Data'''
    model = RandomForestRegressor(random_state=0,n_jobs = -1)
    param_distributions = {"n_estimators": range(1, 10),"max_depth": range(1, 10),"criterion":["squared_error"]}
    model = HalvingGridSearchCV(model, param_distributions,random_state=0,factor=2,n_jobs = -1).fit(x_data, y_data.ravel())
    if patten == 'fit_tif':
        #sg.popup_ok(f'——————RF moddel 运行完毕————————')
        return model
    else:
        logging.info(f'-------RF最佳参数为: {model.best_params_}---------')
        # logging.info(f'-------RF最佳度量值为: {model.best_score_}---------')
        # logging.info(f'-------RF最佳模型为: {model.best_estimator_}---------')
        # logging.info(f'-------RF交叉验证拆分（折叠/迭代）的次数为: {model.n_splits_}---------')
        # logging.info(f'-------RF实际运行的迭代次数为: {model.n_iterations_}---------')
        y_predict = model.predict(x_data).astype('float32')
        r2 = r2_score(y_data, y_predict)
        logging.info(f'-------RF的r2为: {r2}---------')
        rmse = sqrt(mean_squared_error(y_data, y_predict))
        logging.info(f'-------RF的rmse为: {rmse}---------')
        get_plot(r2, rmse, y_predict, y_data, 'RF')
        y_predict_data = y_predict.flatten().tolist()
        #sg.popup_ok(f'——————RF 运行完毕————————')
        return ['RF'] + [r2, rmse] + y_predict_data
def LCE(x_data, y_data,patten):
    '''Get LCERegressor_RR or Get LCERegressor Predicted Data'''
    model = LCERegressor(random_state=123)
    param_distributions2 = {"n_estimators": range(1, 10),"max_depth": range(1, 10),"criterion":["squared_error"]}
    model = HalvingGridSearchCV(model, param_distributions2,
                                    random_state=123,factor=2,n_jobs = -1).fit(x_data, y_data.ravel())
    if patten == 'fit_tif':
        #sg.popup_ok(f'——————LCE model 运行完毕————————')
        return model
    else:
        logging.info(f'-------LCE最佳参数为: {model.best_params_}---------')
        # logging.info(f'-------LCE最佳度量值为: {model.best_score_}---------')
        # logging.info(f'-------LCE最佳模型为: {model.best_estimator_}---------')
        # logging.info(f'-------LCE交叉验证拆分（折叠/迭代）的次数为: {model.n_splits_}---------')
        # logging.info(f'-------LCE实际运行的迭代次数为: {model.n_iterations_}---------')
        y_predict = model.predict(x_data).astype('float32')
        r2 = r2_score(y_data, y_predict)
        logging.info(f'-------LCE的r2为: {r2}---------')
        rmse = sqrt(mean_squared_error(y_data, y_predict))
        logging.info(f'-------LCE的rmse为: {rmse}---------')
        get_plot(r2, rmse, y_predict, y_data, 'LCE')
        y_predict_data = y_predict.flatten().tolist()
        #sg.popup_ok(f'——————LCE 运行完毕————————')
        return ['LCE'] + [r2, rmse] + y_predict_data
def Vote(x_data, y_data,patten):
    '''Get Vote_RR or Get Vote Predicted Data'''
    # model1 = GradientBoostingRegressor(random_state=1)
    model1 = LCERegressor(random_state=123)
    param_distributions1 = {"n_estimators": range(1, 10),"max_depth": range(1, 10),"criterion":["squared_error"]}
    model1 = HalvingGridSearchCV(model1, param_distributions1,
                                    random_state=123,factor=2).fit(x_data, y_data.ravel())

    # model2 = RandomForestRegressor(random_state=1,n_jobs = -1)
    rng = np.random.RandomState(1)
    model2 = AdaBoostRegressor(DecisionTreeRegressor(max_depth=4),
                                random_state=rng)
    param_distributions2 = {"n_estimators": range(1, 10),
                            "loss":["linear", "square", "exponential"]
                            }
    model2 = HalvingGridSearchCV(model2, param_distributions2,
                                    random_state=rng,factor=2).fit(x_data, y_data.ravel())

    # model3 = LinearRegression(n_jobs = -1)
    model3 = RandomForestRegressor(random_state=0,n_jobs = -1)
    param_distributions3 = {"n_estimators": range(1, 10),"max_depth": range(1, 10),"criterion":["squared_error"]}
    model3 = HalvingGridSearchCV(model3, param_distributions3,random_state=0,factor=2).fit(x_data, y_data.ravel())

    estimators = [('lr', RidgeCV()), ('svr', LinearSVR(random_state=42,max_iter=10000))]
    model4 = StackingRegressor(estimators=estimators,
                                final_estimator=RandomForestRegressor(random_state=42)).fit(x_data, y_data.ravel())

    model = VotingRegressor([('LCE', model1), ('Ada', model2), ('RF', model3),('Sta', model4)]).fit(x_data, y_data.ravel())
    if patten == 'fit_tif':
        #sg.popup_ok(f'——————Vote model 运行完毕————————')
        return model
    else:
        y_predict = model.predict(x_data).astype('float32')
        r2 = r2_score(y_data, y_predict)
        logging.info(f'-------Vote的r2为: {r2}---------')
        rmse = sqrt(mean_squared_error(y_data, y_predict))
        logging.info(f'-------Vote的rmse为: {rmse}---------')
        get_plot(r2, rmse, y_predict, y_data, 'Vote')
        y_predict_data = y_predict.flatten().tolist()
        #sg.popup_ok(f'——————Vote 运行完毕————————')
        return ['Vote'] + [r2, rmse] + y_predict_data

def transposition(list_two_dimension):
    # b = tuple(zip(*list_two_dimension))
    # c = list(zip(*list_two_dimension))
    list_T = list(map(list, zip(*list_two_dimension)))
    return list_T

def Fit_Station(x_data, y_data,forest_type):
    Mean_data = Mean(x_data,y_data)
    Median_data = Median(x_data, y_data)
    Weight_data = Weight(x_data, y_data)
    Mul_data = Mul(x_data,y_data,'fit_station')
    Bagging_data = Bagging(x_data, y_data,'fit_station')
    AdaBoosting_data = AdaBoosting(x_data, y_data,'fit_station')
    GraBoosting_data = GradientBoosting(x_data, y_data,'fit_station')
    Staking_data = Stacking(x_data, y_data,'fit_station')
    RF_data = RF(x_data, y_data,'fit_station')
    LCE_data = LCE(x_data, y_data,'fit_station')
    Vote_data = Vote(x_data, y_data,'fit_station')
    Index = ['Forest_type', 'R2', 'RMSE'] + forest_type
    y_ = ['Station',np.nan,np.nan] + y_data.flatten().tolist()
    result_data = transposition([Index,Mean_data,Median_data,Weight_data,Mul_data,Bagging_data,AdaBoosting_data,GraBoosting_data,Staking_data,RF_data,LCE_data,Vote_data,y_])
    result_data = pd.DataFrame(result_data)
    result_data.to_excel(Outpath + os.sep + 'result.xlsx',header=None,index = False)

def A_WriteArray(datalist,Name,var_list):
    '''
    写出数据
    '''
    ds = gdal.Open(Sample_tif)                    # 打开文件
    Projection = ds.GetProjection()
    GeoTransform = ds.GetGeoTransform()
    RasterCount = ds.RasterCount
    del ds
    im_width = minx_minx                          # 获取栅格矩阵的列数
    im_height = miny_miny                         # 获取栅格矩阵的行数                    # 波段的indice起始为1，不为0
    img_datatype = gdal.GDT_Float32               # 数据类型
    outdir = Outpath + os.sep + Name
    logging.info(f'-------输出文件夹为 {outdir}---------')
    if not os.path.exists(outdir):       #判断原始文件路劲是否存在,如果不存在就直接退出
        os.makedirs(outdir)
    for j in range(0,len(datalist)):
        out_ds = gdal.GetDriverByName('GTiff').Create(
              outdir + os.sep + Name + "_" + str(var_list[j]) + '.tif',                   # tif文件所保存的路径
              im_width,                                          # 获取栅格矩阵的列数
              im_height,                                          # 获取栅格矩阵的行数
              RasterCount,                                     # 获取栅格矩阵的波段数
              img_datatype)                                       # 获取第一波段的数据类型
        out_ds.SetProjection(Projection)                # 投影信息
        out_ds.SetGeoTransform(GeoTransform)            # 仿射信息
        out_band = out_ds.GetRasterBand(1)
        data_replacenan = np.array(datalist[j])
        data_replacenan[np.isnan(data_replacenan)] = -9999
        out_band.WriteArray(data_replacenan.reshape(miny_miny,minx_minx).astype('float16'))    # 写入数据 (why)
        out_band.SetNoDataValue(-9999)
        out_ds.FlushCache()  #(刷新缓存)
        del data_replacenan
        del out_ds
        gc.collect()
        logging.info(f' {outdir + os.sep + Name + "_" + str(var_list[j]) + ".tif"} is  ok   !!!!!!!!')
def Fit_tif(model,images_pixels_x,nn):
    logging.info(f'-------正在处理的是: {nn}---------')
    pool.restart()
    model_list = [model]*len(images_pixels_x)
    mean_results = pool.map(Predicted, model_list,images_pixels_x)
    pool.close()
    pool.join()
    A_WriteArray(np.array(mean_results).astype('float16').T.tolist(), nn, years)
    logging.info(f'-------{nn} 处理结束---------')
    #sg.popup_ok(f'——————{nn} tif 运行完毕————————')

def Predicted(model,x_data):
    y_predict_data = []
    for x in x_data.tolist():
        if np.isnan(np.array(x)).any():
            y_predict = -9999
        else:
            try:
                y_predict = model.predict(np.array(x).reshape(1, -1)).tolist()[0]
            except TypeError:
                y_predict = model.predict(np.array(x).reshape(1, -1))
        y_predict_data.append(y_predict)
    return np.array(y_predict_data).astype('float16')

def Fit_Station_tif(x_data,y_data,Setnodata_datas,all_r2):
    # Mul_model = Mul(x_data,y_data,'fit_tif')
    # Bagging_model = Bagging(x_data, y_data,'fit_tif')
    # AdaBoosting_model = AdaBoosting(x_data, y_data,'fit_tif')
    # GraBoosting_model = GradientBoosting(x_data, y_data,'fit_tif')
    # Staking_model = Stacking(x_data, y_data,'fit_tif')
    # RF_model = RF(x_data, y_data,'fit_tif')
    # LCE_model = LCE(x_data, y_data,'fit_tif')
    Vote_mdoel = Vote(x_data, y_data,'fit_tif')

    # logging.info('-------正在处理的是: Mean.Median---------')
    # # Setnodata_datas_array = np.array(Setnodata_datas).astype('float16')
    # images_pixels_mean = np.nanmean(np.array(Setnodata_datas).astype('float16'),axis=0).astype('float16')
    # images_pixels_median = np.nanmedian(np.array(Setnodata_datas).astype('float16'),axis=0).astype('float16')
    # A_WriteArray(images_pixels_mean,'Mean',years)
    # A_WriteArray(images_pixels_median,'Median',years)
    # del images_pixels_mean
    # del images_pixels_median
    # # del Setnodata_datas_array
    # gc.collect()
    # logging.info('-------Mean.Median 处理完毕---------')
    #
    # logging.info('-------正在处理的是: Weight---------')
    # images_pixels_weight = np.array(np.nansum(np.array([Setnodata_datas[i] * all_r2[i] for i in range(len(Setnodata_datas))]),axis=0) / np.nansum(np.array(all_r2)).astype('float16'))
    # A_WriteArray(images_pixels_weight,'Weight',years)
    # del images_pixels_weight
    # gc.collect()
    # logging.info('-------Weight 处理完毕---------')
    #
    images_pixels_x = []  # 用于存放所有年，每年五种数据mean的值，一年一个列表
    for y in tqdm(range(miny_miny), desc='列数'):
        for x in range(minx_minx):
            images_pixels = []
            for year in range(len(years)):
                images_pixels.append(np.array([i[year][y][x] for i in Setnodata_datas]).astype('float16'))
            images_pixels_x.append(np.array(images_pixels).astype('float16'))
    #
    # Fit_tif(Mul_model,images_pixels_x,'Multiply_Regression')
    # Fit_tif(Bagging_model, images_pixels_x, 'Bagging')
    # Fit_tif(AdaBoosting_model, images_pixels_x, 'AdaBoosting')
    # Fit_tif(GraBoosting_model,images_pixels_x,'GradientBoosting')
    # Fit_tif(Staking_model, images_pixels_x, 'Stacking')
    # Fit_tif(RF_model, images_pixels_x, 'RF')
    # Fit_tif(LCE_model, images_pixels_x, 'LCERegressor')
    Fit_tif(Vote_mdoel, images_pixels_x, 'Vote')
    del images_pixels_x
    del Setnodata_datas
    gc.collect()


if __name__ == "__main__":
    vertify_npp = pd.read_excel(vertify_xlsx)
    vertify_clear = vertify_npp[vertify_npp['Mean_GLOPE'] != 0]
    vertify_clear = vertify_clear[vertify_clear['Mean_MODIS'] != -9999 ]
    # vertify_clear = vertify_clear.drop(index=vertify_clear.Mean_MODIS[vertify_clear.Mean_MODIS == -9999].index)
    vertify_clear = vertify_clear.groupby("Forest").mean()
    forest_type = list(vertify_clear.index)
    logging.info('\n')
    x_data_ = np.array(vertify_clear[['Mean_MODIS','Mean_MuSyQ','Mean_GLOPE','Mean_GLASS']])
    y_data_ = np.array(vertify_clear['NPP__t_ha_']).ravel()
    all_r2 = Get_Model(x_data_,y_data_,forest_type)   #返回所有模型的R2->列表

    # Fit_Station(x_data_,y_data_,forest_type)   #站点数据的拟合->输出基本模型和集成分析的预测数据在站点上的R2，RMSE，以及预测值

    logging.info('\n')
    logging.info('-----------------Start----------------------')
    logging.info(f'minx_minx: {minx_minx}')
    logging.info(f'miny_miny: {miny_miny}')

    for year in tqdm(range(styear, edyear + 1), desc='Year'):
        MuSyQ_dir = MuSyQ_path + os.sep + str(year)
        GLASS_dir = GLASS_path + os.sep + str(year)
        MODIS_dir = MODIS_path + os.sep + str(year)
        W_dir = W_path + os.sep + str(year)

        MuSyQ_datas.append(
            gdal.Open(g(MuSyQ_dir + os.sep + MuSyQ_key)[0], gdal.GA_ReadOnly).ReadAsArray(0, 0, minx_minx, miny_miny))
        GLASS_datas.append(
            gdal.Open(g(GLASS_dir + os.sep + GLASS_key)[0], gdal.GA_ReadOnly).ReadAsArray(0, 0, minx_minx, miny_miny))
        MODIS_datas.append(
            gdal.Open(g(MODIS_dir + os.sep + MODIS_key)[0], gdal.GA_ReadOnly).ReadAsArray(0, 0, minx_minx, miny_miny))
        W_datas.append(
            gdal.Open(g(W_dir + os.sep + W_key)[0], gdal.GA_ReadOnly).ReadAsArray(0, 0, minx_minx, miny_miny))

    MuSyQ_datas = np.array(MuSyQ_datas).astype('float16')
    GLASS_datas = np.array(GLASS_datas).astype('float16')
    MODIS_datas = np.array(MODIS_datas).astype('float16')
    W_datas = np.array(W_datas).astype('float16')

    setnodata = SetNodata([MODIS_datas, MuSyQ_datas,W_datas,GLASS_datas], nodatakey)

    pool = newPool(Pools)

    Fit_Station_tif(x_data_,y_data_, setnodata,all_r2)