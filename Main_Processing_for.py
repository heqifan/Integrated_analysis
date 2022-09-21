# -- coding: utf-8 --

"""
Created on Mon Jul  4 20:58:11 2022

@author: HYF
"""
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
import pandas as pd
import numpy as np
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

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [*] %(processName)s %(message)s"
)

Outpath = r'J:\Integrated_analysis_data\Data\Out'

Sample_tif = r'J:\Integrated_analysis_data\Data\Sample\Mask_Mul_2005.tif'

MuSyQ_inpath = r'J:\Integrated_analysis_data\Data\1Y\Geodata_2000_2017_1y'
GLASS_inpath = r'J:\Integrated_analysis_data\Data\1Y\GLASS_2000_2017_1y'
MODIS_path = r'J:\Integrated_analysis_data\Data\1Y\MODIS_2000_2017_1y'
CASA_path = r'J:\Integrated_analysis_data\Data\1Y\TPDC_2000_2017_1y'
W_path = r'J:\Integrated_analysis_data\Data\1Y\W_2000_2017_1y'
LAI_path = r'J:\Integrated_analysis_data\Data\1Y\LAI_2003_2017_1y'

MuSyQ_R2 = r'J:\Integrated_analysis_data\Data\Out\Model_R2\R2_Geodata\R2_Geodata_.tif'  # 每种模型的R2，weight中要用
GLASS_R2 = r'J:\Integrated_analysis_data\Data\Out\Model_R2\R2_GLASS\R2_GLASS_.tif'
MODIS_R2 = r'J:\Integrated_analysis_data\Data\Out\Model_R2\R2_MODIS\R2_MODIS_.tif'
CASA_R2 = r'J:\Integrated_analysis_data\Data\Out\Model_R2\R2_TPDC\R2_TPDC_.tif'
W_R2 = r'J:\Integrated_analysis_data\Data\Out\Model_R2\R2_W\R2_W_.tif'

MuSyQ_key, GLASS_key, MODIS_key, CASA_key, W_key, LAI_key = 'Mask_*.tif', 'Mask_*.tif', 'Mask_*.tif', 'Mask_*.tif', 'Resample_*.tif', 'Mask_*.tif'  # 关键字

nodatakey = [['<-1000'], ['<-1000'], ['<-1000'], ['<-1000'], ['<-1000'], ['<-1000']]  # 每种模型的无效值

na_me = ['Geodata', 'GLASS', 'MODIS', 'TPDC', 'W']
na_me2 = ['Geodata', 'GLASS', 'MODIS', 'TPDC', 'W', 'LAI']


def message(text):
    mail_host = "smtp.qq.com"  # 设置的邮件服务器host必须是发送邮箱的服务器，与接收邮箱无关。
    mail_user = "2051936579"  # qq邮箱登陆名
    mail_pass = "ctbvgvslaakyecci"  # 开启stmp服务的时候并设置的授权码，注意！不是QQ密码。

    sender = '2051936579@qq.com'  # 发送方qq邮箱
    receivers = ['2051936579@qq.com']  # 接收方qq邮箱

    message = MIMEText(text, 'plain', 'utf-8')
    message['From'] = Header("heqifan", 'utf-8')  # 设置显示在邮件里的发件人
    message['To'] = Header("heqifan", 'utf-8')  # 设置显示在邮件里的收件人

    subject = 'python smtp email test'
    message['Subject'] = Header(subject, 'utf-8')  # 设置主题和格式

    try:
        smtpobj = smtplib.SMTP_SSL(mail_host, 465)  # 本地如果有本地服务器，则用localhost ,默认端口２５,腾讯的（端口465或587）
        smtpobj.set_debuglevel(1)
        smtpobj.login(mail_user, mail_pass)  # 登陆QQ邮箱服务器
        smtpobj.sendmail(sender, receivers, message.as_string())  # 发送邮件
        print("邮件发送成功")
        smtpobj.quit()  # 退出
    except smtplib.SMTPException as e:
        print("Error:无法发送邮件")
        print(e)


# Pools = 8
length = 5  # 模型的数量
styear = 2003  # 开始年份
edyear = 2017  # 结束年份

minx_minx = 2671  # 列数
miny_miny = 2101  # 行数

years = [x for x in range(styear, edyear + 1)]  # 年份的列表

MuSyQ_datas, GLASS_datas, MODIS_datas, CASA_datas, W_datas, LAI_datas = [], [], [], [], [], []  # 定义空的列表，存放每年的数据
# var = ['R2','RMSE','MSE','MAE']
var = ['R2', 'RMSE']
'''预处理函数'''


def SetNodata(Datas, nodatakey):
    '''
    设置无效值
    '''
    for data, key in zip(Datas, nodatakey):
        for da in data:
            for k in key:
                symbol = k[0]  # 获取符号
                value = int(k[1:])  # 获取数组
                if symbol == '>':
                    da[da >= value] = np.nan
                    da[da < 0] = np.nan
                else:
                    da[da <= value] = np.nan
                    da[da < 0] = np.nan
    return Datas


def R2_SetNodata(Datas):
    '''
    设置无效值
    '''
    for da in Datas:
        da[da < 0] = np.nan
    return Datas


def SetDatatype(Datas):
    '''
    设置数据类型
    '''
    datas_ = []
    for data in Datas:
        data_ = []
        for da in data:
            da.dtype = np.uint32
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
            da = (da - min_value) / (max_value - min_value)
            data_.append(da)
        data_ = np.array(data_).astype('float16')
        datas_.append(data_)
    return datas_


'''Write'''


def A_WriteArray(datalist, Name, var_list):
    '''
    写出数据
    '''
    ds = gdal.Open(Sample_tif)  # 打开文件
    im_width = minx_minx  # 获取栅格矩阵的列数
    im_height = miny_miny  # 获取栅格矩阵的行数                    # 波段的indice起始为1，不为0
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
        out_band.WriteArray(np.array(datalist[j]).reshape(miny_miny, minx_minx).astype('float16'))  # 写入数据 (why)
        out_ds.FlushCache()  # (刷新缓存)
        logging.info(f' {outdir + os.sep + Name + "_" + str(var_list[j]) + ".tif"} is  ok   !!!!!!!!')
    del ds

def M_R_P(mean_data, y_data, r_name):
    '''Get Multiply_Regression_RR or Get Multiply_Regression Predicted Data'''
    mean_data = np.array(mean_data).astype('float16')
    y_data = np.array(y_data).reshape(-1, 1).astype('float16')
    if r_name == 'RR':
        if np.isnan(mean_data).any() or np.isnan(y_data).any():
            return [-9999, -9999]
        else:
            model = LinearRegression()
            model.fit(mean_data, y_data.ravel())
            y_predict = model.predict(mean_data).reshape(-1, 1).astype('float16')
            r2 = r2_score(y_data, y_predict)
            # mse =  mean_squared_error(y_data, y_predict)
            # mae = mean_absolute_error(y_data, y_predict)
            rmse = sqrt(mean_squared_error(y_data, y_predict))
            return [r2, rmse]
    else:
        if np.isnan(mean_data).any() or np.isnan(y_data).any():
            return [-9999] * len(years)
        else:
            model = LinearRegression()
            model.fit(mean_data, y_data.ravel())
            y_predict_data = model.predict(mean_data).flatten().tolist()
            return y_predict_data


def Ba_R_P(mean_data, y_data, r_name):
    '''Get Bagging_RR or Get Bagging Predicted Data'''
    mean_data = np.array(mean_data).astype('float16')
    y_data = np.array(y_data).reshape(-1, 1).astype('float16')
    if r_name == 'RR':
        if np.isnan(mean_data).any() or np.isnan(y_data).any():
            return [-9999, -9999]
        else:
            model = BaggingRegressor()
            model.fit(mean_data, y_data.ravel())
            y_predict = model.predict(mean_data).reshape(-1, 1).astype('float16')
            r2 = r2_score(y_data, y_predict)
            # mse =  mean_squared_error(y_data, y_predict)
            # mae = mean_absolute_error(y_data, y_predict)
            rmse = sqrt(mean_squared_error(y_data, y_predict))
            return [r2, rmse]
    else:
        if np.isnan(mean_data).any() or np.isnan(y_data).any():
            return [-9999] * len(years)
        else:
            model = BaggingRegressor()
            model.fit(mean_data, y_data.ravel())
            y_predict_data = model.predict(mean_data).flatten().tolist()
            return y_predict_data


def Ada_R_P(mean_data, y_data, r_name):
    '''Get AdaBoost_RR or Get AdaBoost Predicted Data'''
    mean_data = np.array(mean_data).astype('float16')
    y_data = np.array(y_data).reshape(-1, 1).astype('float16')
    rng = np.random.RandomState(1)
    if r_name == 'RR':
        if np.isnan(mean_data).any() or np.isnan(y_data).any():
            return [-9999, -9999]
        else:
            model = AdaBoostRegressor(DecisionTreeRegressor(max_depth=4),
                                      n_estimators=50, random_state=rng)
            model.fit(mean_data, y_data.ravel())
            y_predict = model.predict(mean_data).reshape(-1, 1).astype('float16')
            r2 = r2_score(y_data, y_predict)
            # mse =  mean_squared_error(y_data, y_predict)
            # mae = mean_absolute_error(y_data, y_predict)
            rmse = sqrt(mean_squared_error(y_data, y_predict))
            return [r2, rmse]
    else:
        if np.isnan(mean_data).any() or np.isnan(y_data).any():
            return [-9999] * len(years)
        else:
            model = AdaBoostRegressor(DecisionTreeRegressor(max_depth=4),
                                      n_estimators=50, random_state=rng)
            model.fit(mean_data, y_data.ravel())
            y_predict_data = model.predict(mean_data).flatten().tolist()
            return y_predict_data


def Gra_R_P(mean_data, y_data, r_name):
    '''Get Gradient_RR or Get Gradient Predicted Data'''
    mean_data = np.array(mean_data).astype('float16')
    y_data = np.array(y_data).reshape(-1, 1).astype('float16')
    params = {
        "n_estimators": 50,
        "max_depth": 4,
        "min_samples_split": 5,
        "learning_rate": 0.01,
        "loss": "squared_error",
    }
    if r_name == 'RR':
        if np.isnan(mean_data).any() or np.isnan(y_data).any():
            return [-9999, -9999]
        else:
            model = ensemble.GradientBoostingRegressor(**params)
            model.fit(mean_data, y_data.ravel())
            y_predict = model.predict(mean_data).reshape(-1, 1).astype('float16')
            r2 = r2_score(y_data, y_predict)
            # mse =  mean_squared_error(y_data, y_predict)
            # mae = mean_absolute_error(y_data, y_predict)
            rmse = sqrt(mean_squared_error(y_data, y_predict))
            return [r2, rmse]
    else:
        if np.isnan(mean_data).any() or np.isnan(y_data).any():
            return [-9999] * len(years)
        else:
            model = ensemble.GradientBoostingRegressor(**params)
            model.fit(mean_data, y_data.ravel())
            y_predict_data = model.predict(mean_data).flatten().tolist()
            return y_predict_data


def Sta_R_P(mean_data, y_data, r_name):
    '''Get Stacking_RR or Get Stacking Predicted Data'''
    mean_data = np.array(mean_data).astype('float16')
    y_data = np.array(y_data).reshape(-1, 1).astype('float16')
    estimators = [('lr', RidgeCV()), ('svr', LinearSVR(random_state=42))]
    if r_name == 'RR':
        if np.isnan(mean_data).any() or np.isnan(y_data).any():
            return [-9999, -9999]
        else:
            model = StackingRegressor(estimators=estimators,
                                      final_estimator=RandomForestRegressor(n_estimators=10, random_state=42))
            model.fit(mean_data, y_data.ravel())
            y_predict = model.predict(mean_data).reshape(-1, 1).astype('float16')
            r2 = r2_score(y_data, y_predict)
            # mse =  mean_squared_error(y_data, y_predict)
            # mae = mean_absolute_error(y_data, y_predict)
            rmse = sqrt(mean_squared_error(y_data, y_predict))
            return [r2, rmse]
    else:
        if np.isnan(mean_data).any() or np.isnan(y_data).any():
            return [-9999] * len(years)
        else:
            model = StackingRegressor(estimators=estimators,
                                      final_estimator=RandomForestRegressor(n_estimators=10, random_state=42))
            model.fit(mean_data, y_data.ravel())
            y_predict_data = model.predict(mean_data).flatten().tolist()
            return y_predict_data


def RF_R_P(mean_data, y_data, r_name):
    '''Get RandomForestRegressor_RR or Get RandomForestRegressor Predicted Data'''
    mean_data = np.array(mean_data).astype('float16')
    y_data = np.array(y_data).reshape(-1, 1).astype('float16')
    if r_name == 'RR':
        if np.isnan(mean_data).any() or np.isnan(y_data).any():
            return [-9999, -9999]
        else:
            model = RandomForestRegressor(random_state=0)
            model.fit(mean_data, y_data.ravel())
            y_predict = model.predict(mean_data).reshape(-1, 1).astype('float16')
            r2 = r2_score(y_data, y_predict)
            # mse =  mean_squared_error(y_data, y_predict)
            # mae = mean_absolute_error(y_data, y_predict)
            rmse = sqrt(mean_squared_error(y_data, y_predict))
            return [r2, rmse]
    else:
        if np.isnan(mean_data).any() or np.isnan(y_data).any():
            return [-9999] * len(years)
        else:
            model = RandomForestRegressor(random_state=0)
            model.fit(mean_data, y_data.ravel())
            y_predict_data = model.predict(mean_data).flatten().tolist()
            return y_predict_data


def Vote_R_P(mean_data, y_data, r_name):
    '''Get Vote_RR or Get Vote Predicted Data'''
    mean_data = np.array(mean_data).astype('float16')
    y_data = np.array(y_data).reshape(-1, 1).astype('float16')
    if r_name == 'RR':
        if np.isnan(mean_data).any() or np.isnan(y_data).any():
            return [-9999, -9999]
        else:
            model1 = GradientBoostingRegressor(random_state=1)
            model2 = RandomForestRegressor(random_state=1)
            model3 = LinearRegression()
            model1.fit(mean_data, y_data)
            model2.fit(mean_data, y_data)
            model3.fit(mean_data, y_data)

            model = VotingRegressor([('gb', model1), ('rf', model2), ('lr', model3)])
            model.fit(mean_data, y_data.ravel())
            y_predict = model.predict(mean_data).reshape(-1, 1).astype('float16')
            r2 = r2_score(y_data, y_predict)
            # mse =  mean_squared_error(y_data, y_predict)
            # mae = mean_absolute_error(y_data, y_predict)
            rmse = sqrt(mean_squared_error(y_data, y_predict))
            return [r2, rmse]
    else:
        if np.isnan(mean_data).any() or np.isnan(y_data).any():
            return [-9999] * len(years)
        else:
            model1 = GradientBoostingRegressor(random_state=1)
            model2 = RandomForestRegressor(random_state=1)
            model3 = LinearRegression()
            model1.fit(mean_data, y_data)
            model2.fit(mean_data, y_data)
            model3.fit(mean_data, y_data)

            model = VotingRegressor([('gb', model1), ('rf', model2), ('lr', model3)])
            model.fit(mean_data, y_data.ravel())

            y_predict_data = model.predict(mean_data).flatten().tolist()
            return y_predict_data


def L_R(mean_data, y_data, r_name):
    '''Get liner_Regression_R2 or get liner_Regression_RR'''
    mean_data = np.array(mean_data).reshape(-1, 1).astype('float16')
    y_data = np.array(y_data).reshape(-1, 1).astype('float16')
    if np.isnan(mean_data).any() or np.isnan(y_data).any():
        if r_name == 'R2':
            return [-9999]
        elif r_name == 'RR':
            return [-9999, -9999]
    else:
        model = linear_model.LinearRegression()
        model.fit(mean_data, y_data.ravel())
        y_predict = model.predict(mean_data).reshape(-1, 1).astype('float16')
        r2 = r2_score(y_data, np.array(y_predict))
        # mse =  mean_squared_error(y_data, y_predict)
        # mae = mean_absolute_error(y_data, y_predict)
        rmse = sqrt(mean_squared_error(y_data, y_predict))
        if r_name == 'R2':
            return [r2]
        elif r_name == 'RR':
            return [r2, rmse]


def Cal_R2(Setnodata_datas):
    start = datetime.datetime.now()
    for name, da in tqdm(enumerate(Setnodata_datas[:-1]), desc='Cal_R2'):
        images_pixels1 = da.reshape(da.shape[0], da.shape[1] * da.shape[2]).astype('float16').T.tolist()
        images_pixels5 = Setnodata_datas[-1].reshape(Setnodata_datas[-1].shape[0],
                                                     Setnodata_datas[-1].shape[1] * Setnodata_datas[-1].shape[
                                                         2]).astype('float16').T.tolist()
        name_list = ['R2'] * images_pixels1.shape[0]
        print('————————————————————————————————')
        print('———————————R2 Pool Start—————————————————————')
        try:
            # pool.restart()
            mean_results = []
            for i in tqdm(zip(images_pixels1, images_pixels5, name_list)):
                results = L_R(i[0],i[1],i[2])
                mean_results.append(results)
            # mean_results = pool.map(L_R, images_pixels1, images_pixels5, name_list)
            # pool.close()
            # pool.join()
            # message('R2 运行成功了hhhhhhh')
        except:
            pass # message('R2 运行失败')
        print('————————————————————————————————')
        print('———————————R2 Pool End—————————————————————')
        A_WriteArray([np.array(mean_results)[:, 0].astype('float16')], 'R2_' + na_me[name], [''])
    end = datetime.datetime.now()
    sg.popup_notify(f' Cal_R2     ALL  Task done!!!! Spend-time: {end - start}')


'''total'''


def Mean_Median_RR(Setnodata_datas, nn_mean, nn_median):
    print('——————————————Mean,Median——————————————————')
    Setnodata_datas = np.array(Setnodata_datas)  # 变为一个四维数组
    images_pixels1 = np.nanmean(Setnodata_datas[:-1, :, :, :], axis=0)  # 将所有模型求平均
    images_pixels2 = np.nanmedian(Setnodata_datas[:-1, :, :, :], axis=0)  # 将将所有模型求中值
    images_pixels5 = Setnodata_datas[-1, :, :, :]
    images_pixels1 = images_pixels1.reshape(images_pixels1.shape[0],
                                            images_pixels1.shape[1] * images_pixels1.shape[2]).astype('float16').T.tolist()  # 转置
    images_pixels5 = images_pixels5.reshape(images_pixels5.shape[0],
                                            images_pixels5.shape[1] * images_pixels5.shape[2]).astype('float16').T.tolist()  # 转置
    images_pixels2 = images_pixels2.reshape(images_pixels2.shape[0],
                                            images_pixels2.shape[1] * images_pixels2.shape[2]).astype('float16').T.tolist()  # 转置
    print('————————————————————————————————')
    print('———————————Mean Pool Start—————————————————————')
    name_list = ['RR'] * images_pixels1.shape[0]
    try:
        # pool.restart()
        mean_results = []
        for i in tqdm(zip(images_pixels1, images_pixels5, name_list)):
            results = L_R(i[0], i[1], i[2])
            mean_results.append(results)
        # mean_results = pool.map(L_R, images_pixels1, images_pixels5, name_list)
        # pool.close()
        # pool.join()
        # message(nn_mean + '运行成功了hhhhhhh')
    except:
        pass #message(nn_mean + '运行报错了,快来看看吧55555')
    print('———————————Mean Pool End—————————————————————')
    print('————————————————————————————————')
    mean_results = np.array(mean_results).astype('float16')
    A_WriteArray([mean_results[:, i] for i in range(mean_results.shape[1])], nn_mean, var)
    print('————————————————————————————————')
    print('———————————Median Pool Start—————————————————————')
    name_list = ['RR'] * images_pixels2.shape[0]
    try:
        # pool.restart()
        mean_results = []
        for i in tqdm(zip(images_pixels1, images_pixels5, name_list)):
            results = L_R(i[0], i[1], i[2])
            mean_results.append(results)
        # mean_results = pool.map(L_R, images_pixels2, images_pixels5, name_list)
        # pool.close()
        # pool.join()
        # message(nn_median + '运行成功了hhhhhhh')
    except:
        pass #message(nn_median + '运行报错了,快来看看吧55555')
    print('———————————Median Pool End—————————————————————')
    print('————————————————————————————————')
    mean_results = np.array(mean_results).astype('float16')
    A_WriteArray([mean_results[:, i] for i in range(mean_results.shape[1])], nn_median, var)


def Mean_Median_Year(Setnodata_datas, nn_mean, nn_median):
    print('——————————————Mean,Median——————————————————')
    start = datetime.datetime.now()
    Setnodata_datas = np.array(Setnodata_datas).astype('float16')
    images_pixels1 = np.nanmean(Setnodata_datas[:-1, :, :, :], axis=0).astype('float16')
    images_pixels2 = np.nanmedian(Setnodata_datas[:-1, :, :, :], axis=0).astype('float16')
    A_WriteArray(images_pixels1, nn_mean, years)
    A_WriteArray(images_pixels2, nn_median, years)
    sg.popup_notify(f'Mean-Median_Year Task done! Spend-time: {datetime.datetime.now() - start}')


def Weight_RR(Setnodata_datas, R2_SetNodata, nn):
    print('——————————————Weight——————————————————')
    start = datetime.datetime.now()
    images_pixels1 = np.array(
        np.nansum(np.array([Setnodata_datas[i] * R2_SetNodata[i] for i in range(len(Setnodata_datas[:-1]))]),
                  axis=0) / np.nansum(np.array(R2_SetNodata), axis=0)).astype('float16')
    images_pixels5 = Setnodata_datas[-1].astype('float16')
    images_pixels1 = images_pixels1.reshape(images_pixels1.shape[0],
                                            images_pixels1.shape[1] * images_pixels1.shape[2]).astype('float16').T.tolist()  # 转置
    images_pixels5 = images_pixels5.reshape(images_pixels5.shape[0],
                                            images_pixels5.shape[1] * images_pixels5.shape[2]).astype('float16').T.tolist()  # 转置
    print('————————————————————————————————')
    print('———————————Weight Pool Start—————————————————————')
    name_list = ['RR'] * images_pixels1.shape[0]
    try:
        #
        mean_results = []
        for i in tqdm(zip(images_pixels1, images_pixels5, name_list)):
            results = L_R(i[0], i[1], i[2])
            mean_results.append(results)
        # mean_results = pool.map(L_R, images_pixels1, images_pixels5, name_list)
        # pool.close()
        # pool.join()
        #####message(nn + '运行成功了hhhhhhh')
    except:
        pass #message(nn + '运行报错了,快来看看吧55555')
    print('————————————————————————————————')
    print('———————————Weight Pool End—————————————————————')
    mean_results = np.array(mean_results).astype('float16')
    A_WriteArray([mean_results[:, i] for i in range(mean_results.shape[1])], nn, var)
    sg.popup_notify(f'Weight Task done! Spend-time: {datetime.datetime.now() - start}', display_duration_in_ms=100,
                    fade_in_duration=100)


def Weight_Year(Setnodata_datas, R2_SetNodata, nn):
    print('——————————————Weight Year——————————————————')
    start = datetime.datetime.now()
    images_pixels1 = np.array(
        np.nansum(np.array([Setnodata_datas[i] * R2_SetNodata[i] for i in range(len(Setnodata_datas[:-1]))]),
                  axis=0) / np.nansum(np.array(R2_SetNodata), axis=0)).astype('float16')
    A_WriteArray(images_pixels1, nn, years)
    sg.popup_notify(f'Weight_Year Task done! Spend-time: {datetime.datetime.now() - start}', display_duration_in_ms=100,
                    fade_in_duration=100)


def Multiply_Regression_RR(Setnodata_datas, nn):
    print('——————————————Multiply_Regression——————————————————')
    start = datetime.datetime.now()
    images_pixels1 = []  # 用于存放所有年，每年五种数据mean的值，一年一个列表
    images_pixels5 = []
    for y in tqdm(range(miny_miny), desc='Multiply_Regression'):
        for x in range(minx_minx):
            images_pixels3 = []
            images_pixels6 = []
            for year in range(len(years)):
                images_pixels3.append(np.array([i[year][y][x] for i in Setnodata_datas[:-1]]).astype('float16'))
                images_pixels6.append(Setnodata_datas[-1][year][y][x])
            images_pixels1.append(np.array(images_pixels3).astype('float16'))
            images_pixels5.append(np.array(images_pixels6).astype('float16'))
    print('————————————————————————————————')
    print('———————————Multiply_Regression_RR Pool Start—————————————————————')
    name_list = ['RR'] * images_pixels1.shape[0]
    try:
        # pool.restart()
        mean_results = []
        for i in tqdm(zip(images_pixels1, images_pixels5, name_list)):
            results = M_R_P(i[0], i[1], i[2])
            mean_results.append(np.array(results).astype('float16'))
        # mean_results = pool.map(M_R_P, images_pixels1, images_pixels5, name_list)
        # pool.close()
        # pool.join()
        #####message(nn + '运行成功了hhhhhhh')
    except:
        pass #message(nn + '运行报错了,快来看看吧55555')
    print('————————————————————————————————')
    print('———————————Multiply_Regression_RR Pool End—————————————————————')
    mean_results = np.array(mean_results).astype('float16')
    A_WriteArray([mean_results[:, i] for i in range(mean_results.shape[1])], nn, var)
    sg.popup_notify(f'Multiply_Regression Task done! Spend-time: {datetime.datetime.now() - start}',
                    display_duration_in_ms=1000, fade_in_duration=1000)


def Multiply_Regression_Year(Setnodata_datas, nn):
    print('——————————————Multiply_Regression  Year——————————————————')
    start = datetime.datetime.now()
    images_pixels1 = []  # 用于存放所有年，每年五种数据mean的值，一年一个列表
    images_pixels5 = []
    for y in tqdm(range(miny_miny), desc='Multiply_Regression_Year'):
        for x in range(minx_minx):
            images_pixels3 = []
            images_pixels6 = []
            for year in range(len(years)):
                images_pixels3.append(np.array([i[year][y][x] for i in Setnodata_datas[:-1]]).astype('float16'))
                images_pixels6.append(Setnodata_datas[-1][year][y][x])
            images_pixels1.append(np.array(images_pixels3))
            images_pixels5.append(images_pixels6)
    print('————————————————————————————————')
    print('———————————Multiply_Regression_Year Pool Start—————————————————————')
    name_list = ['Predicted'] * len(images_pixels1)
    try:
        # pool.restart()
        mean_results = []
        for i in tqdm(zip(images_pixels1, images_pixels5, name_list)):
            results = M_R_P(i[0], i[1], i[2])
            mean_results.append(np.array(results).astype('float16'))
        # mean_results = pool.map(M_R_P, images_pixels1, images_pixels5, name_list)
        # pool.close()
        # pool.join()
        #####message(nn + '运行成功了hhhhhhh')
    except:
        pass #message(nn + '运行报错了,快来看看吧55555')
    print('————————————————————————————————')
    print('———————————Multiply_Regression_Year Pool End—————————————————————')
    A_WriteArray(np.array(mean_results).astype('float16').T.tolist(), nn, years)
    sg.popup_notify(f'Multiply_Regression_Year Task done! Spend-time: {datetime.datetime.now() - start}',
                    display_duration_in_ms=1000, fade_in_duration=1000)


def Bagging_RR(Setnodata_datas, nn):
    print('——————————————Bagging——————————————————')
    start = datetime.datetime.now()
    images_pixels1 = []  # 用于存放所有年，每年五种数据mean的值，一年一个列表
    images_pixels5 = []
    for y in tqdm(range(miny_miny), desc='Bagging'):
        for x in range(minx_minx):
            images_pixels3 = []
            images_pixels6 = []
            for year in range(len(years)):
                images_pixels3.append(np.array([i[year][y][x] for i in Setnodata_datas[:-1]]).astype('float16'))
                images_pixels6.append(Setnodata_datas[-1][year][y][x])
            images_pixels1.append(np.array(images_pixels3).astype('float16'))
            images_pixels5.append(np.array(images_pixels6).astype('float16'))
    print('————————————————————————————————')
    print('———————————Bagging_RR Pool Start—————————————————————')
    name_list = ['RR'] * len(images_pixels1)
    # pool.restart()
    mean_results = []
    for i in tqdm(zip(images_pixels1, images_pixels5, name_list)):
        print(i[0], i[1], i[2])
        results = Ba_R_P(i[0], i[1], i[2])
        mean_results.append(np.array(results).astype('float16'))
    # mean_results = pool.map(Ba_R_P, images_pixels1, images_pixels5, name_list)
    # pool.close()
    # pool.join()
    print('————————————————————————————————')
    print('———————————Bagging_RR Pool End—————————————————————')
    mean_results = np.array(mean_results).astype('float16')
    A_WriteArray([mean_results[:, i] for i in range(mean_results.shape[1])], nn, var)
    sg.popup_notify(f'Bagging Task done! Spend-time: {datetime.datetime.now() - start}',
                    display_duration_in_ms=100, fade_in_duration=100)


def Bagging_Year(Setnodata_datas, nn):
    print('——————————————Bagging_Year——————————————————')
    start = datetime.datetime.now()
    images_pixels1 = []  # 用于存放所有年，每年五种数据mean的值，一年一个列表
    images_pixels5 = []
    for y in tqdm(range(miny_miny), desc='Bagging_Year'):
        for x in range(minx_minx):
            images_pixels3 = []
            images_pixels6 = []
            for year in range(len(years)):
                images_pixels3.append(np.array([i[year][y][x] for i in Setnodata_datas[:-1]]).astype('float16'))
                images_pixels6.append(Setnodata_datas[-1][year][y][x])
            images_pixels1.append(np.array(images_pixels3).astype('float16'))
            images_pixels5.append(np.array(images_pixels6).astype('float16'))
    print('————————————————————————————————')
    print('———————————Bagging_Year Pool Start—————————————————————')
    name_list = ['Predicted'] * len(images_pixels1)
    try:
        # pool.restart()
        mean_results = []
        for i in tqdm(zip(images_pixels1, images_pixels5, name_list)):
            results = Ba_R_P(i[0], i[1], i[2])
            print(results)
            mean_results.append(np.array(results).astype('float16'))
        # mean_results = pool.map(Ba_R_P, images_pixels1, images_pixels5, name_list)
        # pool.close()
        # pool.join()
        #####message(nn + '运行成功了hhhhhhh')
    except:
        pass #message(nn + '运行报错了,快来看看吧55555')
    print('————————————————————————————————')
    print('———————————Bagging_Year Pool End—————————————————————')
    A_WriteArray(np.array(mean_results).astype('float16').T.tolist(), nn, years)
    sg.popup_notify(f'Bagging_Year Task done! Spend-time: {datetime.datetime.now() - start}',
                    display_duration_in_ms=100, fade_in_duration=100)


def Ada_RR(Setnodata_datas, nn):
    print('——————————————AdaBoost RR——————————————————')
    start = datetime.datetime.now()
    images_pixels1 = []  # 用于存放所有年，每年五种数据mean的值，一年一个列表
    images_pixels5 = []
    for y in tqdm(range(miny_miny), desc='AdaBoost_RR'):
        for x in range(minx_minx):
            images_pixels3 = []
            images_pixels6 = []
            for year in range(len(years)):
                images_pixels3.append(np.array([i[year][y][x] for i in Setnodata_datas[:-1]]).astype('float16'))
                images_pixels6.append(np.array(Setnodata_datas[-1][year][y][x]).astype('float16'))
            images_pixels1.append(np.array(images_pixels3).astype('float16'))
            images_pixels5.append(np.array(images_pixels6).astype('float16'))
    print('————————————————————————————————')
    print('———————————AdaBoost_RR Pool Start—————————————————————')
    name_list = ['RR'] * len(images_pixels1)
    try:
        # pool.restart()
        mean_results = []
        for i in tqdm(zip(images_pixels1, images_pixels5, name_list)):
            results = Ada_R_P(i[0], i[1], i[2])
            print(results)
            mean_results.append(np.array(results).astype('float16'))
        # mean_results = pool.map(Ada_R_P, images_pixels1, images_pixels5, name_list)
        # pool.close()
        # pool.join()
        #####message(nn + '运行成功了hhhhhhh')
    except:
        pass #message(nn + '运行报错了,快来看看吧55555')
    print('————————————————————————————————')
    print('———————————AdaBoost_RR Pool End—————————————————————')
    mean_results = np.array(mean_results).astype('float16')
    A_WriteArray([mean_results[:, i] for i in range(mean_results.shape[1])], nn, var)
    sg.popup_notify(f'AdaBoost Task done! Spend-time: {datetime.datetime.now() - start}',
                    display_duration_in_ms=100, fade_in_duration=100)


def Ada_Year(Setnodata_datas, nn):
    print('——————————————Ada_Year——————————————————')
    start = datetime.datetime.now()
    images_pixels1 = []  # 用于存放所有年，每年五种数据mean的值，一年一个列表
    images_pixels5 = []
    for y in tqdm(range(miny_miny), desc='Ada_Year'):
        for x in range(minx_minx):
            images_pixels3 = []
            images_pixels6 = []
            for year in range(len(years)):
                images_pixels3.append(np.array([i[year][y][x] for i in Setnodata_datas[:-1]]).astype('float16'))
                images_pixels6.append(Setnodata_datas[-1][year][y][x])
            images_pixels1.append(np.array(images_pixels3).astype('float16'))
            images_pixels5.append(np.array(images_pixels6).astype('float16'))
    print('————————————————————————————————')
    print('———————————Ada_Year Pool Start—————————————————————')
    name_list = ['Predicted'] * len(images_pixels1)
    try:
        # pool.restart()
        mean_results = []
        for i in tqdm(zip(images_pixels1, images_pixels5, name_list)):
            results = Ada_R_P(i[0], i[1], i[2])
            mean_results.append(np.array(results).astype('float16'))
        # mean_results = pool.map(Ada_R_P, images_pixels1, images_pixels5, name_list)
        # pool.close()
        # pool.join()
        #####message(nn + '运行成功了hhhhhhh')
    except:
        pass #message(nn + '运行报错了,快来看看吧55555')
    print('————————————————————————————————')
    print('———————————Ada_Year Pool End—————————————————————')
    A_WriteArray(np.array(mean_results).astype('float16').T.tolist(), nn, years)
    sg.popup_notify(f'Ada_Year Task done! Spend-time: {datetime.datetime.now() - start}',
                    display_duration_in_ms=100, fade_in_duration=100)


def Gra_RR(Setnodata_datas, nn):
    print('——————————————Gradient RR——————————————————')
    start = datetime.datetime.now()
    images_pixels1 = []  # 用于存放所有年，每年五种数据mean的值，一年一个列表
    images_pixels5 = []
    for y in tqdm(range(miny_miny), desc = 'Gradient_RR'):
        for x in range(minx_minx):
            images_pixels3 = []
            images_pixels6 = []
            for year in range(len(years)):
                images_pixels3.append(np.array([i[year][y][x] for i in Setnodata_datas[:-1]]).astype('float16'))
                images_pixels6.append(Setnodata_datas[-1][year][y][x])
            images_pixels1.append(np.array(images_pixels3).astype('float16'))
            images_pixels5.append(np.array(images_pixels6).astype('float16'))
    print('————————————————————————————————')
    print('———————————Gradient_RR Pool Start—————————————————————')
    name_list = ['RR'] * len(images_pixels1)
    try:
        # pool.restart()
        mean_results = []
        for i in tqdm(zip(images_pixels1, images_pixels5, name_list)):
            results = Gra_R_P(i[0], i[1], i[2])
            print(results)
            mean_results.append(np.array(results).astype('float16'))
        # mean_results = pool.map(Gra_R_P, images_pixels1, images_pixels5, name_list)
        # pool.close()
        # pool.join()
        #####message(nn + '运行成功了hhhhhhh')
    except:
        pass #message(nn + '运行报错了,快来看看吧55555')
    print('————————————————————————————————')
    print('———————————Gradient_RR Pool End—————————————————————')
    mean_results = np.array(mean_results).astype('float16')
    A_WriteArray([mean_results[:, i] for i in range(mean_results.shape[1])], nn, var)
    sg.popup_notify(f'Gradient Task done! Spend-time: {datetime.datetime.now() - start}',
                    display_duration_in_ms=100, fade_in_duration=100)


def Gra_Year(Setnodata_datas, nn):
    print('——————————————Gradient_Year——————————————————')
    start = datetime.datetime.now()
    images_pixels1 = []  # 用于存放所有年，每年五种数据mean的值，一年一个列表
    images_pixels5 = []
    for y in tqdm(range(miny_miny), desc='Gradient_Year'):
        for x in range(minx_minx):
            images_pixels3 = []
            images_pixels6 = []
            for year in range(len(years)):
                images_pixels3.append(np.array([i[year][y][x] for i in Setnodata_datas[:-1]]).astype('float16'))
                images_pixels6.append(Setnodata_datas[-1][year][y][x])
            images_pixels1.append(np.array(images_pixels3).astype('float16'))
            images_pixels5.append(np.array(images_pixels6).astype('float16'))
    print('————————————————————————————————')
    print('———————————Gradient_Year Pool Start—————————————————————')
    name_list = ['Predicted'] * len(images_pixels1)
    try:
        # pool.restart()
        mean_results = []
        for i in tqdm(zip(images_pixels1, images_pixels5, name_list)):
            results = Gra_R_P(i[0], i[1], i[2])
            mean_results = mean_results.append(np.array(results).astype('float16'))
        # mean_results = pool.map(Gra_R_P, images_pixels1, images_pixels5, name_list)
        # pool.close()
        # pool.join()
        #####message(nn + '运行成功了hhhhhhh')
    except:
        pass #message(nn + '运行报错了,快来看看吧55555')
    print('————————————————————————————————')
    print('———————————Gradient_Year Pool End—————————————————————')
    A_WriteArray(np.array(mean_results).astype('float16').T.tolist(), nn, years)
    sg.popup_notify(f'Gradient_Year Task done! Spend-time: {datetime.datetime.now() - start}',
                    display_duration_in_ms=100, fade_in_duration=100)


def Sta_RR(Setnodata_datas, nn):
    print('——————————————Stacking RR——————————————————')
    start = datetime.datetime.now()
    images_pixels1 = []  # 用于存放所有年，每年五种数据mean的值，一年一个列表
    images_pixels5 = []
    for y in tqdm(range(miny_miny), desc='Stacking_RR'):
        for x in range(minx_minx):
            images_pixels3 = []
            images_pixels6 = []
            for year in range(len(years)):
                images_pixels3.append(np.array([i[year][y][x] for i in Setnodata_datas[:-1]]).astype('float16'))
                images_pixels6.append(Setnodata_datas[-1][year][y][x])
            images_pixels1.append(np.array(images_pixels3).astype('float16'))
            images_pixels5.append(np.array(images_pixels6).astype('float16'))
    print('————————————————————————————————')
    print('———————————Stacking_RR Pool Start—————————————————————')
    name_list = ['RR'] * len(images_pixels1)
    try:
        # pool.restart()
        mean_results = []
        for i in tqdm(zip(images_pixels1, images_pixels5, name_list)):
            results = Sta_R_P(i[0], i[1], i[2])
            mean_results.append(np.array(results).astype('float16'))
        # mean_results = pool.map(Sta_R_P, images_pixels1, images_pixels5, name_list)
        # pool.close()
        # pool.join()
        #####message(nn + '运行成功了hhhhhhh')
    except:
        pass #message(nn + '运行报错了,快来看看吧55555')
    print('————————————————————————————————')
    print('———————————Stacking_RR Pool End—————————————————————')
    mean_results = np.array(mean_results).astype('float16')
    A_WriteArray([mean_results[:, i] for i in range(mean_results.shape[1])], nn, var)
    sg.popup_notify(f'Stacking Task done! Spend-time: {datetime.datetime.now() - start}',
                    display_duration_in_ms=100, fade_in_duration=100)


def Sta_Year(Setnodata_datas, nn):
    print('——————————————Stacking_Year——————————————————')
    start = datetime.datetime.now()
    images_pixels1 = []  # 用于存放所有年，每年五种数据mean的值，一年一个列表
    images_pixels5 = []
    for y in tqdm(range(miny_miny), desc='Stacking_Year'):
        for x in range(minx_minx):
            images_pixels3 = []
            images_pixels6 = []
            for year in range(len(years)):
                images_pixels3.append(np.array([i[year][y][x] for i in Setnodata_datas[:-1]]).astype('float16'))
                images_pixels6.append(Setnodata_datas[-1][year][y][x])
            images_pixels1.append(np.array(images_pixels3).astype('float16'))
            images_pixels5.append(np.array(images_pixels6).astype('float16'))
    print('————————————————————————————————')
    print('———————————Stacking_Year Pool Start—————————————————————')
    name_list = ['Predicted'] * len(images_pixels1)
    try:
        # pool.restart()
        mean_results = []
        for i in tqdm(zip(images_pixels1, images_pixels5, name_list)):
            results = Sta_R_P(i[0], i[1], i[2])
            mean_results.append(np.array(results).astype('float16'))
        # mean_results = pool.map(Sta_R_P, images_pixels1, images_pixels5, name_list)
        # pool.close()
        # pool.join()
        #####message(nn + '运行成功了hhhhhhh')
    except:
        pass #message(nn + '运行报错了,快来看看吧55555')
    print('————————————————————————————————')
    print('———————————Stacking_Year Pool End—————————————————————')
    A_WriteArray(np.array(mean_results).astype('float16').T.tolist(), nn, years)
    sg.popup_notify(f'Stacking_Year Task done! Spend-time: {datetime.datetime.now() - start}',
                    display_duration_in_ms=100, fade_in_duration=100)


def RF_RR(Setnodata_datas, nn):
    print('——————————————RandomForestRegressor RR——————————————————')
    start = datetime.datetime.now()
    images_pixels1 = []  # 用于存放所有年，每年五种数据mean的值，一年一个列表
    images_pixels5 = []
    for y in tqdm(range(miny_miny), desc='RandomForestRegressor_RR'):
        for x in range(minx_minx):
            images_pixels3 = []
            images_pixels6 = []
            for year in range(len(years)):
                images_pixels3.append(np.array([i[year][y][x] for i in Setnodata_datas[:-1]]).astype('float16'))
                images_pixels6.append(Setnodata_datas[-1][year][y][x])
            images_pixels1.append(np.array(images_pixels3).astype('float16'))
            images_pixels5.append(np.array(images_pixels6).astype('float16'))
    print('————————————————————————————————')
    print('———————————RandomForestRegressor_RR Pool Start—————————————————————')
    name_list = ['RR'] * len(images_pixels1)
    try:
        # pool.restart()
        mean_results = []
        for i in tqdm(zip(images_pixels1, images_pixels5, name_list)):
            results = RF_R_P(i[0], i[1], i[2])
            mean_results.append(np.array(results).astype('float16'))
        # mean_results = pool.map(RF_R_P, images_pixels1, images_pixels5, name_list)
        # pool.close()
        # pool.join()
        #####message(nn + '运行成功了hhhhhhh')
    except:
        pass #message(nn + '运行报错了,快来看看吧55555')
    print('————————————————————————————————')
    print('———————————RandomForestRegressor_RR Pool End—————————————————————')
    mean_results = np.array(mean_results).astype('float16')
    A_WriteArray([mean_results[:, i] for i in range(mean_results.shape[1])], nn, var)
    sg.popup_notify(f'RandomForestRegressor Task done! Spend-time: {datetime.datetime.now() - start}',
                    display_duration_in_ms=100, fade_in_duration=100)


def RF_Year(Setnodata_datas, nn):
    print('——————————————RandomForestRegressor_Year——————————————————')
    start = datetime.datetime.now()
    images_pixels1 = []  # 用于存放所有年，每年五种数据mean的值，一年一个列表
    images_pixels5 = []
    for y in tqdm(range(miny_miny), desc='RandomForestRegressor_Year'):
        for x in range(minx_minx):
            images_pixels3 = []
            images_pixels6 = []
            for year in range(len(years)):
                images_pixels3.append(np.array([i[year][y][x] for i in Setnodata_datas[:-1]]).astype('float16'))
                images_pixels6.append(Setnodata_datas[-1][year][y][x])
            images_pixels1.append(np.array(images_pixels3).astype('float16'))
            images_pixels5.append(np.array(images_pixels6).astype('float16'))
    print('————————————————————————————————')
    print('———————————RandomForestRegressor_Year Pool Start—————————————————————')
    name_list = ['Predicted'] * len(images_pixels1)
    try:
        # pool.restart()
        mean_results = []
        for i in tqdm(zip( images_pixels1, images_pixels5, name_list)):
            results = RF_R_P(i[0], i[1], i[2])
            mean_results.append(np.array(results).astype('float16'))
        # mean_results = pool.map(RF_R_P, images_pixels1, images_pixels5, name_list)
        # pool.close()
        # pool.join()
        #####message(nn + '运行成功了hhhhhhh')
    except:
        pass #message(nn + '运行报错了,快来看看吧55555')
    print('————————————————————————————————')
    print('———————————RandomForestRegressor_Year Pool End—————————————————————')
    A_WriteArray(np.array(mean_results).astype('float16').T.tolist(), nn, years)
    sg.popup_notify(f'RandomForestRegressor_Year Task done! Spend-time: {datetime.datetime.now() - start}',
                    display_duration_in_ms=100, fade_in_duration=100)


def Vote_RR(Setnodata_datas, nn):
    print('——————————————VoteRegressor RR——————————————————')
    start = datetime.datetime.now()
    images_pixels1 = []  # 用于存放所有年，每年五种数据mean的值，一年一个列表
    images_pixels5 = []
    for y in tqdm(range(miny_miny), desc='VoteRegressor_RR'):
        for x in range(minx_minx):
            images_pixels3 = []
            images_pixels6 = []
            for year in range(len(years)):
                images_pixels3.append(np.array([i[year][y][x] for i in Setnodata_datas[:-1]]).astype('float16'))
                images_pixels6.append(Setnodata_datas[-1][year][y][x])
            images_pixels1.append(np.array(images_pixels3).astype('float16'))
            images_pixels5.append(np.array(images_pixels6).astype('float16'))
    print('————————————————————————————————')
    print('———————————VoteRegressor_RR Pool Start—————————————————————')
    name_list = ['RR'] * len(images_pixels1)
    try:
        # pool.restart()
        mean_results = []
        for i in tqdm(zip(images_pixels1, images_pixels5, name_list)):
            results = Vote_R_P(i[0], i[1], i[2])
            mean_results.append(np.array(results).astype('float16'))
        # mean_results = pool.map(Vote_R_P, images_pixels1, images_pixels5, name_list)
        # pool.close()
        # pool.join()
        #####message(nn + '运行成功了hhhhhhh')
    except:
        pass #message(nn + '运行报错了,快来看看吧55555')
    print('————————————————————————————————')
    print('———————————VoteRegressor_RR Pool End—————————————————————')
    mean_results = np.array(mean_results).astype('float16')
    A_WriteArray([mean_results[:, i] for i in range(mean_results.shape[1])], nn, var)
    sg.popup_notify(f'VoteRegressor Task done! Spend-time: {datetime.datetime.now() - start}',
                    display_duration_in_ms=1000, fade_in_duration=1000)


def Vote_Year(Setnodata_datas, nn):
    print('——————————————VoteRegressor_Year——————————————————')
    start = datetime.datetime.now()
    images_pixels1 = []  # 用于存放所有年，每年五种数据mean的值，一年一个列表
    images_pixels5 = []
    for y in tqdm(range(miny_miny), desc='VoteRegressor_Year'):
        for x in range(minx_minx):
            images_pixels3 = []
            images_pixels6 = []
            for year in range(len(years)):
                images_pixels3.append(np.array([i[year][y][x] for i in Setnodata_datas[:-1]]).astype('float16'))
                images_pixels6.append(Setnodata_datas[-1][year][y][x])
            images_pixels1.append(np.array(images_pixels3).astype('float16'))
            images_pixels5.append(np.array(images_pixels6).astype('float16'))
    print('————————————————————————————————')
    print('———————————VoteRegressor_Year Pool Start—————————————————————')
    name_list = ['Predicted'] * len(images_pixels1)
    try:
        # pool.restart()
        mean_results = []
        for i in tqdm(zip(images_pixels1, images_pixels5, name_list)):
            results = Vote_R_P(i[0], i[1], i[2])
            mean_results.append(np.array(results).astype('float16'))
        # mean_results = pool.map(RF_R_P, images_pixels1, images_pixels5, name_list)
        # pool.close()
        # pool.join()
        #####message(nn + '运行成功了hhhhhhh')
    except:
        pass #message(nn + '运行报错了,快来看看吧55555')
    print('————————————————————————————————')
    print('———————————VoteRegressor_Year Pool End—————————————————————')
    A_WriteArray(np.array(mean_results).astype('float16').T.tolist(), nn, years)
    sg.popup_notify(f'VoteRegressor_Year Task done! Spend-time: {datetime.datetime.now() - start}',
                    display_duration_in_ms=100, fade_in_duration=100)


def normalization_Writearray_Spatial(Datas):
    '''
    归一化（空间）
    '''
    ds = gdal.Open(Sample_tif)  # 打开文件
    for data, na in zip(Datas, na_me2):
        for da, year in zip(data, years):
            max_value = np.nanmax(da)
            min_value = np.nanmin(da)
            da = (da - min_value) / (max_value - min_value)
            im_width = minx_minx  # 获取栅格矩阵的列数
            im_height = miny_miny  # 获取栅格矩阵的行数
            img_datatype = gdal.GDT_Float32  # 数据类型
            outdir = Outpath + os.sep + 'Normal_Spatial_' + na
            logging.info(f'-------输出文件夹为 {outdir}---------')
            if not os.path.exists(outdir):  # 判断原始文件路劲是否存在,如果不存在就直接退出
                os.makedirs(outdir)
            out_ds = gdal.GetDriverByName('GTiff').Create(
                outdir + os.sep + 'Normal_Spatial_' + na + "_" + str(year) + '.tif',  # tif文件所保存的路径
                im_width,  # 获取栅格矩阵的列数
                im_height,  # 获取栅格矩阵的行数
                ds.RasterCount,  # 获取栅格矩阵的波段数
                img_datatype)  # 获取第一波段的数据类型
            out_ds.SetProjection(ds.GetProjection())  # 投影信息
            out_ds.SetGeoTransform(ds.GetGeoTransform())  # 仿射信息
            out_band = out_ds.GetRasterBand(1)
            out_band.WriteArray(da.reshape(miny_miny, minx_minx).astype('float16'))  # 写入数据 (why)
            out_ds.FlushCache()  # (刷新缓存)
            logging.info(f' {outdir + os.sep + "Normal_Spatial_" + na + "_" + str(year) + ".tif"} is  ok   !!!!!!!!')
    del ds


def normalization_Writearray_Spatial_time(Datas):
    '''
    归一化（空间和时间）
    '''
    ds = gdal.Open(Sample_tif)  # 打开文件
    [MuSyQ_min, GLASS_min, MODIS_min, CASA_min, W_min, LAI_min] = [np.nanmin(i) for i in Datas]
    [MuSyQ_max, GLASS_max, MODIS_max, CASA_max, W_max, LAI_max] = [np.nanmax(i) for i in Datas]
    print('min:', MuSyQ_min, GLASS_min, MODIS_min, CASA_min, W_min, LAI_min)
    print('max:', MuSyQ_max, GLASS_max, MODIS_max, CASA_max, W_max, LAI_max)
    min_max = {'Geodata': {'min': MuSyQ_min, 'max': MuSyQ_max}, 'GLASS': {'min': GLASS_min, 'max': GLASS_max},
               'MODIS': {'min': MODIS_min, 'max': MODIS_max}, 'TPDC': {'min': CASA_min, 'max': CASA_max},
               'W': {'min': W_min, 'max': W_max}, 'LAI': {'min': LAI_min, 'max': LAI_max}}

    for data, na in zip(Datas, na_me2):
        for da, year in zip(data, years):
            max_value = np.nanmax(min_max[na]['max'])
            min_value = np.nanmin(min_max[na]['min'])
            mean_value = np.nanmean(da)
            da = (da - mean_value) / (max_value - min_value)
            im_width = minx_minx  # 获取栅格矩阵的列数
            im_height = miny_miny  # 获取栅格矩阵的行数
            img_datatype = gdal.GDT_Float32  # 数据类型
            outdir = Outpath + os.sep + 'Normal_' + na
            logging.info(f'-------输出文件夹为 {outdir}---------')
            if not os.path.exists(outdir):  # 判断原始文件路劲是否存在,如果不存在就直接退出
                os.makedirs(outdir)
            out_ds = gdal.GetDriverByName('GTiff').Create(
                outdir + os.sep + 'Normal_' + na + "_" + str(year) + '.tif',  # tif文件所保存的路径
                im_width,  # 获取栅格矩阵的列数
                im_height,  # 获取栅格矩阵的行数
                ds.RasterCount,  # 获取栅格矩阵的波段数
                img_datatype)  # 获取第一波段的数据类型
            out_ds.SetProjection(ds.GetProjection())  # 投影信息
            out_ds.SetGeoTransform(ds.GetGeoTransform())  # 仿射信息
            out_band = out_ds.GetRasterBand(1)
            out_band.WriteArray(da.reshape(miny_miny, minx_minx).astype('float16'))  # 写入数据 (why)
            out_ds.FlushCache()  # (刷新缓存)
            logging.info(
                f' {outdir + os.sep + "Normal_SpatialAndTime_" + na + "_" + str(year) + ".tif"} is  ok   !!!!!!!!')
    del ds


if __name__ == "__main__":
    print('-----------------Start----------------------')
    print(f'minx_minx: {minx_minx}')
    print(f'miny_miny: {miny_miny}')

    for year in tqdm(range(styear, edyear + 1), desc='Year'):
        MuSyQ_dir, GLASS_dir = MuSyQ_inpath + os.sep + str(year), GLASS_inpath + os.sep + str(year)
        MODIS_dir = MODIS_path + os.sep + str(year)
        CASA_dir = CASA_path + os.sep + str(year)
        W_dir = W_path + os.sep + str(year)
        LAI_dir = LAI_path + os.sep + str(year)

        MuSyQ_datas.append(
            gdal.Open(g(MuSyQ_dir + os.sep + MuSyQ_key)[0], gdal.GA_ReadOnly).ReadAsArray(0, 0, minx_minx, miny_miny))
        GLASS_datas.append(
            gdal.Open(g(GLASS_dir + os.sep + GLASS_key)[0], gdal.GA_ReadOnly).ReadAsArray(0, 0, minx_minx, miny_miny))
        MODIS_datas.append(
            gdal.Open(g(MODIS_dir + os.sep + MODIS_key)[0], gdal.GA_ReadOnly).ReadAsArray(0, 0, minx_minx, miny_miny))
        CASA_datas.append(
            gdal.Open(g(CASA_dir + os.sep + CASA_key)[0], gdal.GA_ReadOnly).ReadAsArray(0, 0, minx_minx, miny_miny))
        W_datas.append(
            gdal.Open(g(W_dir + os.sep + W_key)[0], gdal.GA_ReadOnly).ReadAsArray(0, 0, minx_minx, miny_miny))
        LAI_datas.append(
            gdal.Open(g(LAI_dir + os.sep + LAI_key)[0], gdal.GA_ReadOnly).ReadAsArray(0, 0, minx_minx, miny_miny))

        # MuSyQ_datas.append(gdal.Open(g(MuSyQ_dir + os.sep + MuSyQ_key)[0],gdal.GA_ReadOnly).ReadAsArray(100, 100, 100, 100))
        # GLASS_datas.append(gdal.Open(g(GLASS_dir + os.sep + GLASS_key)[0],gdal.GA_ReadOnly).ReadAsArray(100, 100, 100, 100))
        # MODIS_datas.append(gdal.Open(g(MODIS_dir + os.sep + MODIS_key)[0],gdal.GA_ReadOnly).ReadAsArray(100, 100, 100, 100))
        # CASA_datas.append(gdal.Open(g(CASA_dir + os.sep + CASA_key)[0],gdal.GA_ReadOnly).ReadAsArray(100, 100, 100, 100))
        # W_datas.append(gdal.Open(g(W_dir + os.sep + W_key)[0],gdal.GA_ReadOnly).ReadAsArray(100, 100, 100, 100))
        # LAI_datas.append(gdal.Open(g(LAI_dir + os.sep + LAI_key)[0],gdal.GA_ReadOnly).ReadAsArray(100, 100, 100, 100))

    MuSyQ_datas = np.array(MuSyQ_datas).astype('float16')
    GLASS_datas = np.array(GLASS_datas).astype('float16')
    MODIS_datas = np.array(MODIS_datas).astype('float16')
    CASA_datas = np.array(CASA_datas).astype('float16')
    W_datas = np.array(W_datas).astype('float16')
    LAI_datas = np.array(LAI_datas).astype('float16')

    # pool = newPool(Pools)
    set = SetNodata([MuSyQ_datas, GLASS_datas, MODIS_datas, CASA_datas, W_datas, LAI_datas], nodatakey)
    nor = normalization(set)
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
    Bagging_RR(nor,'Liner_Bagging')
    # Ada_RR(nor,'Liner_AdaBoost')
    #
    # Gra_RR(nor,'Liner_Gradient')
    #
    # Sta_RR(nor,'Liner_Stacking')
    # RF_RR(nor,'Liner_RandomForestRegressor')
    #
    # Vote_RR(nor,'Liner_Vote')
    # '''再计算每种方法的每年的值（归一化和没有归一化的）'''
    # Mean_Median_Year(nor,'Normal_Mean_Year','Normal_Median_Year')
    # Mean_Median_Year(set,'Mean_Year','Median_Year')
    # Weight_Year(nor,R2_SetNodata([MuSyQ_r2,GLASS_r2,MODIS_r2,CASA_r2,W_r2]),'Normal_Weight_Year')
    # Weight_Year(set,R2_SetNodata([MuSyQ_r2,GLASS_r2,MODIS_r2,CASA_r2,W_r2]),'Weight_Year')
    # Multiply_Regression_Year(nor,'Normal_Multiply_Regression_Year')
    # Multiply_Regression_Year(set,'Multiply_Regression_Year')
    # Bagging_Year(nor, 'Normal_Bagging_Year')
    # Bagging_Year(set, 'Bagging_Year')
    # Ada_Year(nor, 'Normal_AdaBoost_Year')
    # Ada_Year(set, 'AdaBoost_Year')
    #
    # Gra_Year(nor, 'Normal_Gradient_Year')
    # Gra_Year(set, 'Gradient_Year')
    #
    # Sta_Year(nor, 'Normal_Stacking_Year')
    # Sta_Year(set, 'Stacking_Year')
    # RF_Year(nor, 'Normal_RandomForestRegressor_Year')
    # RF_Year(set, 'RandomForestRegressor_Year')
    # Vote_Year(nor, 'Normal_VoteRegressor_Year')
    # Vote_Year(set, 'VoteRegressor_Year')
    # normalization_Writearray_Spatial(set)
    # normalization_Writearray_Spatial_time(set)
    sg.popup_notify(title='Task done!', display_duration_in_ms=1000, fade_in_duration=1000)


