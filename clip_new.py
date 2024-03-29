# -- coding: utf-8 --
import os
import gdal
import numpy as np

# print(np.__path__)

#  读取tif数据集
def readTif(fileName):
    dataset = gdal.Open(fileName)
    if dataset == None:
        pass
    return dataset


#  保存tif文件函数
def writeTiff(im_data, im_geotrans, im_proj, path):
    #指定数据类型
    # if 'int8' in im_data.dtype.name: #输入的im_data是一个numpy。array类型，dtype的名字来对应gdal的datatype类型
    #     datatype = gdal.GDT_Byte
    # elif 'int16' in im_data.dtype.name:
    #     datatype = gdal.GDT_UInt16
    # else:
    im_data[im_data<=-1000] = np.nan
    im_data[np.isnan(im_data)] = -9999

    datatype = gdal.GDT_Float32
    if len(im_data.shape) == 3: #形状 gadal读取出来的数据默认按波段、高、宽的形状排列[[每个波段对应的图像位置和值][][]]
        im_bands, im_height, im_width = im_data.shape
    elif len(im_data.shape) == 2: #如果只有二维意味着灰度，即[]，此时在外面再加一维表示波段即[[]]，类型为array，才可以识别为图片数组
        print("单波段")
        im_data = np.array([im_data]) #再加一维是可以直接实现的
        im_bands, im_height, im_width = im_data.shape
        print("im_bands：",im_bands)
        print("im_height：", im_height)
        print("im_width：", im_width)
    # 创建文件
    driver = gdal.GetDriverByName("GTiff") #创建一个带有名字指代驱动器对象
    dataset = driver.Create(path, int(im_width), int(im_height), int(im_bands), datatype) #给出文件的路径，宽度，高度，波段数（倒过来）和数据类型，创建空文件，并确定开辟多大内存，像素值的类型用gdal类型
    #设置头文件信息：仿射变化参数，投影信息，数据体
    if (dataset != None):
        dataset.SetProjection(im_proj)  # SetProjection写入投影im_proj
        dataset.SetGeoTransform(im_geotrans)  # driver对象创建的dataset对象具有SetGeoTransform方法，写入仿射变换参数GEOTRANS
    for i in range(im_bands): #对每个波段进行处理 写入数据体
        dataset.GetRasterBand(i + 1).WriteArray(im_data[i]) #i从0开始，因此GetRasterBand(i+1),每个波段里用writearray写入图像信息数组im_data
        dataset.GetRasterBand(i + 1).SetNoDataValue(-9999)
    del dataset #写完之后释放内存空间


'''
滑动窗口裁剪函数
TifPath 影像路径
SavePath 裁剪后保存目录
CropSize 裁剪尺寸
RepetitionRate 重复率
'''


def TifCrop(TifPath, SavePath, CropSize, RepetitionRate): #tif文件裁剪函数，输入tif文件路径，保存路径，裁剪尺寸和重复率
    dataset_img = readTif(TifPath) #首先读取文件，采用gdal.open的方法
    width = dataset_img.RasterXSize #宽度是x方向上的size
    height = dataset_img.RasterYSize #高度是y方向上的size
    proj = dataset_img.GetProjection() #得到数据集的投影信息
    geotrans = dataset_img.GetGeoTransform() #得到数据集的地理仿射信息,是一个包含六个元素的元组

    img = dataset_img.ReadAsArray(0,0, width, height)  # 获取数据 Readasarray支持按块读取栅格，可以针对dataset也可以针对波段band,xoff = 0，yoff= 0xoff与yoff指定想要读取的部分原点位置在整张图像中距离全图的原点位置,这样就是读取全部数组信息了,GDAL读取的原点为左上角位置

    #  获取当前文件夹的文件个数len,并以len+1命名即将裁剪得到的图像
    new_name = len(os.listdir(SavePath)) + 1 #os.listdir(path)将path里面的文件夹列出来，os是operating system

    #  裁剪图片,重复率为RepetitionRate
    for i in range(int((height - CropSize * RepetitionRate) / (CropSize * (1 - RepetitionRate)))): #有重复率的裁减时，这个公式判断按照总高度height一共裁剪出多少幅图像 int函数将结果向下取整
        for j in range(int((width - CropSize * RepetitionRate) / (CropSize * (1 - RepetitionRate)))): #有重复率的裁剪时，按宽度一共裁剪出多少幅图像
            #  如果图像是单波段
            if (len(img.shape) == 2):
                cropped = img[
                          int(i * CropSize * (1 - RepetitionRate)): int(i * CropSize * (1 - RepetitionRate)) + CropSize,
                          int(j * CropSize * (1 - RepetitionRate)): int(j * CropSize * (1 - RepetitionRate)) + CropSize] #img[x:x+cropsize, y:y+cropsize] 多维数组的裁剪可以直接这么切片，
            # 如果图像是多波段
            else:
                cropped = img[:,
                          int(i * CropSize * (1 - RepetitionRate)): int(i * CropSize * (1 - RepetitionRate)) + CropSize,
                          int(j * CropSize * (1 - RepetitionRate)): int(j * CropSize * (1 - RepetitionRate)) + CropSize] #多波段，[波段，高，宽]的形状shape
            # 更新裁剪后的位置信息
            local_geotrans = list(geotrans) #每一张裁剪图的本地放射变化参数，0，3代表左上角坐标
            local_geotrans[0] = geotrans[0] + int(j * CropSize * (1 - RepetitionRate)) * geotrans[1]#分别更新为裁剪后的每一张局部图的左上角坐标，为滑动过的像素数量乘以分辨率
            local_geotrans[3] = geotrans[3] + int(i * CropSize * (1 - RepetitionRate)) * geotrans[5]
            local_geotrans = tuple(local_geotrans)
            # 写图像
            writeTiff(cropped, local_geotrans, proj, SavePath + "/%d.tif" % new_name) #数组、仿射变化参数、投影、保存路径
            #  文件名 + 1
            new_name = new_name + 1 #梅循环一次文件名的尾数+1

    # 向前裁剪最后一列
    for i in range(int((height - CropSize * RepetitionRate) / (CropSize * (1 - RepetitionRate)))):
        if (len(img.shape) == 2):
            cropped = img[int(i * CropSize * (1 - RepetitionRate)): int(i * CropSize * (1 - RepetitionRate)) + CropSize,
                      (width - CropSize): width] #刚好一个cropsize的宽，也就是最后一列跟前面重复率可能为repetition~1
        else:
            cropped = img[:,
                      int(i * CropSize * (1 - RepetitionRate)): int(i * CropSize * (1 - RepetitionRate)) + CropSize,
                      (width - CropSize): width]
        # 更新裁剪后的位置信息
        local_geotrans = list(geotrans)  # 每一张裁剪图的本地放射变化参数，0，3代表左上角坐标
        local_geotrans[0] = geotrans[0] + (width - CropSize) * geotrans[1]  # 分别更新为裁剪后的每一张局部图的左上角坐标
        local_geotrans[3] = geotrans[3] + int(i * CropSize * (1 - RepetitionRate)) * geotrans[5]
        local_geotrans = tuple(local_geotrans)
        # 写图像
        writeTiff(cropped, local_geotrans, proj, SavePath + "/%d.tif" % new_name)
        new_name = new_name + 1

    # 向前裁剪最后一行
    for j in range(int((width - CropSize * RepetitionRate) / (CropSize * (1 - RepetitionRate)))):
        if (len(img.shape) == 2):
            cropped = img[(height - CropSize): height,
                      int(j * CropSize * (1 - RepetitionRate)): int(j * CropSize * (1 - RepetitionRate)) + CropSize]#刚好一个cropsize的高，也就是最后一列跟前面重复率可能为repetition~1
        else:
            cropped = img[:,
                      (height - CropSize): height,
                      int(j * CropSize * (1 - RepetitionRate)): int(j * CropSize * (1 - RepetitionRate)) + CropSize]
        # 更新裁剪后的位置信息
        local_geotrans = list(geotrans)  # 每一张裁剪图的本地放射变化参数，0，3代表左上角坐标
        local_geotrans[0] = geotrans[0] + int(j * CropSize * (1 - RepetitionRate)) * geotrans[1]  # 分别更新为裁剪后的每一张局部图的左上角坐标
        local_geotrans[3] = geotrans[3] + (height - CropSize) * geotrans[5]
        local_geotrans = tuple(local_geotrans)
        #写文件
        writeTiff(cropped, local_geotrans, proj, SavePath + "/%d.tif" % new_name)
        #  文件名 + 1
        new_name = new_name + 1

    # 裁剪右下角
    if (len(img.shape) == 2):
        cropped = img[(height - CropSize): height,
                  (width - CropSize): width] #保证右下角也完全裁剪到，因为这种滑动窗口类似有向裁剪，在终点处的部分不一定能完全被裁剪到，所以右下角是各个位置为起点的终点，需要单独考虑
    else:
        cropped = img[:,
                  (height - CropSize): height,
                  (width - CropSize): width]
    # 更新裁剪后的位置信息
    local_geotrans = list(geotrans)  # 每一张裁剪图的本地放射变化参数，0，3代表左上角坐标
    local_geotrans[0] = geotrans[0] +  (width - CropSize) * geotrans[1]  # 分别更新为裁剪后的每一张局部图的左上角坐标
    local_geotrans[3] = geotrans[3] + (height - CropSize) * geotrans[5]
    local_geotrans = tuple(local_geotrans)
    writeTiff(cropped, local_geotrans, proj, SavePath + "/%d.tif" % new_name)
    new_name = new_name + 1

#  将影像1裁剪为重复率为0.1的256×256的数据集

Sample_tif = r'E:\Integrated_analysis_data\Data\Sample\Mask_Mul_1989.tif'
out_dir = r'E:\Integrated_analysis_data\Data\clip_temp'
TifCrop(Sample_tif, out_dir, 500, 0.1)
