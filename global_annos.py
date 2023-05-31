## 公共区域函数
import pandas as pd
import numpy as np
from global_ import new_bbox_annos_path


def annos():  # 收集有结节图的名字
    annos = pd.read_excel(new_bbox_annos_path)  # 读取bbox_annos.xls
    annos = np.array(annos)  # 读取为数组
    annos = annos.tolist()  # 变成列表便于操作
    a = []
    for k in annos:  # 逐行读取
        print(k)
        if len(k) == 3:  # 由于版本不同，有的有头标，有的没有
            jiejie = 2  # 有头标
        else:
            jiejie = 1  # 没头标
        if k[jiejie] != "[]":  # 结节部分不为空的话
            a.append(k[jiejie - 1])  # 添加 有结节图的名字到 a
    return a  # 返回所有 有结节的图名


annos = annos()


# 在调用这个函数之前，结节信息已经被转换为字符串类型，并且两端有一些额外的引号和中括号，因此需要使用一些字符串方法进行清洗和处理，将其转换为二维列表形式
def str_to_int(aaa):  # str → list
    if aaa == "[]":
        b = []
    else:
        aaa = aaa.lstrip("'[[")  # lstrip()方法:去掉了字符串开头处的'[[的这三个字符
        aaa = aaa.rstrip("]]'")  # rstrip()方法:去掉了字符串末尾处的]]'的这三个字符
        b = aaa.split("], [")  # 用split()方法:按照 ], [ 这个字符串作为分隔符进行分割
        for i in range(len(b)):
            b[i] = b[i].split(",")  # 用 split() 方法将该元素拆分为一个由四个字符串元素组成的列表
            b[i][0] = int(float(b[i][0]))
            b[i][1] = int(float(b[i][1]))  # 对每个字符串元素分别使用 float() 函数将其转换为浮点数类型
            b[i][2] = int(float(b[i][2]))  # 再使用int() 函数将其转换为整数类型，最后将这四个整数作为一个列表元素重新组合
            b[i][3] = int(float(b[i][3]))
    return b  # 函数返回转换后的二维列表。


def annos_list():  # 收集有结节图的名字
    annos = pd.read_excel(new_bbox_annos_path)  # 读取bbox_annos.xls
    annos = np.array(annos)  # 读取为数组
    annos = annos.tolist()  # 变成列表便于操作
    a = []
    for k in annos:  # 逐行读取（三个的-名字再第二位/俩个的名字再第一位）【3：标签-名字-肺结节信息】【2：名字-肺结节信息】
        if len(k) == 3:  # 由于版本不同，有的有头标，有的没有
            jiejie = 2  # 有头标
        else:
            jiejie = 1  # 没头标
        if k[jiejie] != "[]":  # 结节部分不为空的话
            a.append([k[jiejie - 1], str_to_int(k[jiejie])])  # 添加 有结节图的名字到 a excel表中（肺结节信息位置-1=肺结节的名字位置)
    return a  # 返回所有 有结节的图名


print(annos_list())
annos_list = annos_list()

##读取bbox_annos.xls文件（之前数据集CT图像一系列的数据出来然后保存的文件），选取有肺结节的图，并且得到图名列）
