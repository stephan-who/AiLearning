__coding__ = 'utf-8'
__author__ = "Ng WaiMing"

from numpy import *
from time import sleep
import matplotlib
from matplotlib import pyplot as plt

def loadDataSet(fileName):
    dataSet = []
    fr = open(fileName)
    for line in fr.readlines():
        curLine = line.strip().split('\t')
        fltLine = list(map(float,curLine))
        # fltLine = [float(x) for x in curLine]
        dataSet.append(fltLine)

    return dataSet

def distEclud(vecA, vecB):
    return sqrt(sum(power(vecA - vecB, 2)))


def randCent(dataMat, k):
    """
    构建一个包含K个随机质心的集合
    :param dataMat:
    :param k:
    :return:
    """
    m, n = shape(dataMat)
    centroids = mat(zeros((k, n))) # 1行是1个质心
    for j in range(n):
        minJ = min(dataMat[:, j])
        rangeJ = float(max(dataMat[:,j]) - minJ)
        centroids[:, j] = mat(minJ + rangeJ * random.rand(k, 1))
    return centroids

def kMeans(dataMat, k, distMeas=distEclud, createCent=randCent):
    m, n = shape(dataMat)
    # clusterAssment包含两个列:一列记录簇索引值,第二列存储误差
    clusterAssment = mat(zeros((m, 2)))
    centroids = createCent(dataMat, k)
    clusterChanged = True
    while clusterChanged:
        clusterChanged = False
        for i in range(m):
            minDist = inf
            minIndex = -1
            for j in range(k):
                distJI = distMeas(centroids[j,:], dataMat[i, :])
                if distJI < minDist:
                    minDist = distJI
                    minIndex = j
            if clusterAssment[i, 0] != minIndex:
                clusterChanged = True
            clusterAssment[i, :] = minIndex, minDist ** 2
        # 遍历所有质心并更新它们的取值
        for cent in range(k):
            # 通过数据过滤来获取给定簇的所有点
            ptsInClust = dataMat[nonzero(clusterAssment[:, 0].A == cent)[0]]
            # axis=0表示沿矩阵的列方向进行均值计算
            centroids[cent, :] = mean(ptsInClust, axis=0)
    print(centroids)
    # 返回所有的类质心与点的分配结果
    return centroids, clusterAssment

def biKmeans(dataMat, k, distMeas=distEclud):
    """

    :param dataMat:
    :param k:
    :param distMeas:
    :return:
    """
    m, n = shape(dataMat)
    # 创建一个矩阵来存储数据集中每个点的簇分配结果及平方误差
    clusterAssment = mat(zeros((m, 2)))
    centroid0 = mean(dataMat, axis=0).tolist()[0]
    centList = [centroid0]

    for j in range(m):
        clusterAssment[j, 1] = distMeas(mat(centroid0), dataMat[j, :]) ** 2

    while(len(centList) < k):
        lowestSSE = inf
        # 通过考察簇列表中的值来获得当前簇的树木，遍历所有的簇来决定最佳的簇进行划分
        for i in range(len(centList)):
            ptsInCurrCluster = dataMat[nonzero(clusterAssment[:, 0].A == i)[0], :]
            centroidMat, splitClustAss = kMeans(ptsInCurrCluster, 2, distMeas)
            #将误差值与剩余数据集的误差之和作为本次划分的误差
            sseSplit = sum(splitClustAss[:, 1])
            sseNotSplit = sum(clusterAssment[nonzero(clusterAssment[:, 0].A != i)[0], 1])
            print('sseSplit, and not Split:' ,sseSplit, sseNotSplit)
            if (sseSplit + sseNotSplit) < lowestSSE:
                bestCentToSplit = i
                bestNewCents = centroidMat
                bestClustAss = splitClustAss.copy()
                lowestSSE = sseSplit + sseNotSplit
        #找出最好的簇分配结果
        """
        当k=2是k-means返回的是类别0，1两类，
        因此这里讲类别为1的更改为其质心的长度(即新的簇类别序号)，而类别为0的返回的是该簇原先的类别。
         举个例子：
        例如：目前划分成了0，1
        两个簇，而要求划分成3个簇，则在算法进行时，假设对1进行划分得到的SSE最小，则将1划分成了2个簇，其返回值为0，1
        两个簇，将返回为1的簇改成2，返回为0的簇改成1，因此现在就有0，1，2
        三个簇了。
        """
        bestClustAss[nonzero(bestClustAss[:, 0].A ==1)[0], 0] = len(centList)
        bestClustAss[nonzero(bestClustAss[:, 0].A ==0)[0], 0] = bestCentToSplit
        print(bestClustAss)
        print('the best Cent to Split:', bestCentToSplit)
        print('the len of bestClustAss is :', len(bestClustAss))
        centList[bestCentToSplit] = bestNewCents[0, :].tolist()[0]
        centList.append(bestNewCents[1, :].tolist()[0])
        clusterAssment[nonzero(clusterAssment[:, 0].A == bestCentToSplit)[0], :] = bestClustAss
    return mat(centList), clusterAssment

import urllib.parse, urllib.request
import json
def geoGrab(stAddress, city):
    apiStem = 'http://where.yahooapis.com/geocode?'
    params = {}
    params['flags'] = 'J'
    params['appid'] = 'ppp69N8t'
    params['location'] = '%s %s' % (stAddress, city)
    url_params = urllib.parse.urlencode(params)
    yahooApi = apiStem + url_params
    print(yahooApi)
    c = urllib.request.urlopen(yahooApi)
    return json.loads(c.read())

from time import sleep
def massPlaceFind(fileName):
    fw = open('places.txt', 'w')
    for line in open(fileName).readlines():
        line = line.strip()
        lineArr = line.split('\t')
        retDict = geoGrab(lineArr[1], lineArr[2])
        if retDict['ResultSet']['Error'] == 0:
            lat = float(retDict['ResultSet']['Results'][0]['latitude'])
            lng = float(retDict['ResultSet']['Results'][0]['longitude'])
            print("%s\t%f\t%f" % (lineArr[0], lat, lng))
            fw.write('%s\t%f\t%f\n' % (line, lat, lng))
        else:
            print("error fetching")
        sleep(1)
    fw.close()

def distSLC(vecA, vecB):
    """
    球面余弦定理
    :param vecA:
    :param vecB:
    :return:
    """
    # 经度和维度用角度作为单位，但是sin()和cos()以弧度为输入
    # 可以将角度除以180度然后再乘以圆周率pi转换为弧度
    a = sin(vecA[0, 1] * pi / 180) * sin(vecB[0, 1] * pi /180)
    b = cos(vecA[0, 1] * pi / 180) * cos(vecB[0, 1] * pi /180) * \
        cos(pi * (vecB[0, 0] - vecA[0, 0]) / 180)
    return arccos(a + b) * 6371.0

def clusterClubs(fileName, imgName, numClust=5):
    datList = []
    for line in open(fileName).readlines():
        lineArr = line.split('\t')
        datList.append([float(lineArr[4]), float(lineArr[3])])
    datMat = mat(datList)
    myCentroids, clustAssing = biKmeans(datMat, numClust, distMeas=distSLC)
    fig = plt.figure()
    rect = [0.1, 0.1, 0.8, 0.8]
    scatterMarkers = ['s', 'o', '^', '8', 'p', 'd', 'v', 'h', '>', '<']
    axprops = dict(xticks=[], yticks=[])
    ax0 = fig.add_axes(rect, label='ax0', **axprops)
    imgP = plt.imread(imgName)
    ax0.imshow(imgP)
    ax1 = fig.add_axes(rect, label='ax1', frameon=False)
    for i in range(numClust):
        ptsInCurrCluster = datMat[nonzero(clustAssing[:, 0].A == i)[0], :]
        markerStyle = scatterMarkers[i % len(scatterMarkers)]
        ax1.scatter(ptsInCurrCluster[:, 0].flatten().A[0], ptsInCurrCluster[:, 1].flatten().A[0], marker=markerStyle, s=90)
    ax1.scatter(myCentroids[:, 0].flatten().A[0], myCentroids[:, 1].flatten().A[0], marker='+', s=300)
    plt.show()

if __name__ == '__main__':
# Test for codes
#     dataMat = mat(loadDataSet('testSet.txt'))
#     print(randCent(dataMat, 2))
#     print(distEclud(dataMat[0], dataMat[1]))
#     myCentroids, clustAssing = kMeans(dataMat, 4)
#
#     dataMat3 = mat(loadDataSet('testSet2.txt'))
#     centList, myNewAssments = biKmeans(dataMat3, 3)
#     # print('centList:', centList)
#
#     print('New Assments:\n', myNewAssments)

    clusterClubs('places.txt', 'Portland.png')
# KMeans sklearn
# import numpy as np
# from sklearn.cluster import KMeans
# dataMat = []
# fr = open("testSet.txt")
# for line in fr.readlines():
#     curLine = line.strip().split('\t')
#     fltLine = list(map(float, curLine))
#     dataMat.append(fltLine)
#
# km = KMeans(n_clusters=5)
# km.fit(dataMat)
# km_pred = km.predict(dataMat)
# centers = km.cluster_centers_
# print(centers)
# plt.scatter(np.array(dataMat)[:, 1], np.array(dataMat)[:, 0], c=km_pred)
# plt.scatter(centers[:, 1], centers[:, 0], c="r")
# plt.show()