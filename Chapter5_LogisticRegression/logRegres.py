import math
import numpy as np
import sys, os
import matplotlib.pyplot as plt


def loadDataSet():
    dataMat = []
    labelMat = []
    fr = open('testSet.txt')
    for line in fr.readlines():
        lineArr = line.strip().split()
        dataMat.append([1.0, float(lineArr[0]), float(lineArr[1])])
        labelMat.append(int(lineArr[2]))

    return dataMat, labelMat

def sigmoid(inX):
    res = np.mat([1.0/(1+math.exp(-x)) for x in inX]).transpose()
    return res

def sigmoid0(inX):
    return 1.0/(1+math.exp(-inX))

def gradAscent(dataMatIn, classLabels):
    dataMatrix = np.mat(dataMatIn)
    labelMat = np.mat(classLabels).transpose()  #矩阵转置
    m, n = np.shape(dataMatrix) #m = 100, n = 3
    alpha = 0.001
    maxCycles = 500
    weights = np.ones((n, 1))
    for k in range(maxCycles):
        h = sigmoid(dataMatrix * weights)
        error = (labelMat - h)
        weights = weights + alpha * dataMatrix.transpose() * error
        #weights: 3 * 1
        #dataMatrix: 100 * 3
        #dataMatrix.transpose() : 3*100
        #error: 100*1

    return weights

def stocGradAscent0(dataMatrix, classLabels):
    m, n = np.shape(dataMatrix)
    alpha = 0.01
    weights = np.ones(n)
    for i in range(m):
        h = sigmoid0(sum(dataMatrix[i] * weights))
        error = classLabels[i] - h
        weights = weights + alpha * error *dataMatrix[i]
    return weights

def stocGradAscent1(dataMatrix, classLabels, numIter=150):
    m, n = np.shape(dataMatrix)
    weights = np.ones(n)
    iterTimes = 0
    for j in range(numIter):
        dataIndex = list(range(m))
        for i in range(m):
            alpha = 4/(1.0 + j + i) +0.01
            randIndex = int(np.random.uniform(0, len(dataIndex)))
            h = sigmoid0(sum(dataMatrix[randIndex] * weights))
            error = classLabels[randIndex] - h
            weights = weights + alpha * error * dataMatrix[randIndex]
            del(dataIndex[randIndex])
            iterTimes += 1

    print(iterTimes)
    return weights

def plotBestFit(weights):
    dataMat, labelMat = loadDataSet()
    dataArr = np.array(dataMat)
    n = np.shape(dataArr)[0]
    xcord1 = []
    ycord1 = []
    xcord2 = []
    ycord2 = []

    for i in range(n):
        if (int(labelMat[i]) == 1):
            xcord1.append(dataArr[i, 1])
            ycord1.append(dataArr[i, 2])
        else:
            xcord2.append(dataArr[i, 1])
            ycord2.append(dataArr[i, 2])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(xcord1, ycord1, s=30, c='red', marker='s')
    ax.scatter(xcord2, ycord2, s=30, c='green')
    x = np.arange(-3.0, 3.0, 0.1)
    y = (-weights[0] - weights[1]*x)/weights[2]
    ax.plot(x, y)
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.show()



if __name__ == '__main__':
    dataArr, labelMat = loadDataSet()
    weights = stocGradAscent1(np.array(dataArr), labelMat)
    plotBestFit(weights)  #getA()函数与mat()函数的功能相反，是将一个numpy矩阵转换为数组