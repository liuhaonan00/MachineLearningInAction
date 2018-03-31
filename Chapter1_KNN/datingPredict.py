#!/usr/bin/python3
# -*- coding:utf-8 -*-

from numpy import *
import operator
import sys
import matplotlib
import matplotlib.pyplot as plt


def createDataSet():
    group = array([[1.0, 1.1], [1.0, 1.0], [0, 0], [0, 0.1]])
    labels = ['A', 'A', 'B', 'B']
    return group, labels

def classify0(inX, dataSet, labels, k):
    dataSetSize = dataSet.shape[0]
    # print(dataSet.shape)
    diffMat = tile(inX, (dataSetSize, 1)) - dataSet

    sqDiffMat = diffMat**2

    sqDistances = sqDiffMat.sum(axis=1)

    distances = sqDistances**0.5
    sortedDistIndicies = distances.argsort()
    # print(distances)
    # print(sortedDistIndicies)
    classCount = {}

    for i in range(k):
        voteIlabel = labels[sortedDistIndicies[i]]
        classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1
    # print(classCount.items())
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    # print(sortedClassCount)
    return sortedClassCount[0][0]

def file2matrix(filename):
    fr = open(filename)
    arrayOfLines = fr.readlines()  #redlines 一次把所有行都读进数组,
    numberOfLines = len(arrayOfLines)
    random.shuffle(arrayOfLines) #打乱每一行的顺序

    returnMat = zeros((numberOfLines, 3))
    classLabelVector = []
    index = 0
    for line in arrayOfLines:
        line = line.strip()
        listFromLine = line.split('\t')
        returnMat[index, :] = listFromLine[0:3]
        classLabelVector.append(listFromLine[-1])
        index += 1

    return returnMat, classLabelVector

def autoNorm(dataSet):
    minVals = dataSet.min(0)
    maxVals = dataSet.max(0)
    ranges = maxVals - minVals
    normDataSet = zeros(shape(dataSet))
    m = dataSet.shape[0]
    normDataSet = dataSet - tile(minVals, (m, 1))
    normDataSet = normDataSet/tile(ranges, (m, 1))

    return normDataSet, ranges, minVals

def datingClassTest():
    path = "datingTestSet2.txt"
    hoRatio = 0.10 #样本比例抽样比
    datingDataMat, datingLabels = file2matrix(path)
    normMat, ranges, minVals = autoNorm(datingDataMat)

    m = normMat.shape[0]   #m 为样本总容量
    numTestVecs = int(m*hoRatio)
    errorCount = 0.0

    for i in range(numTestVecs):
        # normMat[i, :] 当前点的坐标
        # normMat[numTestVecs:m, :]、datingLabels[numTestVecs:m] 剩余点的坐标和label
        classifierResult = classify0(normMat[i, :], normMat[numTestVecs:m, :], datingLabels[numTestVecs:m], 3)
        print("the classifier came back with: %d, the real answer is: %d" %(int(classifierResult), int(datingLabels[i])))

        if (classifierResult != datingLabels[i]):
            errorCount += 1.0

    print("the total error rate is: %f" %(errorCount/float(numTestVecs)))


def classifyPerson():
    path = "datingTestSet.txt"

    result_list = ['not at all', 'in small doses', 'in large doses']
    print("Percentage of time spent playing video games?")
    percentTats = float(input())

    print("Frequent flier miles earned per year?")
    ffMiles = float(input())

    print("Liters of ice cream consumed per year?")
    iceCream = float(input())

    datingDataMat, datingLabels = file2matrix(path)
    normMat, ranges, minVals = autoNorm(datingDataMat)

    inArr = array([ffMiles, percentTats, iceCream])
    classifierResult = classify0((inArr-minVals)/ranges, normMat, datingLabels,3)

    print("You will probably like this person:", classifierResult)


if __name__ == '__main__':
    datingClassTest()
