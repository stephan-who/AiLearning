print(__doc__)

import operator
from math import log
import decisionTreePlot as dtPlot
from collections import Counter
def createDataSet():
    dataset = [[1, 1, 'yes'],
               [1, 1, 'yes'],
               [1, 0, 'no'],
               [0, 1, 'no'],
               [0, 1, 'no']]
    labels = ['no surfacing ', 'flippers']
    return dataset, labels

def calcShannonEnt(dataset):
    """

    :param dataset:
    :return:
    """
    numEntries = len(dataset)

    labelCounts = {}

    for featVec in dataset:
        currentLabel = featVec[-1]

        if currentLabel not in labelCounts.keys():
            labelCounts[currentLabel] = 0
        labelCounts[currentLabel] += 1

    shannonEnt = 0.0
    for key in labelCounts:
        prob = float(labelCounts[key]) / numEntries
        shannonEnt -= prob * log(prob, 2)

    return shannonEnt

def splitDataSet(dataSet, index, value):
    """

    :param dataset:
    :param index:
    :param value:
    :return:
    """
    retDataSet = []
    for featVec in dataSet:
        if featVec[index] == value:
            reducedFeatVec = featVec[:index]
            reducedFeatVec.extend(featVec[index+1 :])
            retDataSet.append(reducedFeatVec)
    return retDataSet

def chooseBestFeatureToSplit(dataSet):
    """

    :param dataSet:
    :return:
    """
    numFeatures = len(dataSet[0]) - 1
    baseEntropy = calcShannonEnt(dataSet)
    bestInfoGain , bestFeature = 0.0, -1
    #iterate over all the features
    for i in range(numFeatures):
        featList = [example[i] for example in dataSet]
        uniqueVals = set(featList) # 特征取值
        newEntropy = 0.0
        for value in uniqueVals:
            subDataSet = splitDataSet(dataSet, i, value)
            prob = len(subDataSet) / float(len(dataSet))
            newEntropy += prob * calcShannonEnt(subDataSet)
        infoGain = baseEntropy - newEntropy
        if (infoGain > bestInfoGain):
            bestInfoGain = infoGain
            bestFeature = i
    return bestFeature

    # py 3.x
    # base_entropy = calcShannonEnt(dataSet)
    # best_info_gain = 0
    # best_feature = -1
    # for i in range(len(dataSet[0]) - 1):
    #     feature_count = Counter([data[i] for data in dataSet])
    #     #
    #     new_entropy = sum(feature[1] / float(len(dataSet)) * calcShannonEnt(splitDataSet(dataSet, i, feature[0])) \
    #                       for feature in feature_count.items())
    #     info_gain = base_entropy - new_entropy
    #     if info_gain > best_info_gain:
    #         best_info_gain = info_gain
    #         best_feature = i
    #
    # return best_feature

def majorityCnt(classList):
    classCount = {}
    for vote in classList:
        if vote not in classCount.keys():
            classCount[vote] = 0
        classCount[vote] += 1
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]

    # py 3.x
    # major_label = Counter(classList).most_common(1)[0]
    # return major_label

def createTree(dataSet, labels):
    classList = [example[-1] for example in dataSet]
    if classList.count(classList[0]) == len(classList):
        return classList[0]
    if len(dataSet[0]) == 1:
        return majorityCnt(classList) # 遍历所有特征时返回出现次数最多的类别
    bestFeat = chooseBestFeatureToSplit(dataSet)
    bestFeatLabel = labels[bestFeat]
    myTree = {bestFeatLabel:{}}
    del(labels[bestFeat])
    featValues = [example[bestFeat] for example in dataSet]
    uniqueVals = set(featValues)
    for value in uniqueVals:
        subLabels = labels[:]
        myTree[bestFeatLabel][value] = createTree(splitDataSet(dataSet, bestFeat, value), subLabels)

    return myTree

def classify(inputTree, featLabels, testVec):
    firstStr = list(inputTree.keys())[0]
    secondDict = inputTree[firstStr]
    featIndex = featLabels.index(firstStr)

    key = testVec[featIndex]
    valueOfFeat = secondDict[key]
    print('+++')
    if isinstance(valueOfFeat, dict):
        classLabel = classify(valueOfFeat, featLabels, testVec)
    else:
        classLabel = valueOfFeat
    return classLabel

def storeTree(inputTree, filename):
    import pickle
    fw = open(filename, 'wb')
    pickle.dump(inputTree, fw)
    fw.close()
    # with open(filename, 'wb') as fw:
    #     pickle.dump(inputTree, fw)

def grabTree(filename):
    import pickle
    fr = open(filename, 'rb')
    return pickle.load(fr)

def fishTest():
    myDat, labels = createDataSet()

    import copy
    myTree = createTree(myDat, copy.deepcopy(labels))
    print(myTree)
    print(classify(myTree, labels, [1, 1]))
    dtPlot.createPlot(myTree)


def ContactLensesTest():
    fr = open('data/3.DecisionTree/lenses.txt')
    lenses = [inst.strip().split('\t') for inst in fr.readlines()]
    lensesLabels = ['age', 'prescript', 'astigmatic', 'tearRate']
    lensesTree = createTree(lenses, lensesLabels)
    print(lensesTree)
    dtPlot.createPlot(lensesTree)

ContactLensesTest()
