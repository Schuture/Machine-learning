from math import log
import numpy as np
import collections
import operator
import pickle

# sum of -p(X)logp(X)
def calcShannonEnt(dataSet):
    '''
    Input: a MxN array with label at the last column
    Output: Shannon entropy of dataset
    '''
    numEntries = len(dataSet)
    labelCounts = dict(collections.Counter(dataSet[:,-1]))
    shannonEnt = 0.0
    for key in labelCounts.keys():
        prob = float(labelCounts[key])/numEntries
        shannonEnt -= prob * log(prob, 2)
    return shannonEnt


def splitDataSet(dataSet, axis, value):
    '''
    Input:  a MxN array with label at the last column
            the 'axis'-th feature
            which value of the feature to split at
    Output: splitted dataset without this feat and the axis-th feature = value
    '''
    valid = dataSet[np.where(dataSet[:,axis] == value)]
    # don't consider 'axis' column anymore
    return np.hstack((valid[:, :axis], valid[:, axis+1:]))


def chooseBestFeatureToSplit(dataSet):
    '''
    Input:  a MxN array with label at the last column
    Output: the feature that can maximize infomation gain, 0,1,...
    '''
    numFeatures = len(dataSet[0]) - 1
    baseEntropy = calcShannonEnt(dataSet)
    bestInfoGain = 0.0
    bestFeature = -1
    # calculate each feature for its max entropy
    for i in range(numFeatures): 
        featList = dataSet[:,i].reshape(-1).tolist()
        uniqueVals = set(featList)
        newEntropy = 0.0
        for value in uniqueVals:
            # split dataset for each value in a feature
            subDataSet = splitDataSet(dataSet, i, value)  
            prob = len(subDataSet)/float(len(dataSet))
            newEntropy += prob * calcShannonEnt(subDataSet)
        infoGain = baseEntropy - newEntropy
        if infoGain > bestInfoGain:
            bestInfoGain = infoGain
            bestFeature = i
    return bestFeature


def majorityCnt(classList):
    '''
    Input:  a list of class
    Output: the most frequent class
    '''
    classCount = dict(collections.Counter(classList))
    sortedClassCount = sorted(classCount.items(), 
                              key = operator.itemgetter(1), reverse = True)
    return sortedClassCount[0][0]


def createTree(dataSet, labels):
    '''
    Input:  'dataSet' is a MxN array with label at the last column
            'labels' is the labels for all features (columns)
    Output: the tree denoted by a dict
    '''
    classList = dataSet[:,-1].reshape(-1).tolist()
    # all samples have the same class
    if classList.count(classList[0]) == len(classList):
        return classList[0]
    # all features have been traversed
    if len(dataSet[0]) == 1:
        return majorityCnt(classList)
    bestFeat = chooseBestFeatureToSplit(dataSet)
    bestFeatLabel = labels[bestFeat]
    myTree = {bestFeatLabel : {}}
    del labels[bestFeat]
    featValues = dataSet[:, bestFeat] # best feature column
    uniqueVals = set(featValues)
    for value in uniqueVals:
        subLabels = labels[:]
        myTree[bestFeatLabel][value] = createTree(splitDataSet\
              (dataSet, bestFeat, value), subLabels)
    return myTree
    
    
def classify(inputTree, featLabels, testVec):
    '''
    Input:  a tree model, dict
            feature labels in a list
            test vector
    Output: classification result
    '''
    firstStr = list(inputTree.keys())[0]
    secondDict = inputTree[firstStr]
    featIndex = featLabels.index(firstStr)
    for key in secondDict.keys():
        if testVec[featIndex] == key:
            if type(secondDict[key]).__name__ == 'dict':
                classLabel = classify(secondDict[key], featLabels, testVec)
            else:
                classLabel = secondDict[key]
    return classLabel


def storeTree(inputTree, filename):
    ''' store a dict tree in a file'''
    with open(filename,'w') as fw:
        pickle.dump(inputTree, fw)
        
        
def grabTree(filename):
    ''' get a tree from a file '''
    fr = open(filename)
    return pickle.load(fr)


def main():
    myTree = {'no surfacing': {0: 'no', 1:{ 'flippers': {0: "no", 1: 'yes'}}}}
    labels = ['no surfacing', 'flippers']
    print(classify(myTree, labels, [1,1]))


if __name__ == '__main__':
    main()












