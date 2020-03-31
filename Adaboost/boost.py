import numpy as np


def stumpClassify(dataMatrix, dimen, threshVal, threshIneq):
    '''
    Classify the data points based on comparing the threthold
	用训练好的决策树桩来进行分类
    
    Inputs:
        dataMatrix: the input data
        dimen: one data dimension for classification
        threshVal: the classification threshold value
        threIneq: the mode, 'less than' or 'greater than'
    Outputs:
        retArray: the class of every data point
    '''
    retArray = np.ones((np.shape(dataMatrix)[0], 1))
    if threshIneq == 'lt':
        retArray[dataMatrix[:,dimen] <= threshVal] = -1.0
    else:
        retArray[dataMatrix[:,dimen] > threshVal] = -1.0
    return retArray


def buildStump(dataArr, classLabels, D):
    '''
    Find the best decision stump for dataset by trying all inputs for stumpClassify()
	一次分裂，训练决策树桩
    
    Inputs:
        dataArr: the data array
        classLabels: the data labels
        D: the weight vector of data points
    Outputs:
        bestStump: a dict, containing information of the best stump
        minError: achieved minimal error
        bestClasEst: prediction results of the best classifier
    '''
    dataMatrix = np.mat(dataArr)
    labelMat = np.mat(classLabels).T
    m, n = np.shape(dataMatrix)
    numSteps = 10.0
    bestStump = {} # 存储最佳决策树的信息（哪个特征，分裂点如何，大/小于）
    bestClasEst = np.mat(np.zeros((m, 1)))
    minError = float('inf')
    for i in range(n):
        rangeMin = dataMatrix[:,i].min()
        rangeMax = dataMatrix[:,i].max()
        stepSize = (rangeMax - rangeMin) / numSteps # 将[rangeMin, rangeMax]分成numSteps份，搜索分裂点
        for j in range(-1, int(numSteps) + 1):
            for inequal in ['lt', 'gt']: # 对于每一个分裂点，有两种选择，正负分类可互相交换
                threshVal = (rangeMin + float(j) * stepSize) # 分裂点
                predictedVals = stumpClassify(dataMatrix, i, threshVal, inequal) # 当前分裂点的决策树预测结果
                errArr = np.mat(np.ones((m, 1)))
                errArr[predictedVals == labelMat] = 0 # 分类正确的样本对错误率无贡献
                weightedError = D.T * errArr # 加权计算错误率
                
                if weightedError < minError:
                    minError = weightedError
                    bestClasEst = predictedVals.copy()
                    bestStump['dim'] = i
                    bestStump['thresh'] = threshVal
                    bestStump['ineq'] = inequal
    return bestStump, minError, bestClasEst




























