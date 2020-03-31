'''
这个文档有一个初始化简单数据集的函数loadSimpData()，
使用adaBoostTrainDS()函数在训练集上训练出一系列的分类器
当然也可以载入外部数据，只要提供数据和标签即可
'''
import numpy as np
from boost import stumpClassify,buildStump


def loadSimpData():
    '''
    Initialize a simple dataset with 5 data points
    '''
    datMat = np.matrix([[1., 2.1], [2., 1.1], [1.3, 1.], [1., 1.], [2., 1.]])
    classLabels = [1.0, 1.0, -1.0, -1.0, 1.0]
    return datMat, classLabels


def loadDataSet(filename):
    '''
    Load a dataset. The dataset is in a txt file and the last column is label
    '''
    numFeat = len(open(filename).readline().split('\t'))
    dataMat = []
    labelMat = []
    fr = open(filename)
    for line in fr.readlines():
        lineArr = []
        curLine = line.strip().split('\t')
        for i in range(numFeat - 1):
            lineArr.append(float(curLine[i]))
        dataMat.append(lineArr)
        labelMat.append(float(curLine[-1]))
    return dataMat, labelMat


def adaBoostTrainDS(dataArr, classLabels, numIt = 40):
    '''
    Train a series of adaboost weak classifier on a dataset
    
    Input:
        dataArr: data array
        classLabels: data labels
        numIt: iteration number / classifier number
    Output:
        weakClassArr: information(dim, thresh, ineq, alpha) about a 
                      series of trained weak classifier
    '''
    weakClassArr = []
    m = np.shape(dataArr)[0]
    D = np.mat(np.ones((m, 1)) / m) # 样本权重，均匀
    aggClassEst = np.mat(np.zeros((m, 1)))
    for i in range(numIt):
        bestStump, error, classEst = buildStump(dataArr, classLabels, D) # 树桩信息，最小错误率，预测结果
        #print('D:',D.T)
        alpha = float(0.5 * np.log((1.0 - error) / max(error, 1e-16))) # 当前分类器权重，由错误率计算得出
        bestStump['alpha'] = alpha
        weakClassArr.append(bestStump) # 将当前决策树桩记录下来
        #print('classEst:', classEst.T)
        expon = np.multiply(-1 * alpha * np.mat(classLabels).T, classEst) # 每个预测结果与真实标签逐元素相乘
        D = np.multiply(D, np.exp(expon)) # 通过上一棵树的结果计算下一轮每一个样本的权重
        D = D / D.sum() # 样本权重归一化
        aggClassEst += alpha * classEst # 将预测结果加权到总结果（可理解为模型的相加）
        #print('aggClassEst:', aggClassEst.T)
        aggErrors = np.multiply(np.sign(aggClassEst) != np.mat(classLabels).T, np.ones((m, 1))) # 总模型每个样本预测正确与否
        errorRate = aggErrors.sum() / m # 总模型错误率
        print('total error:', errorRate, '\n')
        if errorRate == 0.0:
            break
    return weakClassArr # 返回一个所有树桩的列表
        

def adaClassify(datToClass, classifierArr):
    '''
    classify data with our trained classifiers
    '''
    dataMatrix = np.mat(datToClass)
    m = np.shape(dataMatrix)[0]
    aggClassEst = np.mat(np.zeros((m, 1)))
    for i in range(len(classifierArr)):
        classEst = stumpClassify(dataMatrix, classifierArr[i]['dim'],\
                                 classifierArr[i]['thresh'],\
                                 classifierArr[i]['ineq'])
        aggClassEst += classifierArr[i]['alpha'] * classEst
        #print(aggClassEst)
    return np.sign(aggClassEst)


if __name__ == '__main__':
    datArr, labelArr = loadDataSet('horseColicTraining.txt')
    labelArr = np.array(labelArr)
    labelArr[labelArr == 0.0] = -1
    print('Training start: 使用马疝病数据\n')
    classifierArray = adaBoostTrainDS(datArr, labelArr, 100)
    
    testArr, testLabelArr = loadDataSet('horseColicTest.txt')
    testLabelArr = np.array(testLabelArr)
    testLabelArr[testLabelArr == 0.0] = -1
    print('Test:\n')
    prediction = adaClassify(testArr, classifierArray)
    errArr = np.mat(np.ones((67,1)))
    print('错分样本数：',errArr[prediction != np.mat(testLabelArr).T].sum())
    print('测试错误率：',round(errArr[prediction != np.mat(testLabelArr).T].sum()/67,2))
    
    
    