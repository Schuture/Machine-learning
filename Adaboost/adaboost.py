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
    D = np.mat(np.ones((m, 1)) / m)
    aggClassEst = np.mat(np.zeros((m, 1)))
    for i in range(numIt):
        bestStump, error, classEst = buildStump(dataArr, classLabels, D)
        #print('D:',D.T)
        alpha = float(0.5 * np.log((1.0 - error) / max(error, 1e-16)))
        bestStump['alpha'] = alpha
        weakClassArr.append(bestStump)
        #print('classEst:', classEst.T)
        expon = np.multiply(-1 * alpha * np.mat(classLabels).T, classEst)
        D = np.multiply(D, np.exp(expon))
        D = D / D.sum()
        aggClassEst += alpha * classEst
        #print('aggClassEst:', aggClassEst.T)
        aggErrors = np.multiply(np.sign(aggClassEst) != np.mat(classLabels).T, np.ones((m, 1)))
        errorRate = aggErrors.sum() / m
        print('total error:', errorRate, '\n')
        if errorRate == 0.0:
            break
    return weakClassArr
        

def adaClassify(datToClass, classifierArr):
    '''
    classify data with our classifiers
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
    
    
    