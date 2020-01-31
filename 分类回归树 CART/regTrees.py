'''
functions:
    loadDataSet:        dataloader
    binSplitDataSet:    split a dataset into two
    regLeaf:            calculate leaf value
    regErr:             caculate regression error
    chooseBestSplit:    choose the best feature and value to split tree
    createTree:         create a tree based on a dataset
    
    isTree:             judge whether an object is a dict
    getMean:            return the mean value of a tree
    prune:              pruning
    
    linearSolve:        use data at the leaf to do linear regression
    modelLeaf:          get weight of linear regression
    modelErr:           get error of linear regression
    regTreeEval:        return the value of leaf
    modelTreeEval:      calculate the prediction using data and model
    treeForeCast:       traverse the tree until find a leaf
    createForeCast:     return prediction of data using the model on a leaf
    
    predict:            prediction function of classification
    classification:     demo of classification
    regression:         demo of regression
'''

import numpy as np
import matplotlib.pyplot as plt

def loadDataSet(fileName):
    '''
    convert each data line into floating point and create a data matrix
    
    Input:  file segmented by tab
    Output: data matrix
    '''
    dataMat = []
    fr = open(fileName)
    for line in fr.readlines():
        curLine = line.strip().split('\t')
        fltLine = map(float, curLine)
        dataMat.append(fltLine)
    return dataMat


def binSplitDataSet(dataSet, feature, value):
    '''
    split a dataset to two subsets w.r.t. a feature and return subsets
    
    Input:  dataSet
            feature to split
            what value of the feature to split
    Output: two subsets of the dataset
    '''
    mat0 = dataSet[np.nonzero(dataSet[:,feature] > value)[0],:]
    mat1 = dataSet[np.nonzero(dataSet[:,feature] <= value)[0],:]
    return mat0, mat1


def regLeaf(dataSet):
    '''
    calculate the leaf value
    '''
    return np.mean(dataSet[:,-1])


def regErr(dataSet):
    '''
    the sum of (Xi-mean)^2 denotes regression error
    '''
    return np.var(dataSet[:,-1]) * np.shape(dataSet)[0]


def chooseBestSplit(dataSet, leafType = regLeaf, errType = regErr, ops = (1,4)):
    '''
    find best binary splitting way
    
    Input:  numpy array dataset with label at the last column
            leafType is the function to build leaf node
            errType is the function to calculate error
            ops is other parameters for preprune a tree
    '''
    tolS = ops[0] # allowed error decreasing quantity
    tolN = ops[1] # least sample size to split
    if len(set(dataSet[:-1].T.tolist()[0])) == 1:  # only one class in leaf
        return None, leafType(dataSet)
    m, n = np.shape(dataSet)
    S = errType(dataSet)
    bestS = float('Inf')
    bestIndex = 0
    bestValue = 0
    for featIndex in range(n-1): # for each feature, the last column is label
        for splitVal in set(list(dataSet[:,featIndex])): # for each feature value
            mat0, mat1 = binSplitDataSet(dataSet, featIndex, splitVal)
            if np.shape(mat0)[0] < tolN or np.shape(mat1)[0] < tolN: # too few samples
                continue
            newS = errType(mat0) + errType(mat1)
            # error decreases, choose this splitting as the best
            if newS < bestS: 
                bestIndex = featIndex
                bestValue = splitVal
                bestS = newS
    # we can't get splitting way that is good enough
    if S - bestS < tolS: 
        return None, leafType(dataSet)
    mat0, mat1 = binSplitDataSet(dataSet, bestIndex, bestValue)
    # too few samples for one subset
    if np.shape(mat0)[0] < tolN or np.shape(mat1)[0] <tolN:
        return None, leafType(dataSet)
    return bestIndex, bestValue


def createTree(dataSet, leafType = regLeaf, errType = regErr, ops = (1,4)):
    '''
    recursively create a tree from dataset:
        first we choose the best feature and value to split the tree
        and then we split the data into left data and right data
        finally build left tree and right tree according to left/right data
    
    Input:  dataSet
            leafType is the function to build leaf node
            errType is the function to calculate error
            ops is other parameters for building a tree
    Output: a tree denoted by a dict
    '''
    feat, val = chooseBestSplit(dataSet, leafType, errType, ops)
    if feat == None:
        return val
    retTree = {}
    retTree['spInd'] = feat
    retTree['spVal'] = val
    lSet, rSet = binSplitDataSet(dataSet, feat, val)
    retTree['left'] = createTree(lSet, leafType, errType, ops)
    retTree['right'] = createTree(rSet, leafType, errType, ops)
    return retTree


def isTree(obj):
    '''
    judge whether obj is a tree but not a leaf
    '''
    return type(obj).__name__ == 'dict'


def getMean(tree):
    '''
    return the mean value of a tree
    '''
    if isTree(tree['right']):
        tree['right'] = getMean(tree['right'])
    if isTree(tree['left']):
        tree['left'] = getMean(tree['left'])
    return (tree['left'] + tree['right'])/2.0


def prune(tree, testData):
    '''
    if overfit the training data, testing data will cause pruning
    
    Input:  a tree in a dict
            testing data to prune the tree
    Output: the tree after pruning
    '''
    if np.shape(testData)[0] == 0: # testing data doesn't exist
        return getMean(tree)
    
    # get two subtrees and prune them recursively
    if isTree(tree['right']) or isTree(tree['left']):
        lSet, rSet = binSplitDataSet(testData, tree['spInd'], tree['spVal'])
    if isTree(tree['left']):
        tree['left'] = prune(tree['left'], lSet)
    if isTree(tree['right']):
        tree['right'] = prune(tree['right'], rSet)
        
    if not isTree(tree['left']) and not isTree(tree['right']):
        lSet, rSet = binSplitDataSet(testData, tree['spInd'], tree['spVal'])
        errorNoMerge = sum(np.power(lSet[:,-1] - tree['left'], 2)) +\
                        sum(np.power(rSet[:,-1] - tree['right'], 2))
        treeMean = (tree['left'] + tree['right'])/2.0
        errorMerge = sum(np.power(testData[:,-1] - treeMean, 2))
        # print('testing')
        if errorMerge < errorNoMerge: # merge two trees if error can decrease
            print('merging')
            return treeMean
        else:
            return tree
    else:
        return tree
    
    
def linearSolve(dataSet):
    '''
    use a linear function to fit the data
    
    Input:  dataset with label at the last column
    Output: weight of linear regression model
            data part
            label part
    '''
    m, n = np.shape(dataSet)
    X = np.mat(np.ones((m,n)))
    Y = np.mat(np.ones((m,1)))
    X[:,1:n] = dataSet[:,0:n-1]                     # data
    Y = dataSet[:,-1].reshape((len(Y),1))           # label
    xTx = X.T * X
    if np.linalg.det(xTx) == 0.0:
        raise NameError('This matrix is singular, cannot do inverse,\n\
                        try increasing the second value of ops')
    ws = xTx.I * (X.T * Y)      # solve the linear equation
    return ws, X, Y


def modelLeaf(dataSet):
    '''
    get the weight of linear regression model of this dataset
    '''
    ws, X, Y = linearSolve(dataSet)
    return ws


def modelErr(dataSet):
    '''
    The square error of the dataset on a trained linear model
    
    Argument:
        dataSet (ndarray): a dataset with label at the last column
    Returns:
        the sum of square error of every data on the model
    '''
    ws, X, Y = linearSolve(dataSet)
    yHat = X * ws
    return sum(np.power(Y - yHat, 2))


def regTreeEval(model, inDat):
    '''
    calculate the leaf value of tree
    
    Argument:
        model (dict tree): a tree model
        inDat: input data, not used, just for keeping consistent with modelTreeEval
    Returns:
        the leaf value of a tree
    '''
    return float(model)


def modelTreeEval(model, inDat):
    '''
    calculate the prediction value of data
    
    Argument:
        model (dict tree): a tree model
        inDat (ndarray): data to predict
    Returns:
        y = X * beta
    '''
    n = np.shape(inDat)[1]
    X = np.mat(np.ones((1, n+1)))
    X[:, 1:n+1] = inDat
    return float(X * model)


def treeForeCast(tree, inData, modelEval = regTreeEval):
    '''
    Find a leaf for the datum and calculate prediction
    
    Argument:
        tree (dict): a tree model
        inData (ndarray): datum to predict
        modelEval (function): evaluation function
    Returns:
        the prediction of a data point
    '''
    if not isTree(tree):
        return modelEval(tree, inData)
    if inData[tree['spInd']] > tree['spVal']:
        if isTree(tree['left']):
            return treeForeCast(tree['left'], inData, modelEval)
        else:
            return modelEval(tree['left'], inData)
    else:
        if isTree(tree['right']):
            return treeForeCast(tree['right'], inData, modelEval)
        else:
            return modelEval(tree['right'], inData)


def createForeCast(tree, testData, modelEval = regTreeEval):
    '''
    Predict data
    
    Argument:
        tree (dict): a tree model
        testData (ndarray): data to predict
        modelEval (function): evaluation function
    Returns:
        yHat (ndarray): the prediction of data
    '''
    m = len(testData)
    yHat = np.mat(np.zeros((m,1)))
    for i in range(m):
        yHat[i,0] = treeForeCast(tree, np.mat(testData[i]), modelEval)
    return yHat


def predict(tree, data):
    '''
    predict using a regression tree
    
    Input:  a tree denoted by a dict
            1 dimensional data array
    Output: the classification result
    '''
    if not isTree(tree): # leaf, return value
        return tree
    feat = tree['spInd']
    val = tree['spVal']
    if data[feat] > val:
        return predict(tree['left'], data)
    elif data[feat] <= val:
        return predict(tree['right'], data)


def classification():
    print('Classification demo')
    
    # generate training data points
    pos1 = np.array([np.random.randn(2) for i in range(300)])
    pos1 = np.hstack((pos1, np.ones((300,1))))
    mu1 = np.array([2,2])
    neg1 = np.array([np.random.randn(2) + mu1 for i in range(300)])
    neg1 = np.hstack((neg1, np.zeros((300,1))))
    training_data = np.vstack((pos1, neg1))
    
    # show how the data look like
    print('Data points:')
    plt.figure(figsize = (15,15))
    plt.scatter(pos1[:,0],pos1[:,1])
    plt.scatter(neg1[:,0],neg1[:,1])
    plt.show()
    
    # generate testing data for pruning
    pos2 = np.array([np.random.randn(2) for i in range(300)])
    pos2 = np.hstack((pos2, np.ones((300,1))))
    mu2 = np.array([2,2])
    neg2 = np.array([np.random.randn(2) + mu2 for i in range(300)])
    neg2 = np.hstack((neg2, np.zeros((300,1))))
    testing_data = np.vstack((pos2, neg2))
    
    # build a tree and prune it
    tree1 = createTree(training_data, ops = (0.0001,10))
    # print('Regression tree: \n', tree1)
    tree2 = prune(tree1, testing_data)
    # print('Regression tree after pruning: \n', tree2)
    
    # test the model
    TP = 0
    TN = 0
    FP = 0
    FN = 0
    for i in range(len(testing_data)):
        pred = predict(tree2, testing_data[i])
        label = testing_data[i][-1]
        if pred > 0.5 and label == 1:
            TP += 1
        elif pred > 0.5 and label == 0:
            FP += 1
        elif pred < 0.5 and label == 1:
            FN += 1
        elif pred < 0.5 and label == 0:
            TN += 1
    Accuracy = (TP+TN)/(TP+TN+FP+FN)
    Precision = TP/(TP+FP)
    Recall = TP/(TP+FN)
    F1 = 2*(Precision * Recall)/(Precision + Recall)

    print('Accuracy: {:.2f}%'.format(100 * Accuracy))
    print('Precision: {:.2f}%'.format(100 * Precision))
    print('Recall: {:.2f}%'.format(100 * Recall))
    print('F1 measure: {:.2f}%'.format(100 * F1))


def regression():
    print('Regression demo:')
    
    # generate data
    x = np.linspace(0,10,1000).reshape((1000,1))
    y1 = (0.5 * x[:400] + 1).reshape(400,1)
    y2 = (-2 * x[400:] +11).reshape(600,1)
    y_train = np.vstack((y1, y2)) + 0.2 * np.random.randn(1000,1)
    y_test = np.vstack((y1, y2)) + 0.2 * np.random.randn(1000,1)
    training_data = np.hstack((x, y_train))
    testing_data = np.hstack((x, y_test))
    
    # show how the data look like
    print('Data points:')
    plt.figure(figsize = (15,15))
    plt.scatter(training_data[:,0],training_data[:,1])
    plt.show()
    
    # regression tree
    tree1 = createTree(training_data, ops = (1,20))
    yHat = createForeCast(tree1, testing_data[:,0])
    print('R^2 of regression tree:')
    print(np.corrcoef(yHat, testing_data[:,1], rowvar = 0)[0,1], '\n') # R square
    
    # model tree, this part may cause several seconds of waiting
    tree2 = createTree(training_data, modelLeaf, modelErr, (1,20))
    yHat = createForeCast(tree2, testing_data[:,0], modelTreeEval)
    print('R^2 of model tree:')
    print(np.corrcoef(yHat, testing_data[:,1], rowvar = 0)[0,1], '\n') # R square
    
    # linear regression
    ws, X, Y = linearSolve(training_data)
    for i in range(np.shape(testing_data)[0]):
        yHat[i] = testing_data[i,0] * ws[1,0] + ws[0,0]
    print('R^2 of linear regression:')
    print(np.corrcoef(yHat, testing_data[:,1], rowvar = 0)[0,1]) # R square


if __name__ == '__main__':
    #classification()
    regression()
    

























