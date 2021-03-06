import numpy as np
import matplotlib.pyplot as plt
import random
import time


############################## functions for data #############################
def loadDataSet(fileName):
    dataMat = []; labelMat = []
    fr = open(fileName)
    for line in fr.readlines():                                   
        lineArr = line.strip().split('\t')
        dataMat.append([float(lineArr[0]), float(lineArr[1])])     
        labelMat.append(float(lineArr[2]))                      
    return dataMat, labelMat


def showDataSet(dataMat, labelMat):
    '''
    Show the data points
    '''
    print('Original data points:')
    dataMat = np.array(dataMat)
    labelMat = np.array(labelMat)
    print('The size of this dataset is: {}'.format(labelMat.shape[0]))
    pos = dataMat[np.where(labelMat > 0),:].reshape((300,2))
    neg = dataMat[np.where(labelMat < 0),:].reshape((300,2))
    plt.figure(figsize = (15,15))
    plt.scatter(pos[:,0], pos[:,1])
    plt.scatter(neg[:,0], neg[:,1])
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.show()


####################### functions for simplified method #######################
def selectJrand(i, m):
    '''
    Select a number j randomly from 0,1,...,m
    Inputs:
        i (int): a number, j must not be equal to j
        m (int): select j from (0,m)
    Outputs:
        j (int):
            a random integer from 0 to m
    '''
    j = i
    while j == i:
        j = int(random.uniform(0,m))
        return j
    

def clipAlpha(aj, H, L):
    '''
    Restrict aj between two numbers
    Inputs:
        aj (int): the number to restrict
        H (int): the upper bound
        L (int): the lower bound
    Output:
        aj (int): the number between to bounds
    '''
    if aj > H:
        aj = H
    if L > aj:
        aj = L
    return aj


def smoSimple(dataMatIn, classLabels, C, toler, maxIter):
    '''
    The simplified SMO algorithm
    The training will stop until 40 continuous tesing pass
    '''
    dataMatrix = np.mat(dataMatIn)
    labelMat = np.mat(classLabels).transpose()
    b = 0
    m,n = np.shape(dataMatrix)
    alphas = np.mat(np.zeros((m,1)))
    iter = 0
    while (iter < maxIter):
        alphaPairsChanged = 0
        for i in range(m):
            fXi = float(np.multiply(alphas,labelMat).T*\
               (dataMatrix*dataMatrix[i,:].T)) + b
            Ei = fXi - float(labelMat[i])
            if ((labelMat[i]*Ei < -toler) and (alphas[i] < C)) or \
                ((labelMat[i]*Ei > toler) and (alphas[i] > 0)):
                j = selectJrand(i,m)
                fXj = float(np.multiply(alphas,labelMat).T*\
                            (dataMatrix*dataMatrix[j,:].T)) + b
                Ej = fXj - float(labelMat[j])
                alphaIold = alphas[i].copy()
                alphaJold = alphas[j].copy()
                if (labelMat[i] != labelMat[j]):
                    L = max(0, alphas[j] - alphas[i])
                    H = min(C, C + alphas[j] - alphas[i])
                else:
                    L = max(0, alphas[j] + alphas[i] - C)
                    H = min(C, alphas[j] + alphas[i])
                if L==H: 
                    #print("L==H")
                    continue
                eta = 2.0 * dataMatrix[i,:]*dataMatrix[j,:].T - \
                      dataMatrix[i,:]*dataMatrix[i,:].T - \
                      dataMatrix[j,:]*dataMatrix[j,:].T
                if eta >= 0: 
                    #print("eta>=0")
                    continue
                alphas[j] -= labelMat[j]*(Ei - Ej)/eta
                alphas[j] = clipAlpha(alphas[j],H,L)
                if (abs(alphas[j] - alphaJold) < 0.00001):
                    #print('j mot moving enough')
                    continue
                alphas[i] += labelMat[j] * labelMat[i] * (alphaJold - alphas[j])
                b1 = b - Ei- labelMat[i] * (alphas[i]-alphaIold) * \
                    dataMatrix[i,:] * dataMatrix[i,:].T - \
                    labelMat[j] * (alphas[j]-alphaJold) * \
                    dataMatrix[i,:] * dataMatrix[j,:].T
                b2 = b - Ej- labelMat[i] * (alphas[i]-alphaIold) * \
                    dataMatrix[i,:] * dataMatrix[j,:].T - \
                    labelMat[j] * (alphas[j]-alphaJold) * \
                    dataMatrix[j,:] * dataMatrix[j,:].T
                if (0 < alphas[i]) and (C > alphas[i]): 
                    b = b1
                elif (0 < alphas[j]) and (C > alphas[j]): 
                    b = b2
                else: 
                    b = (b1 + b2)/2.0
                alphaPairsChanged += 1
                print("iter: {}, i: {}, pairs changed: {}".\
                      format(iter, i, alphaPairsChanged))
        if (alphaPairsChanged == 0): 
            iter += 1
        else: 
            iter = 0
        print("iteration number: {}".format(iter))
    return b,alphas


####################### functions for complete method #########################
class optStruct:
    '''
    The object for completed SMO algorithm
    '''
    def __init__(self, dataMatIn, classLabels, C, toler):
        self.X = dataMatIn
        self.labelMat = classLabels
        self.C = C
        self.tol = toler
        self.m = np.shape(dataMatIn)[0]
        self.alphas = np.mat(np.zeros((self.m, 1)))
        self.b = 0
        self.eCache = np.mat(np.zeros((self.m, 2)))


def calcEk(oS, k):
    fXk = float(np.multiply(oS.alphas, oS.labelMat).T * \
                (oS.X * oS.X[k,:].T)) + oS.b
    Ek = fXk - float(oS.labelMat[k])
    return Ek


def selectJ(i, oS, Ei):
    maxK = -1
    maxDeltaE = 0
    Ej = 0
    oS.eCache[i] = [1, Ei]
    validEcacheList = np.nonzero(oS.eCache[:, 0].A)[0]
    if len(validEcacheList) > 1:
        for k in validEcacheList:
            if k == i:
                continue
            Ek = calcEk(oS, k)
            deltaE = abs(Ei - Ek)
            if deltaE > maxDeltaE:
                maxK = k
                maxDeltaE = deltaE
                Ej = Ek
        return maxK, Ej
    else:
        j = selectJrand(i, oS.m)
        Ej = calcEk(oS, j)
    return j, Ej


def updateEk(oS, k):
    Ek = calcEk(oS, k)
    oS.eCache[k] = [1, Ek]


def get_w(dataMat, labelMat, alphas):
    alphas, dataMat, labelMat = np.array(alphas), np.array(dataMat), np.array(labelMat)
    w = np.dot((np.tile(labelMat.reshape(1, -1).T, (1, 2)) * dataMat).T, alphas)
    return w.tolist()


def innerL(i, oS):
    '''
    Inner loop of complete SMO
    '''
    Ei = calcEk(oS, i)
    if ((oS.labelMat[i] * Ei < -oS.tol) and (oS.alphas[i] < oS.C)) or \
        ((oS.labelMat[i] * Ei > oS.tol) and (oS.alphas[i] > 0)):
        j,Ej = selectJ(i, oS, Ei)
        alphaIold = oS.alphas[i].copy(); alphaJold = oS.alphas[j].copy();
        if (oS.labelMat[i] != oS.labelMat[j]):
            L = max(0, oS.alphas[j] - oS.alphas[i])
            H = min(oS.C, oS.C + oS.alphas[j] - oS.alphas[i])
        else:
            L = max(0, oS.alphas[j] + oS.alphas[i] - oS.C)
            H = min(oS.C, oS.alphas[j] + oS.alphas[i])
        if L == H: 
            #print("L==H")
            return 0
        eta = 2.0 * oS.X[i,:] * oS.X[j,:].T - oS.X[i,:] * oS.X[i,:].T - oS.X[j,:] * oS.X[j,:].T
        if eta >= 0: 
            #print("eta>=0")
            return 0
        oS.alphas[j] -= oS.labelMat[j] * (Ei - Ej)/eta
        oS.alphas[j] = clipAlpha(oS.alphas[j],H,L)
        updateEk(oS, j)
        if (abs(oS.alphas[j] - alphaJold) < 0.00001): 
            #print("alpha_j变化太小")
            return 0
        oS.alphas[i] += oS.labelMat[j]*oS.labelMat[i]*(alphaJold - oS.alphas[j])
        updateEk(oS, i)
        b1 = oS.b - Ei- oS.labelMat[i] * (oS.alphas[i]-alphaIold) * \
            oS.X[i,:] * oS.X[i,:].T - oS.labelMat[j] * \
            (oS.alphas[j]-alphaJold) * oS.X[i,:] * oS.X[j,:].T
        b2 = oS.b - Ej- oS.labelMat[i] * (oS.alphas[i]-alphaIold) * \
            oS.X[i,:] * oS.X[j,:].T - oS.labelMat[j] * \
            (oS.alphas[j]-alphaJold) * oS.X[j,:] * oS.X[j,:].T
        if (0 < oS.alphas[i]) and (oS.C > oS.alphas[i]): oS.b = b1
        elif (0 < oS.alphas[j]) and (oS.C > oS.alphas[j]): oS.b = b2
        else: oS.b = (b1 + b2)/2.0
        return 1
    else: 
        return 0


def smoP(dataMatIn, classLabels, C, toler, maxIter):
    '''
    The outer loop for complete SMO
    '''
    oS = optStruct(np.mat(dataMatIn), np.mat(classLabels).transpose(), C, toler)					
    iter = 0 																						
    entireSet = True; alphaPairsChanged = 0
    while (iter < maxIter) and ((alphaPairsChanged > 0) or (entireSet)):							
        alphaPairsChanged = 0
        if entireSet:																					
            for i in range(oS.m):        
                alphaPairsChanged += innerL(i,oS)												
                #print("全样本遍历:第%d次迭代 样本:%d, alpha优化次数:%d" % (iter,i,alphaPairsChanged))
            iter += 1
        else: 																						
            nonBoundIs = np.nonzero((oS.alphas.A > 0) * (oS.alphas.A < C))[0]						
            for i in nonBoundIs:
                alphaPairsChanged += innerL(i,oS)
                #print("非边界遍历:第%d次迭代 样本:%d, alpha优化次数:%d" % (iter,i,alphaPairsChanged))
            iter += 1
        if entireSet:																				
            entireSet = False
        elif (alphaPairsChanged == 0):																
            entireSet = True  
        print("迭代次数: %d" % iter)
    return oS.b,oS.alphas


############################## functions for demo #############################
def showClassifer(dataMat, labelMat, alphas, w, b):
    '''
    Show the datapoints, supporting vectors and splitting line
    '''
    plt.figure(figsize = (15,15))
    # extract positive and negative samples
    dataMat = np.array(dataMat)
    labelMat = np.array(labelMat)
    pos = dataMat[np.where(labelMat > 0),:].reshape((300,2))
    neg = dataMat[np.where(labelMat < 0),:].reshape((300,2))     
    plt.scatter(pos[:,0], pos[:,1])
    plt.scatter(neg[:,0], neg[:,1])
    # draw splitting line
    x1 = max(dataMat[:,0])
    x2 = min(dataMat[:,0])
    a1, a2 = w
    b = float(b)
    a1 = float(a1[0])
    a2 = float(a2[0])
    y1, y2 = (-b- a1*x1)/a2, (-b - a1*x2)/a2
    plt.plot([x1, x2], [y1, y2])
    # extract the closest points to the splitting line
    for i, alpha in enumerate(alphas):
        if abs(alpha) > 0:
            x, y = dataMat[i]
            plt.scatter([x], [y], s=150, c='none', alpha=0.7, linewidth=1.5, edgecolor='red')
    plt.show()


def simpleMethod():
    '''
    Demo for simplified SMO method
    '''
    start = time.time()
    dataMat, labelMat = loadDataSet('testSet.txt')
    showDataSet(dataMat, labelMat)
    b, alphas = smoSimple(dataMat, labelMat, 0.6, 0.001, 40)
    w = get_w(dataMat, labelMat, alphas)
    showClassifer(dataMat, labelMat, alphas, w, b)
    print('Simple method consumes {} seconds'.format(time.time() - start))


def completeMethod():
    '''
    Demo for complete SMO method
    '''
    start = time.time()
    dataMat, labelMat = loadDataSet('testSet.txt')
    showDataSet(dataMat, labelMat)
    b, alphas = smoP(dataMat, labelMat, 0.6, 0.001, 40)
    w = get_w(dataMat, labelMat, alphas)
    showClassifer(dataMat, labelMat, alphas, w, b)
    print('Complete method consumes {} seconds'.format(time.time() - start))


if __name__ == '__main__':
    #simpleMethod()
    completeMethod()
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    