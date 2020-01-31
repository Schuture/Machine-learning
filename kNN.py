import numpy as np
import matplotlib.pyplot as plt
np.random.seed(2)


def createDataSet():
    '''
    Generate a 2 dimensional two class dataset
    '''
    pos_data = np.ones((100, 2)) + np.random.randn(100, 2)
    pos_label = np.ones((100, 1))
    neg_data = -np.ones((100, 2)) + np.random.randn(100, 2)
    neg_label = np.zeros((100,1))
    
    positive = np.hstack((pos_data, pos_label))
    negative = np.hstack((neg_data, neg_label))
    dataset = np.vstack((positive, negative))
    
    return dataset


def showDataSet(dataset):
    '''
    Show all datapoints with scatter plot
    '''
    positive = dataset[dataset[:,2] == 1, :2]
    negative = dataset[dataset[:,2] == 0, :2]
    print('The dataset:\n')
    
    plt.figure(figsize = (15,15))
    
    plt.scatter(positive[:,0], positive[:,1], c = 'r')
    plt.scatter(negative[:,0], negative[:,1], c = 'b')
    
    plt.show()
    
    
def EuclideanDistance(point1, point2):
    '''
    Compute the Euclidean distance between two points
    
    Input: two points
    Output: the distance
    '''
    return np.linalg.norm((point1 - point2), 2)
    
    
def knnClassifier(dataset, point, k=9):
    '''
    Classify the data point
    
    Input:
        dataset: the dataset for computing distance
        point: a list, e.g. [1,2]
        k: how many neighbors
    Output:
        clazz: prediction of the point
    '''
    distances = []
    
    point = np.array(point)
    
    for p in dataset:
        distances.append([EuclideanDistance(point, p[:2]), p[2]]) # distance, class
        
    distances = sorted(distances, key = lambda x: x[0])
    pos_num = 0
    neg_num = 0
    for i in range(k):
        if distances[i][1] == 1:
            pos_num += 1
        else:
            neg_num += 1
            
    if pos_num >= neg_num:
        print('This point is positive')
        return 1
    else:
        print('This point is negative')
        return 0
    
    
if __name__ == '__main__':
    dataset = createDataSet()
    
    pred = []
    for datum in dataset[:, :2]:
        pred.append(knnClassifier(dataset, datum))
    pred = np.array(pred).reshape((len(dataset), 1))
    
    acc = (dataset[:, 2].reshape(((len(dataset), 1))) == pred).sum() / len(dataset)
    
    showDataSet(dataset)
    print('Accuracy is: {}%'.format(100 * acc))

































