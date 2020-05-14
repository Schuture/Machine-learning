import numpy as np
import bisect

def mean_squared_error(y_true, y_pred):
    """ 计算y_true, y_pred的均方误差 """
    mse = np.mean(np.power(y_true - y_pred, 2))
    return mse
    

def shuffle_data(X, y, seed=None):
    """ 随机打乱X,y中的样本 """
    if seed:
        np.random.seed(seed)
    # X, y使用相同的打乱法
    idx = np.arange(X.shape[0])
    np.random.shuffle(idx)
    return X[idx], y[idx]
    

def train_test_split(X, y, test_size=0.5, shuffle=True, seed=None):
    """ 将数据集分割为训练、测试集 """
    if shuffle:
        X, y = shuffle_data(X, y, seed)

    split_i = len(y) - int(len(y) // (1 / test_size)) # 分割点
    X_train, X_test = X[:split_i], X[split_i:]
    y_train, y_test = y[:split_i], y[split_i:]

    return X_train, X_test, y_train, y_test
    

def divide_on_feature(X, feature_i, threshold):
    """ 基于某特征是否大于阈值，将数据集分割为两半 """
    split_func = None
    if isinstance(threshold, int) or isinstance(threshold, float):
        split_func = lambda sample: sample[feature_i] >= threshold
    else: # 字符型变量，只有相同/不相同两种，分到左右节点
        split_func = lambda sample: sample[feature_i] == threshold

    X_1 = np.array([sample for sample in X if split_func(sample)])
    X_2 = np.array([sample for sample in X if not split_func(sample)])

    return np.array([X_1, X_2])


def divide_on_feature2(X, sortedX, feature_i, threshold):
    ''' 
    基于某特征是否大于阈值，将数据集分割为两半
    假设X已经按照feature_i的值排好序了，第一列是id 
    '''
    if isinstance(threshold, int) or isinstance(threshold, float):
        split_point = bisect.bisect_left(sortedX[:, feature_i+1], threshold)
        idx1 = np.array(sortedX[split_point:, 0], dtype = np.int32) # 大于等于阈值的为X1，取后半边
        idx2 = np.array(sortedX[:split_point, 0], dtype = np.int32) # 小于阈值的为X2，取前半边
        X_1 = X[idx1, 1:] # 注意！！！！！这里是在原X中选取，不是在sortedX中选
        X_2 = X[idx2, 1:]

        return np.array([X_1, X_2])
    else: # 字符型变量，只有相同/不相同两种，分到左右节点
        X_1 = X[np.where(X[:,feature_i+1]==threshold),1:]
        X_2 = X[np.where(X[:,feature_i+1]!=threshold),1:]
        
        return np.array([X_1, X_2])
    