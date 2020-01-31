'''
staking是机器学习的一种ensemble方法
类似于K fold交叉验证，它也同样把训练样本划分为5份，然后用四份训练，一份做预测
不同的是，5次的预测结果最后整合起来，当作一个第二轮的训练集，并且5次都在测试集上预测
将5次的预测结果取平均，当作第一轮的预测结果，以及第二轮的测试集

如果我们只做一层staking，到这里就可以结束了，将这5次测试集的预测结果平均数当作最终结果
'''

import numpy as np
import lightgbm as lgb
from sklearn.model_selection import KFold, train_test_split
from sklearn.datasets import load_breast_cancer

# 从sklearn中载入内置的乳腺癌数据集
canceData = load_breast_cancer()
X = canceData.data
y = canceData.target
# 划分训练、测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0, test_size=0.2)

# out-of-fold predictions
ntrain = X_train.shape[0]
ntest = X_test.shape[0]
kf = KFold(n_splits = 5, random_state = 2019)

clf = lgb.LGBMClassifier()

def get_oof(clf, X_train, y_train, X_test):
    oof_train = np.zeros((ntrain, ))    # 1 * 训练数据维度
    oof_test = np.zeros((ntest, ))      # 1 * 测试集维度
    oof_test_skf = np.empty((5, ntest)) # 5 * 测试集维度
    
    for i, (train_index, test_index) in enumerate(kf.split(X_train)):
        kf_X_train = X_train[train_index]   # 训练数据中划分出的大份，当作子训练集'
        kf_y_train = y_train[train_index]   # 子训练集'标签
        kf_X_test = X_train[test_index]     # 原训练集中划分出的小份，当作子测试集'
        
        clf.fit(kf_X_train, kf_y_train)     # 在原训练集中划分出的两部分上训练
        
        oof_train[test_index] = clf.predict(kf_X_test)  # 这次的子模型在子测试集'的预测
        oof_test_skf[i, :] = clf.predict(X_test)        # 这次的子模型在原测试集的预测
        
    oof_test[:] = oof_test_skf.mean(axis = 0)           # 5次在原测试集的预测取平均
    return oof_train.reshape(-1, 1), oof_test.reshape(-1, 1) # 下一层的训练集、测试集

second_train, second_test = get_oof(clf, X_train, y_train, X_test)