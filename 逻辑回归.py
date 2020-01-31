import numpy as np
import matplotlib.pyplot as plt

################################## 超参数 #####################################
N = 500
p = 20
epochs = 8
alpha = 0.05
threshold = 0.5

################################## 定义函数 ####################################
def Sigmoid(x, beta): # P(x,beta)
    '''
    Input: data x,      N x p
           weight beta, p x 1
    Output: P(Y=1|X=x), N x 1
    '''
    exp = np.exp(np.dot(x, beta))
    return exp / (1 + exp)


def Loss(beta):
    '''
    Input: weight beta,                   p x 1
    Output: log-likelihood loss function, 1 x 1
    '''
    y_hat = np.where(Sigmoid(x, beta_hat) < threshold, 0, 1)
    return - y.dot(np.log(1e-5 + y_hat)) - (1 - y).dot(np.log(1 - y_hat + 1e-5))


################################## 生成数据点 ##################################
x = np.random.randn(N, p) 
beta = np.ones(p)
y = np.where(Sigmoid(x, beta) < threshold, 0, 1) # 小于阈值为负例，否则正例

################################# 牛顿法估计参数 ###############################
beta_hat = np.random.randn(p) # 初始化参数估计量
loss = np.zeros(epochs)
for epoch in range(epochs):
    P = Sigmoid(x, beta_hat)
    W = np.diag( P * (1 - P))
    beta_hat = beta_hat + alpha * np.linalg.inv(x.T.dot(W).dot(x)).dot(x.T).dot(y-P)
    loss[epoch] = Loss(beta_hat)
    
################################### 测试模型 ##################################
y_hat = np.where(Sigmoid(x, beta_hat) < threshold, 0, 1)
accuracy = np.mean(np.where(y_hat==y, 1, 0))
print('Accuracy is: {:.2f}%'.format(accuracy*100))

TP = np.sum(np.where((y_hat==1) & (y_hat==y), 1, 0))
TN = np.sum(np.where((y_hat==0) & (y_hat==y), 1, 0))
FP = np.sum(np.where((y_hat==1) & (y_hat!=y), 1, 0))
FN = np.sum(np.where((y_hat==0) & (y_hat!=y), 1, 0))
precision = TP/(TP+FP)
recall = TP/(TP+FN)
print('Precision is: {:.2f}%'.format(100*precision))
print('Recall is: {:.2f}%'.format(100*recall))
print('F1 measure is: {:.2f}%'.format(100*2*precision*recall/(precision+recall)))

# loss iteratioin
plt.figure(figsize=(15, 10))
plt.xlabel('epochs')
plt.ylabel('loss')
plt.title('loss iteration')
plt.plot(loss)

# ROC
TPR = [0]
FPR = [0]
P = Sigmoid(x, beta_hat)
samples = np.stack((y, P), axis = 1)
samples = sorted(samples, key = lambda x: x[1], reverse = True) # 按照概率从高到低
m_plus = np.sum(list(map(lambda x: x == 1, y)))
m_minus = np.sum(list(map(lambda x: x == 0, y)))
for i in range(N):
    if samples[i][0] == 1:
        TPR.append(TPR[i] + 1/m_plus) # 正例则y向上挪
        FPR.append(FPR[i])
    else:
        TPR.append(TPR[i])
        FPR.append(FPR[i] + 1/m_minus) # 负例则x向右挪

plt.figure(figsize=(15, 10))
plt.xlabel('FPR')
plt.ylabel('TPR')
plt.title('ROC curve')
plt.plot(FPR, TPR)

# AUC
AUC = 0
for i in range(N): # 近似做积分
    AUC += (FPR[i+1] - FPR[i]) * TPR[i+1]
print('The AUC is {:.4f}'.format(AUC))


























