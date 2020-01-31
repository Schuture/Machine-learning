import numpy as np
import matplotlib.pyplot as plt

# 定义训练数据
x = np.arange(0,5,0.1)
y = 5 * x + 3 + np.random.randn(len(x))

################################## 回归方程求取函数 ############################
def fit(x,y):
    if len(x) != len(y):
        return
    numerator = 0.0
    denominator = 0.0
    x_mean = np.mean(x)
    y_mean = np.mean(y)
    for i in range(len(x)):
        numerator += (x[i]-x_mean)*(y[i]-y_mean)
        denominator += np.square((x[i]-x_mean))
    print('numerator:',numerator,'denominator:',denominator)
    b0 = numerator/denominator
    b1 = y_mean - b0*x_mean
    return b0,b1

# 定义预测函数
def predit(x,b0,b1):
    return b0*x + b1

# 求取回归方程
b0,b1 = fit(x,y)
print('Line is:y = {:.2f}x + {:.2f}'.format(b0,b1))

# 预测
x_test = np.array([0.5,1.5,2.5,3,4])
y_test = np.zeros((1,len(x_test)))
for i in range(len(x_test)):
    y_test[0][i] = predit(x_test[i],b0,b1)

# 绘制图像
xx = np.linspace(0, 5)
yy = b0*xx + b1
plt.figure(figsize = (20, 16))
plt.plot(xx,yy,'k-')
plt.scatter(x,y)
plt.scatter(x_test,y_test[0])
plt.title('by single-variable regression')
plt.show()

############################## 法方程的解求参数 ################################
data = np.stack((x, np.ones(len(x))), axis = 1)
beta = np.linalg.inv(np.dot(data.T, data)).dot(data.T).dot(y)
b0,b1 = beta[0], beta[1]

# 线性回归求出来的直线
print('Line is:y = {:.2f}x + {:.2f}'.format(b0,b1))

# 预测
x_test = np.array([0.5,1.5,2.5,3,4])
y_test = np.zeros((1,len(x_test)))
for i in range(len(x_test)):
    y_test[0][i] = predit(x_test[i],b0,b1)

# 绘制图像
xx = np.linspace(0, 5)
yy = b0*xx + b1
plt.figure(figsize = (20, 16))
plt.plot(xx,yy,'k-')
plt.scatter(x,y)
plt.scatter(x_test,y_test[0])
plt.title('by multi-variable regression')
plt.show()