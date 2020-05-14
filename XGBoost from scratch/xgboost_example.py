import sys
sys.path.append(sys.path)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from utils import train_test_split
from utils import mean_squared_error
from xgboost_model import XGBoost

np.random.seed(10)

def main():
    print ("-- XGBoost --")

    # 载入气温数据
    data = pd.read_csv('TempLinkoping2016.txt', sep="\t")

    time = np.atleast_2d(data["time"].values).T
    temp = np.atleast_2d(data["temp"].values).T

    X = time.reshape((-1, 1))               # Xi为0-1之间，一年中的比例
    X = np.insert(X, 0, values=1, axis=1)   # 偏置项，当作第一个特征
    
    # 数据增强，扩充到16倍
    #X = np.vstack((X,X,X,X))
    #X = np.vstack((X,X,X,X))
    #temp = np.vstack((temp,temp+0.01,temp+0.02,temp+0.03))
    #temp = np.vstack((temp,temp+0.1,temp+0.2,temp+0.3))
    
    y = temp[:, 0]                          # Temperature. 减少到一维

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
    #print(y_train)
    model = XGBoost()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    #y_pred_line = model.predict(X) # 使用训练好的模型对原数据进行预测
    
    mse = mean_squared_error(y_test, y_pred)
    print ("Mean Squared Error: {:.2f}".format(mse))
    
    # Color map
    cmap = plt.get_cmap('viridis')
    # Plot the results
    plt.figure(figsize=(12,12))
    m1 = plt.scatter(366 * X_train[:, 1], y_train, color=cmap(0.9), s=10)
    m2 = plt.scatter(366 * X_test[:, 1], y_test, color=cmap(0.5), s=10)
    m3 = plt.scatter(366 * X_test[:, 1], y_pred, color='black', s=10)
    plt.suptitle("XGBoost Regression Tree", fontsize=28)
    plt.title("MSE: {:.2f}".format(mse), fontsize=20)
    plt.xlabel('Day', fontsize=18)
    plt.ylabel('Temperature in Celcius', fontsize=16)
    plt.tick_params(labelsize=15) # 刻度字体大小
    plt.legend((m1, m2, m3), ("Training data", "Test data", "Prediction"),
               loc='lower right', fontsize=15)
    plt.show()


if __name__ == "__main__":
    main()








































