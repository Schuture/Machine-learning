from sklearn import datasets
from sklearn.model_selection import train_test_split
import xgboost as xgb
from sklearn import metrics

# 导入鸢尾花的数据
iris = datasets.load_iris()
# 特征数据
data = iris.data[:100] # 有4个特征
# 标签
label = iris.target[:100] # 只取前100个则仅含有两类

# 提取训练集和测试集，按照3:1的比例
# random_state：是随机数的种子
train_x, test_x, train_y, test_y = train_test_split(data, label, random_state=0)

# 将ndarray数据加载为DMatrix类型，使得xgb能够读取
dtrain = xgb.DMatrix(train_x, label = train_y) 
dtest = xgb.DMatrix(test_x)

# 参数设置
params = {'booster':'gbtree',           # 树模型
    'objective': 'binary:logistic',     # 二分类问题，逻辑
    'eval_metric': 'auc',               # 评价指标
    'max_depth':4,                      # 树最大深度，四个参数对应四层
    'lambda':10,                        # 控制模型复杂度的权重值的L2正则化项参数
    'subsample':0.75,                   # 随机采样75%训练样本用于一次训练，防止过拟合
    'colsample_bytree':0.75,            # 生成树时进行的列采样
    'min_child_weight':2,
    'eta': 0.025,                       # 如同学习率
    'seed':0,                           # 随机种子
    'nthread':8,                        # cpu 线程数
    'silent':0}                         # 设置成1则没有运行信息输出，最好是设置为0

watchlist = [(dtrain,'train')]

# 训练模型并保存
print('training start')
bst = xgb.train(params, dtrain, num_boost_round = 10, evals = watchlist)

# 预测
print('prediction start')
ypred = bst.predict(dtest)

# 设置阈值, 输出一些评价指标
# 0.5为阈值，ypred >= 0.5输出0或1
y_pred = (ypred >= 0.5) * 1

# ROC曲线下与坐标轴围成的面积
print ('AUC: %.4f' % metrics.roc_auc_score(test_y,ypred))
# 准确率
print ('ACC: %.4f' % metrics.accuracy_score(test_y,y_pred))
# 精确率和召回率
print ('Recall: %.4f' % metrics.recall_score(test_y,y_pred))
print ('Precesion: %.4f' %metrics.precision_score(test_y,y_pred))
# F1
print ('F1-score: %.4f' %metrics.f1_score(test_y,y_pred))















