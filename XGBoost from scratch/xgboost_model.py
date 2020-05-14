from __future__ import division, print_function
import numpy as np
import progressbar
from utils import divide_on_feature, divide_on_feature2

bar_widgets = [
    'Training: ', progressbar.Percentage(), ' ', progressbar.Bar(marker="-", left="[", right="]"),
    ' ', progressbar.ETA(), '\n'
]


class LeastSquaresLoss(): # loss类，包含一阶二阶梯度两个方法
    """ 最小二乘损失 """

    def gradient(self, actual, predicted):
        return actual - predicted # 对于平方损失而言，负梯度就是残差

    def hess(self, actual, predicted):
        return np.ones_like(actual) # 对于平方损失而言，二阶导数为1
    
    
class DecisionNode():
    """ 决策树的节点类

    Parameters:
    -----------
    feature_i: int
        分裂的特征索引号
    threshold: float
        分裂特征的分裂阈值
    value: float
        节点的预测值（如果是叶子节点）
    true_branch: DecisionNode
        左子节点，包含分裂特征超过阈值的样本
    false_branch: DecisionNode
        右子节点，包含分裂特征小于阈值的样本
    """

    def __init__(self, feature_i=None, threshold=None,
                 value=None, true_branch=None, false_branch=None):
        self.feature_i = feature_i
        self.threshold = threshold
        self.value = value
        self.true_branch = true_branch
        self.false_branch = false_branch


class DecisionTree(object):
    """ 决策树的父类，各种决策树（分类、回归）都要继承它

    Parameters:
    -----------
    min_samples_split: int
        继续分裂时，节点的最小样本数
    min_impurity: float
        继续分裂时，节点的最小不纯度
    max_depth: int
        树的最大深度
    loss: function
        给梯度提升模型的损失函数
    """

    def __init__(self, min_samples_split=2, min_impurity=1e-7,
                 max_depth=float("inf"), loss=None):
        self.root = None  # 决策树的根节点
        self.min_samples_split = min_samples_split
        self.min_impurity = min_impurity
        self.max_depth = max_depth
        # 不纯度下降量的计算方法，分类树：gini系数，回归树：方差
        self._impurity_calculation = None
        # 树节点取值的方法，分类树：选取出现最多次数的值，回归树：取所有值的平均值
        self._leaf_value_calculation = None
        # y是否是one-hot编码的(multi-dim/one-dim)
        self.one_dim = None
        # 是否梯度提升，梯度提升方法需要指定loss函数
        self.loss = loss

    def fit(self, X, y, loss=None):
        """ 建立决策树 """
        self.one_dim = len(np.shape(y)) == 1
        self.root = self._build_tree(X, y)
        self.loss = None

    def _build_tree(self, X, y, current_depth=0):
        """ 递归地建立决策树 """
        largest_impurity = 0
        best_criteria = None  # 特征索引和阈值
        best_sets = None  # 数据的子集

        # 把一维的y变成二维
        if len(np.shape(y)) == 1:
            y = np.expand_dims(y, axis=1) # 扩展y的维度，例如(366,) -> (366,1)

        # 将X, y拼接成一个数据集
        Xy = np.concatenate((X, y), axis=1)
        n_samples, n_features = np.shape(X) # X的维度
        
        accelerate = True # 选择直接遍历/预排序（加速）
        
        # 数据表Xy添加索引维度
        if accelerate:
            Xy = np.hstack((np.arange(len(Xy)).reshape((-1,1)), Xy))
        
        if n_samples >= self.min_samples_split and current_depth <= self.max_depth:
            # 遍历特征
            for feature_i in range(n_features):
                # 特征i的所有值
                feature_values = np.expand_dims(X[:, feature_i], axis=1)
                unique_values = np.unique(feature_values)
                
                # 数据表按照当前特征的值排序，只排序数值型变量
                if isinstance(unique_values[0], int) or isinstance(unique_values[0], float):
                    sorted_Xy = np.array(sorted(Xy, key = lambda x: x[feature_i+1]))
                # 遍历特征i的所有不同的值并计算出不纯度
                for threshold in unique_values:
                    # 将数据集根据阈值分开，直接遍历法/预排序法
                    if accelerate:
                        Xy1, Xy2 = divide_on_feature2(Xy, sorted_Xy, feature_i, threshold)
                    else:
                        Xy1, Xy2 = divide_on_feature(Xy, feature_i, threshold)
                    
                    if len(Xy1) > 0 and len(Xy2) > 0: # 两个数据集都不是空的
                        # 提取出两个数据集的y值来计算不纯度的下降
                        y1 = Xy1[:, -2:]
                        y2 = Xy2[:, -2:]
                        impurity = self._impurity_calculation(y, y1, y2)
                        
                        # 找到相比不分裂，最大增益的分裂点
                        if impurity > largest_impurity:
                            largest_impurity = impurity
                            best_criteria = {"feature_i": feature_i, "threshold": threshold}
                            best_sets = {
                                "leftX": Xy1[:, :n_features],  # 左子树的X
                                "lefty": Xy1[:, -2:],  # 左子树的y
                                "rightX": Xy2[:, :n_features],  # 右子树的X
                                "righty": Xy2[:, -2:]  # 右子树的y
                            }

        if largest_impurity > self.min_impurity: # 分裂增益大于阈值才分裂
            # 给左子树、右子树继续分裂。返回的是分裂的特征信息，不包含数据集
            true_branch = self._build_tree(best_sets["leftX"], best_sets["lefty"], current_depth + 1)
            false_branch = self._build_tree(best_sets["rightX"], best_sets["righty"], current_depth + 1)
            return DecisionNode(feature_i=best_criteria["feature_i"], threshold=best_criteria[
                "threshold"], true_branch=true_branch, false_branch=false_branch)

        # 如果没有继续分裂，则为叶子节点，计算这个节点的预测值
        leaf_value = self._leaf_value_calculation(y)
        return DecisionNode(value=leaf_value)

    def predict_value(self, x, tree=None):
        """ 递归地搜索树的叶子节点，将叶子结点的值当作预测值
        Parameters:
        -----------
        x: ndarray
            一个样本，是一个包含各个特征的向量
        tree: DecisionNode
            一棵决策树
        """

        if tree is None:
            tree = self.root

        # 到了叶子节点，返回预测值
        if tree.value is not None:
            return tree.value

        # 如果没到叶子节点，就选择当前节点的分裂特征对应的值
        feature_value = x[tree.feature_i]

        # 往左子树还是右子树走
        branch = tree.false_branch
        if isinstance(feature_value, int) or isinstance(feature_value, float):
            if feature_value >= tree.threshold: # 如果是数值变量就直接对比判断
                branch = tree.true_branch
        elif feature_value == tree.threshold: # 类别型变量只有相等时才往true子树走
            branch = tree.true_branch

        # 递归
        return self.predict_value(x, branch)

    def predict(self, X):
        """ 一个一个预测数据集X中的样本 """
        y_pred = []
        for x in X:
            y_pred.append(self.predict_value(x))
        return y_pred


class XGBoostRegressionTree(DecisionTree): # 继承了决策树类的回归树类
    """
    XGBoost回归树
    - Reference -
    http://xgboost.readthedocs.io/en/latest/model.html
    """

    def _split(self, y):
        """ y包含y_true为数据集左列，y_pred为右列。将其分开为两个矩阵 """
        col = int(np.shape(y)[1]/2)
        y, y_pred = y[:, :col], y[:, col:]
        return y, y_pred

    def _gain(self, y, y_pred): # 单个节点的增益，根据gradient, hessian来计算
        nominator = np.power((self.loss.gradient(y, y_pred)).sum(), 2)
        denominator = self.loss.hess(y, y_pred).sum()
        return 0.5 * (nominator / denominator)

    def _gain_by_taylor(self, y, y1, y2): # 分裂相比不分裂的增益
        # 将y分割成y, y_pred
        y, y_pred = self._split(y)
        y1, y1_pred = self._split(y1)
        y2, y2_pred = self._split(y2)

        true_gain = self._gain(y1, y1_pred)
        false_gain = self._gain(y2, y2_pred)
        gain = self._gain(y, y_pred)
        return true_gain + false_gain - gain

    def _approximate_update(self, y): # 更新的估计（一阶导数/二阶导数）
        # 将y分割成y, y_pred
        y, y_pred = self._split(y)
        gradient = np.sum(self.loss.gradient(y, y_pred),axis=0)
        hessian = np.sum(self.loss.hess(y, y_pred), axis=0)
        update_approximation =  gradient / hessian
        return update_approximation

    def fit(self, X, y):
        self._impurity_calculation = self._gain_by_taylor
        self._leaf_value_calculation = self._approximate_update
        super(XGBoostRegressionTree, self).fit(X, y)


class XGBoost(object):
    """ 模型实现，包含训练和预测两个方法
    Parameters:
    -----------
    n_estimators: int
        决策树的数量
    learning_rate: float
        拟合目标为负梯度乘学习率
    min_samples_split: int
        分裂节点时，节点上最小的样本量
    min_impurity: float
        分裂节点时，节点上最小的不纯度
    max_depth: int
        树的最大深度
    """

    def __init__(self, n_estimators=200, learning_rate=0.01, min_samples_split=2,
                 min_impurity=1e-7, max_depth=2):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.min_samples_split = min_samples_split
        self.min_impurity = min_impurity
        self.max_depth = max_depth

        self.bar = progressbar.ProgressBar(widgets=bar_widgets)

        # 回归损失：最小二乘损失
        self.loss = LeastSquaresLoss()

        # 初始化回归树
        self.trees = [] # 记录一系列的树，每一棵树都是XGBoostRegressionTree
        for _ in range(n_estimators):
            tree = XGBoostRegressionTree(
                min_samples_split=self.min_samples_split,
                min_impurity=min_impurity,
                max_depth=self.max_depth,
                loss=self.loss)

            self.trees.append(tree)    # 迭代加入初未经训练的树

    def fit(self, X, y):
        # y = to_categorical(y)
        m = X.shape[0]
        y = np.reshape(y, (m, -1))
        y_pred = np.zeros(np.shape(y)) # 初始化y的预测值，全0
        for i in self.bar(range(self.n_estimators)): # 一轮轮迭代训练树
            tree = self.trees[i]
            y_and_pred = np.concatenate((y, y_pred), axis=1)# 把y, y_pred左右拼接
            tree.fit(X, y_and_pred)                         # 回归树拟合样本和标签
            update_pred = tree.predict(X)                   # 拟合的残差
            update_pred = np.reshape(update_pred, (m, -1))
            y_pred += update_pred                           # 直接将残差加到y_pred上

    def predict(self, X):
        y_pred = None
        m = X.shape[0]
        # Make predictions
        for tree in self.trees:
            # Estimate gradient and update prediction
            update_pred = tree.predict(X)                   # 这一棵树的预测值
            update_pred = np.reshape(update_pred, (m, -1))  # reshape为样本数的形状
            if y_pred is None:                              # 初始化y_pred
                y_pred = np.zeros_like(update_pred)
            y_pred += update_pred                           # 将所有树的预测值相加

        return y_pred
