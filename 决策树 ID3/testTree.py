import sys
sys.path.append('D:/学习/python算法/机器学习/决策树 ID3')

import numpy as np
import tree
import treePlotter

fr = open('D:/学习/python算法/机器学习/决策树 ID3/lenses.txt')
lenses = np.array([inst.strip().split() for inst in fr.readlines()])[:,1:]
lensesLabels = ['age', 'prescript', 'astigmatic', 'tearRate']
lensesTree = tree.createTree(lenses, lensesLabels)
print(lensesTree)
treePlotter.createPlot(lensesTree)