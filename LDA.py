import numpy as np
import matplotlib.pyplot as plt

############################### generate samples ##############################
# positive samples
mu1 = np.array([5,10])
sigma1 = np.array([[1,-0.4],[-0.6,1]])
pos = np.array([np.random.randn(2).dot(sigma1) + mu1 for i in range(500)])

# negative samples
mu2 = np.array([7,10])
sigma2 = np.array([[1,-0.4],[-0.6,1]])
neg = np.array([np.random.randn(2).dot(sigma2) + mu2 for j in range(500)])

# show how the data look like
print('Data points:')
plt.figure(figsize = (15,15))
plt.scatter(pos[:,0],pos[:,1])
plt.scatter(neg[:,0],neg[:,1])
plt.show()


################################## train LDA ##################################
# within-class scatter matrix
print('Training start!')
Mu1 = np.mean(pos, axis = 0)
Mu2 = np.mean(neg, axis = 0)
Sigma1 = (pos - Mu1).T.dot(pos - Mu1)
Sigma2 = (neg - Mu2).T.dot(neg - Mu2)
Sw = Sigma1 + Sigma2

# between-class scatter matrix
Sb = (Mu1 - Mu2).T.dot(Mu1 - Mu2)

# the model: w, centers of each clss
w = np.linalg.inv(Sw).dot(Mu1 - Mu2)
center1 = w.T.dot(Mu1)
center2 = w.T.dot(Mu2)

# show how the projection direction look like
print('Data points and projection direction:')
x = np.linspace(4, 12, 10)
y = (w[1]/w[0]) * x            # y = kx
plt.figure(figsize = (15,15))
plt.scatter(pos[:,0],pos[:,1])
plt.scatter(neg[:,0],neg[:,1])
plt.plot(x,y)
plt.show()


###################################### test ###################################
print('Testing start!')
TP = 0
TN = 0
FP = 0
FN = 0
for i in range(len(pos)):
    projection = w.T.dot(pos[i])
    if np.linalg.norm(projection - center1) < np.linalg.norm(projection - center2):
        TP += 1
    else:
        FN += 1
for i in range(len(neg)):
    projection = w.T.dot(neg[i])
    if np.linalg.norm(projection - center2) < np.linalg.norm(projection - center1):
        TN += 1
    else:
        FP += 1

Accuracy = (TP+TN)/(TP+TN+FP+FN)
Precision = TP/(TP+FP)
Recall = TP/(TP+FN)
F1 = 2*(Precision * Recall)/(Precision + Recall)

print('Accuracy: {:.2f}%'.format(100 * Accuracy))
print('Precision: {:.2f}%'.format(100 * Precision))
print('Recall: {:.2f}%'.format(100 * Recall))
print('F1 measure: {:.2f}%'.format(100 * F1))
















