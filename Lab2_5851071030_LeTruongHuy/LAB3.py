
# ### Setting up the data (10 points)
# 
# The following is the snippet of code to load the datasets, and split it into train and validation data:

import numpy as np
import matplotlib.pyplot as plt
import mltools as ml
import warnings
# warnings.filterwarnings("ignore")
np.random.seed(0)

# Data Loading
X = np.genfromtxt('data/X_train.txt', delimiter=None)
Y = np.genfromtxt('data/Y_train.txt', delimiter=None)
X,Y = ml.shuffleData(X,Y)

def print_info(X, name):
    for i in range(X.shape[1]):
        print(i + 1)
        print(name + " min is:", np.min(X[:, i]), name + " max is:", np.max(X[:, i]))
        print(name + " mean is:",np.mean(X[:,i]), name + " variance is:",np.var(X[:,i]))

# 1.1
print_info(X, '')

# 1.2
Xtr, Xva, Ytr, Yva = ml.splitData(X, Y)
Xt, Yt = Xtr[:5000], Ytr[:5000]
Xv, Yv = Xva[:2000], Yva[:2000]
XtS, params = ml.rescale(Xt)
XvS, _ = ml.rescale(Xv, params)

print('---XtS-----')

print_info(XtS, 'XtS')

print('---XvS-----')

print_info(XvS, 'XvS')

# ### Linear Classifiers (20 points)

def plot(xlist, tr_auc, va_auc, xname):
    plt.plot(xlist, tr_auc, c='r', label='train')
    plt.plot(xlist, va_auc, c='b', label='validation')
    plt.xlabel(xname)
    plt.ylabel('auc')
    plt.legend()
    plt.show()

def linear_classfier_print(learner, XtS, Yt, XvS, Yv):
    reg = [0.5,1.0,1.5,2.0,2.5,3.0,3.5,4.0] 
    tr_auc = []
    va_auc = []
    for r in reg:
        learner.train(XtS, Yt, reg=r, initStep=0.5, stopTol=1e-6, stopIter=100)
        tr_auc.append(learner.auc(XtS, Yt))
        va_auc.append(learner.auc(XvS, Yv))
    plot(reg, tr_auc, va_auc, 'reg')


# 2.1
learner = ml.linearC.linearClassify()
linear_classfier_print(learner, XtS, Yt, XvS, Yv)

# 2.2
Xt2 = ml.transforms.fpoly(Xt, 2, bias=False)
Xv2 = ml.transforms.fpoly(Xv, 2, bias=False)
print(Xt2.shape[1])

# We originally have 14 features from x1 -> x14
# we pick 2 different from them to combine a new xi * xj feature, it will be 14 * 13 / 2 -> 91
# we convert every feature to its square  x1 -> x1 * x1 , it will be 14

# so, totally 14 + 14 + 91 = 119


# 2.3
XtS2, params = ml.rescale(Xt2)
XvS2, _ = ml.rescale(Xv2, params)
learner_trans = ml.linearC.linearClassify()
linear_classfier_print(learner_trans, XtS2, Yt, XvS2, Yv)

# ###  Nearest Neighbors (20 points)

def nearest_neighbors_print(XtS, Yt, XvS, Yv):
    klist = [1, 5, 10, 50, 100, 200, 400] 
    tr_auc = []
    va_auc = []
    for k in klist:
        learner = ml.knn.knnClassify()
        learner.train(XtS, Yt, K=k, alpha=0.0)
        tr_auc.append(learner.auc(XtS, Yt))
        va_auc.append(learner.auc(XvS, Yv))
        
    plot(klist, tr_auc, va_auc, 'k')

# 3.1
nearest_neighbors_print(XtS, Yt, XvS, Yv)

# 3.2
nearest_neighbors_print(Xt, Yt, Xv, Yv)

# 3.3
def plot_2d_auc(_2d_auc, x_list, y_list):
    f, ax = plt.subplots(1, 1, figsize=(8, 5))
    cax = ax.matshow(_2d_auc, interpolation='nearest')
    f.colorbar(cax)
    ax.set_xticklabels(['']+list(x_list))
    ax.set_yticklabels(['']+list(y_list))
    plt.show()
    
K = range(1,20,4)
A = range(1,5,1) # Or something else
tr_auc = np.zeros((len(K),len(A)))
va_auc = np.zeros((len(K),len(A)))
for i,k in enumerate(K):
    for j,a in enumerate(A):
        learner = ml.knn.knnClassify()
        learner.train(XtS, Yt, K=k, alpha=a)
        tr_auc[i][j] = learner.auc(XtS, Yt)  # train learner using k and a
        va_auc[i][j] = learner.auc(XvS, Yv)
        
plot_2d_auc(tr_auc, A, K)
plot_2d_auc(va_auc, A, K)

# I would recommand the K is 17 and a is 1

# ### Decision Trees (20 points)


# 4.1
depths = range(1,15,3)
tr_auc = []
va_auc = []

for d in depths:
    learner = ml.dtree.treeClassify(XtS, Yt, maxDepth=d)
    tr_auc.append(learner.auc(XtS, Yt))
    va_auc.append(learner.auc(XvS, Yv))

plot(depths, tr_auc, va_auc, 'depths')


# 4.2
node_minParent_2 = []
node_minParent_4 = []

for d in depths:
    learner = ml.dtree.treeClassify(XtS, Yt, minParent=2, maxDepth=d)
    node_minParent_2.append(learner.sz)
    
    learner = ml.dtree.treeClassify(XtS, Yt, minParent=4, maxDepth=d)
    node_minParent_4.append(learner.sz)

plot(depths, node_minParent_2, node_minParent_4, 'depths')


# 4.3
minParents = range(2,10,1)
minLeaves = range(1,10,1)

tr_auc = np.zeros((len(minParents),len(minLeaves)))
va_auc = np.zeros((len(minParents),len(minLeaves)))
for i,p in enumerate(minParents):
    for j,l in enumerate(minLeaves):
        learner = ml.dtree.treeClassify(XtS, Yt, maxDepth=5, minParent=p, minLeaf=l)
        tr_auc[i][j] = learner.auc(XtS, Yt)
        va_auc[i][j] = learner.auc(XvS, Yv)
        
plot_2d_auc(tr_auc, minLeaves, minParents)
plot_2d_auc(va_auc, minLeaves, minParents)

# I would recommand the minParent is 5 and minLeaf is 6

# ### Neural Networks (20 points)


# 5.1    
nodes = range(1,5,1)
layers = range(1,10,1)
tr_auc = np.zeros((len(nodes),len(layers)))
va_auc = np.zeros((len(nodes),len(layers)))
for i,n in enumerate(nodes):
    for j,l in enumerate(layers):
        nn = ml.nnet.nnetClassify()
        nn.init_weights([XtS.shape[1]] + [n for x in range(1,l+1)] + [2], 'random', XtS, Yt)
        nn.train(XtS, Yt, stopTol=1e-8, stepsize=.25, stopIter=100) # 100 is fast to train and have good loss
        tr_auc[i][j] = nn.auc(XtS, Yt)
        va_auc[i][j] = nn.auc(XvS, Yv)

plot_2d_auc(tr_auc, layers, nodes)
plot_2d_auc(va_auc, layers, nodes)

# I would recommand node in each layer is 4 and layer is 1

# 5.2
def sig(z): return np.atleast_2d(z)
def dsig(z): return np.atleast_2d(1)

def activation_switch(name):
    nn = ml.nnet.nnetClassify()
    nn.init_weights([XtS.shape[1],5,2], 'random', XtS, Yt) 
    nn.setActivation(name, sig, dsig)
    nn.train(XtS, Yt, stopTol=1e-8, stepsize=.25, stopIter=300)
    print(name + " trian auc:",nn.auc(XtS,Yt))
    print(name + " validation auc:",nn.auc(XvS,Yv))
    print('--------------------------------------')

activation_switch('custom')
activation_switch('logistic')
activation_switch('htangent')

# In this case, the custom activation is better than the logistic activation and a little worse than the htangent activation.

# ### Conclusions (5 points)

# I prefer the decision tree, my name is Jitao, 
# and my leaderboard is 10 with score 0.72677
Xte = np.genfromtxt('data/X_test.txt', delimiter=None)
learner = ml.dtree.treeClassify(X, Y, maxDepth=30, minParent=10, minLeaf=10)
Yte = np.vstack((np.arange(Xte.shape[0]), learner.predictSoft(Xte)[:,1])).T
np.savetxt('Y_submit.txt', Yte, '%d, %.2f', header='ID,Prob1', comments='', delimiter=',')
print('finished')

# ### Statement of Collaboration (5 points)