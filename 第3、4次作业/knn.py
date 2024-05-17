import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.neighbors import KNeighborsClassifier
from sklearn import datasets
# 导入鸢尾花数据
iris = datasets.load_iris()
# 这里使用鸢尾花的前两个特征的数据作为训练集
X = iris.data[:, :2]
# 定义标签
y = iris.target
# 创建一个包含不同k值的列表
k_list = [1,2, 3, 5]
h = .02
# 创建不同颜色的画布
cmap_light = ListedColormap(['orange', 'cyan', 'cornflowerblue'])
cmap_bold = ListedColormap(['darkorange', 'c', 'darkblue'])

plt.figure(figsize=(15,14))
# 根据不同的k值进行可视化
for ind,k in enumerate(k_list):
    clf = KNeighborsClassifier(k)  # 定义knn分类模型，并设定k值
    clf.fit(X, y)                  # 将定义好的模型在训练集上进行训练
    # 画出决策边界
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1  # +1， -1 是为了在每幅图的四边留出一定的区域，让图更美观
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    # 在x和y的方向上，以h=0.02为间隔进行划分
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    
    Z = Z.reshape(xx.shape)

    plt.subplot(321+ind)  
    # 根据边界填充颜色
    plt.pcolormesh(xx, yy, Z, cmap=cmap_light)
    # 将训练数据点可视化到画布
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap_bold,
                edgecolor='k', s=20)
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.title("3-Class classification (k = %i)"% k)

plt.show()
