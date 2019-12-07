# encoding: utf-8

"""
@author: sunxianpeng
@file: UnsupervisedNN.py
@time: 2017/11/16 11:21
"""

from  sklearn.neighbors import NearestNeighbors
import numpy as np
from sklearn.neighbors import KDTree
from sklearn.neighbors.nearest_centroid import NearestCentroid

class Main():
    def __init__(self):
        pass

    def indoors(self):
        """关于最近邻算法，如果两个 neighbors ( 邻居 )，neighbor 和 neighbor k 具有相同的距离，但是具有不同的 label ( 标签 ) ，结果取决于训练数据的排序。"""
        X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
        #它作为三个不同的最近邻算法的统一接口： BallTree ， KDTree 和基于 sklearn.metrics.pairwise 中的 brute-force
        # algorithm based on routines ( 例程的强力算法 ) 。neighbors search algorithm ( 邻域搜索算法 ) 的选择通过关键字
        # “algorithm” 进行控制，该算法必须是 ['auto'，'ball_tree'，'kd_tree'，'brute'] 之一。
        nbrs = NearestNeighbors(n_neighbors=2, algorithm='ball_tree').fit(X)#
        distances, indices = nbrs.kneighbors(X)#返回距离每个点k个最近的点和距离指数，indices可以理解为表示点的下标，distances为距离
        print indices#个数和x个数相同，下表代表在x中的位置，元素代表x中的位置
        print distances
        print nbrs.kneighbors_graph(X).toarray()

    def kdtree_BallTree_Classes(self):
        X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
        kdt = KDTree(X, leaf_size=30, metric='euclidean')
        print kdt.query(X, k=2, return_distance=False)

    def classify(self):
        X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
        y = np.array([1, 1, 1, 2, 2, 2])
        clf = NearestCentroid()
        clf.fit(X, y)
        print(clf.predict([[-0.8, -1]]))


if __name__ == '__main__':
    m=Main()
    m.indoors()
    m.kdtree_BallTree_Classes()
    m.classify()