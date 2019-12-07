# encoding: utf-8

"""
@author: sunxianpeng
@file: least_square_method.py
@time: 2017/11/10 17:55
"""

from sklearn import linear_model
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
import numpy as np
class Main():
    def __init__(self):
        """1、最小二乘的系数估计依赖于模型特征项的独立性。当特征项相关并且设计矩阵X 的列近似的线性相关时，
        设计矩阵便接近于一个奇异矩阵，因此最小二乘估计对观测点中的随机误差变得高度敏感，产生大的方差。"""
        pass
    def ordinary_least_squares(self):
        """ 1、最小二乘的系数估计依赖于模型特征项的独立性。当特征项相关并且设计矩阵X 的列近似的线性相关时，
        设计矩阵便接近于一个奇异矩阵，因此最小二乘估计对观测点中的随机误差变得高度敏感，产生大的方差。"""
        reg = linear_model.LinearRegression()
        reg.fit([[0, 0], [1, 1], [2, 2]], [0, 1, 2])
        print reg.coef_  # LinearRegression类的成员函数 fit 以数组X和y为输入，并将线性模型的系数 w 存储在其成员变量coef_ 中
        print "======================================="

    def ridge_regression(self):
        """岭回归通过对系数的大小施加惩罚来解决 普通最小二乘 的一些问题。 ridge coefficients ( 岭系数 ) 最小化一个带罚项的残差平方和"""
        reg = linear_model.Ridge(alpha=0.5)
        reg.fit([[0, 0], [0, 0], [1, 1]], [0, 0.1, 1])
        print reg.coef_  # LinearRegression类的成员函数 fit 以数组X和y为输入，并将线性模型的系数 w 存储在其成员变量coef_ 中
        print reg.intercept_
        print "======================================="

    def ridge_complexity(self):
        """RidgeCV 通过内置的 Alpha 参数的交叉验证来实现岭回归,该对象的工作方式与 GridSearchCV 相同，只是它默认为
        Generalized Cross-Validation ( 通用交叉验证 ) （GCV），这是一种有效的留一交叉验证法："""
        reg=linear_model.RidgeCV([0.1,1.0,10.0])
        reg.fit([[0, 0], [0, 0], [1, 1]], [0, 0.1, 1])
        print reg.alpha_
        print reg.predict([2,2])
        print "======================================="
    def lasso(self):
        """Lasso 是估计稀疏系数的线性模型。它在一些情况下是有用的，因为它倾向于使用具有较少参数值的解决方案，有效地减少给定解决方案所依赖的变量的数量。"""
        reg=linear_model.Lasso(alpha=0.1)
        reg.fit([[0, 0], [1, 1]], [0, 1])
        print reg.intercept_
        print reg.coef_
        print reg.predict([[1,1]])
        print "======================================="

    def bayesian_regression(self):
        X = [[0., 0.], [1., 1.], [2., 2.], [3., 3.]]
        Y = [0., 1., 2., 3.]
        reg = linear_model.BayesianRidge()
        reg.fit(X, Y)
        print reg.predict([[1, 0.]])
        print  reg.coef_
        print "======================================="


    def polynomial_regression_not_boolean(self):
        X = np.arange(6).reshape(3, 2)
        # poly = PolynomialFeatures(degree=2)
        # poly.fit_transform(X)
        model=Pipeline([("poly",PolynomialFeatures(degree=3)),
                        ("linear",linear_model.LinearRegression(fit_intercept=False))])
        x=np.arange(5)
        y = 3 - 2 * x + x ** 2 - x ** 3
        model = model.fit(x[:, np.newaxis], y)
        print  model.named_steps['linear'].coef_
        print "======================================="

    def polynomial_regression_boolean(self):
        X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
        y = X[:, 0] ^ X[:, 1]
        X = PolynomialFeatures(interaction_only=True).fit_transform(X).astype(int)
        clf = linear_model.Perceptron(fit_intercept=False, n_iter=10, shuffle=False).fit(X, y)
        print clf.predict(X)
        print clf.score(X, y)
        print "======================================="



if __name__ == '__main__':
    m=Main()
    m.ordinary_least_squares()
    m.ridge_regression()
    m.ridge_complexity()
    m.lasso()
    m.bayesian_regression()
    m.polynomial_regression_not_boolean()
    m.polynomial_regression_boolean()