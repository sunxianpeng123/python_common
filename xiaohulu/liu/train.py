# encoding: utf-8

"""
@author: sunxianpeng
@file: train.py
@time: 2021/7/6 15:31
"""
from sklearn.linear_model import LogisticRegression


class Main():
    def __init__(self):
        pass

    from sklearn.linear_model import LogisticRegression
    from sklearn.externals import joblib

    def trainClassifier_sag():
        frTrain = open('data_0403/anchorTraining.txt')
        frTest = open('data_0403/anchorTest.txt')
        trainingSet = [];
        trainingLabels = []
        testSet = [];
        testLabels = []
        for line in frTrain.readlines():
            currLine = line.strip().split('\t')
            lineArr = []
            for i in range(len(currLine) - 1):
                lineArr.append(float(currLine[i]))
            trainingSet.append(lineArr)
            trainingLabels.append(float(currLine[-1]))
        for line in frTest.readlines():
            currLine = line.strip().split('\t')
            lineArr = []
            for i in range(len(currLine) - 1):
                lineArr.append(float(currLine[i]))
            testSet.append(lineArr)
            testLabels.append(float(currLine[-1]))
        classifier = LogisticRegression(solver='sag', max_iter=5000).fit(trainingSet, trainingLabels)
        test_accurcy = classifier.score(testSet, testLabels) * 100
        print('正确率:%f%%' % test_accurcy)

        return classifier

    classifier = trainClassifier_sag()  # 训练分类器
    # joblib.dump(classifier, "data_0403/classifier.m")  # 保存分类器
    #测试就用这个
    # classifier.predict_proba(testdata)

if __name__ == '__main__':
    pass