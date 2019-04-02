import graphviz
import matplotlib.pyplot as plt

import random
import math
import csv
import statistics

from DecisionTreeRegressor import DecisionTreeRegressor
from RandomForestRegressor import RandomForestRegressor
from BoostedDecisionTreeRegressor import BoostedDecisionTreeRegressor
from SKLearnDecisionTree import SKLearnDecisionTree

from sklearn import tree
from sklearn import ensemble


with open("dataset.csv", encoding="utf-8") as inputFile:
    reader = csv.reader(inputFile)

    header = next(reader)

    X = [row for row in reader]
    random.seed(12345)
    random.shuffle(X)
    for row in X:
        for i in range(len(row)):
            row[i] = float(row[i])
    y = [row.pop() for row in X]

    trainingSetSize = int(len(X) * 0.7)
    X_train = X[:trainingSetSize]
    y_train = y[:trainingSetSize]
    X_test = X[trainingSetSize:]
    y_test = y[trainingSetSize:]

    regressors = [
        ("SK Learn Regressor", SKLearnDecisionTree(tree.DecisionTreeRegressor(min_samples_split=5))),
        ("Regressor", DecisionTreeRegressor()),
        ("SK Learn Boosted", SKLearnDecisionTree(ensemble.GradientBoostingRegressor(n_estimators=10, min_samples_split=5, max_depth=4, learning_rate=0.1))),
        ("Boosted Regressor", BoostedDecisionTreeRegressor(usedTrees=10, maxDepth=4)),
        ("SK Learn Random", SKLearnDecisionTree(ensemble.RandomForestRegressor(n_estimators=10, min_samples_split=5, max_depth=4))),
        ("Random Regressor", RandomForestRegressor(usedTrees=10, maxDepth=4))
    ]

    for name, regressor in regressors:
        regressor.fit(X_train, y_train)
        print(name, ":\t", regressor.validate(X_test, y_test))
        regressor.toJSON(header=header, name=name)
        regressor.visualize(header=header, name=name)


    #predictionValues = []
    #nodes = [regressor.decisionTree]
    #while 0 < len(nodes):
    #    node = nodes.pop()

    #    if not node.left and not node.right:
    #        predictionValues.append(node.value[0])
    #        continue

    #    if node.left:
    #        nodes.append(node.left)
    #    if node.right:
    #        nodes.append(node.right)
    #print(sorted(predictionValues))
    #plt.hist(y, bins=30, zorder=1)
    #plt.scatter(sorted(predictionValues), [0] * len(predictionValues), marker=".", c="r", zorder=2)
    #plt.savefig("LeafValueDistribution.png")
    #plt.show()