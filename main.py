import graphviz
import matplotlib.pyplot as plt

import numpy as np
import math
import csv
import statistics

from DecisionTreeRegressor import DecisionTreeRegressor
from RandomForestRegressor import RandomForestRegressor
from BoostedDecisionTreeRegressor import BoostedDecisionTreeRegressor
from SKLearnDecisionTree import SKLearnDecisionTree

from sklearn import tree
from sklearn import ensemble


def bin(X, n):
    min = math.inf
    max = -math.inf
    for x in X:
        if min > x[-1]:
            min = x[-1]
        if max < x[-1]:
            max = x[-1]

    binnedData = [[] for i in range(n)]
    bins = np.arange(min + (max - min) / n, max, (max - min) / n)
    for x in X:
        for i in range(n):
            if x[-1] <= bins[i]:
                binnedData[i].append(x)
                break
            if i == len(bins) - 1:
                binnedData[len(bins)].append(x)
                break
    
    return binnedData

def loadData():
    with open("dataset.csv", encoding="utf-8") as inputFile:
        reader = csv.reader(inputFile)

        header = next(reader)

        X = [row for row in reader]
        for row in X:
            for i in range(len(row)):
                row[i] = float(row[i])

    train_X = []
    train_y = []
    test_X = []
    test_y = []
    validate_X = []
    validate_y = []
    for b in bin(X, 10):
        amtTrain = int(max(1, len(b) * 0.7))
        amtTest = int(max(1, len(b) * 0.3))
        amtValidation = 0#int(max(1, len(b) * 0.2))
        amtTrain += len(b) - amtTrain - amtTest - amtValidation

        np.random.shuffle(b)
        for i in range(len(b)):
            if (i < amtTrain):
                train_y.append(b[i].pop())
                train_X.append(b[i])
            elif (i < amtTrain + amtTest):
                test_y.append(b[i].pop())
                test_X.append(b[i])
            elif (i < amtTrain + amtTest + amtValidation):
                validate_y.append(b[i].pop())
                validate_X.append(b[i])

    return train_X, train_y, test_X, test_y, validate_X, validate_y, header


def plot(regressor, name, X_train, y_train, X_test, y_test):
    plt.scatter(regressor.predict(X_train), y_train, marker=".", c="b")
    plt.scatter(regressor.predict(X_test), y_test, marker=".", c="r")
    plt.savefig(name + ".svg")
    plt.clf()


X_train, y_train, X_test, y_test, X_validate, y_validate, header = loadData()

usedTrees = 200

maxDepth = 10
regressors = [
    ("SK Learn Regressor", SKLearnDecisionTree(tree.DecisionTreeRegressor(min_samples_split=5, random_state=np.random))),
    ("Regressor", DecisionTreeRegressor()),
    ("SK Learn Boosted", SKLearnDecisionTree(ensemble.GradientBoostingRegressor(n_estimators=usedTrees, min_samples_split=5, max_depth=maxDepth, learning_rate=0.05, random_state=np.random))),
    ("Boosted Regressor", BoostedDecisionTreeRegressor(usedTrees=usedTrees, maxDepth=maxDepth)),
    ("SK Learn Random", SKLearnDecisionTree(ensemble.RandomForestRegressor(n_estimators=usedTrees, min_samples_split=5, max_depth=maxDepth, random_state=np.random))),
    ("Random Regressor", RandomForestRegressor(usedTrees=usedTrees, maxDepth=maxDepth))
]

print(len(header))

for name, regressor in regressors:
    regressor.fit(X_train, y_train)
    regressor.prune(X_train, y_train)
    print(name, ":\t", regressor.validate(X_test, y_test))
    regressor.toJSON(header=header, name=name)
    #regressor.visualize(header=header, name=name)
    plot(regressor, name, X_train, y_train, X_test, y_test)


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