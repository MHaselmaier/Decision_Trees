import graphviz
import matplotlib.pyplot as plt

import random
import math
import csv
import statistics

from DecisionTreeRegressor import DecisionTreeRegressor
from RandomForestRegressor import RandomForestRegressor
from BoostedDecisionTreeRegressor import BoostedDecisionTreeRegressor

from sklearn import tree
from sklearn.ensemble import GradientBoostingRegressor
from sklearn import ensemble


with open("dataset.csv", encoding="UTF-8") as inputFile:
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

    regressor = DecisionTreeRegressor(minSamples=5)
    regressor.fit(X_train, y_train)
    print("Regressor:", regressor.validate(X_test, y_test))

    boostedRegressor = BoostedDecisionTreeRegressor(usedTrees=5)
    boostedRegressor.fit(X_train, y_train)
    print("Boosted Regressor:", boostedRegressor.validate(X_test, y_test))

    randomRegressor = RandomForestRegressor(usedTrees=5)
    randomRegressor.fit(X_train, y_train)
    print("Random Regressor:", randomRegressor.validate(X_test, y_test))

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