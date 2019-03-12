import graphviz
import matplotlib.pyplot as plt

import random
import math
import csv

from DecisionTreeRegressor import DecisionTreeRegressor
from RandomForestRegressor import RandomForestRegressor
from BoostedDecisionTreeRegressor import BoostedDecisionTreeRegressor

with open("dataset.csv", encoding="UTF-8") as inputFile:
    reader = csv.reader(inputFile)

    header = next(reader)

    X = [row for row in reader]
    #random.shuffle(X)
    for row in X:
        for i in range(len(row)):
            row[i] = float(row[i])
    y = [row.pop() for row in X]

    trainingSetSize = int(len(X) * 0.8)
    X_train = X[:trainingSetSize]
    y_train = y[:trainingSetSize]
    X_test = X[trainingSetSize:]
    y_test = y[trainingSetSize:]

fig = plt.figure()

if True:
    regressor = DecisionTreeRegressor(minSamples=10)
    regressor.fit(X_train, y_train)
    print("Regressor:", regressor.validate(X_test, y_test))
    regressor.visualize(header=header)
    ax1 = fig.add_subplot(131)
    ax1.plot([0, 1000], [0, 1000], "k-")
    ax1.plot(y_train, regressor.predict(X_train), "b.")
    ax1.plot(y_test, regressor.predict(X_test), "r.")
    ax1.set_title("Regressor")
    ax1.set_xlabel("True Values for 50%-DF")
    ax1.set_ylabel("Predicted Values for 50%-DF")

if False:
    boostedRegressor5 = BoostedDecisionTreeRegressor(usedTrees=100)
    boostedRegressor5.fit(X_train, y_train)
    print("Boosted Regressor (5):", boostedRegressor5.validate(X_test, y_test))
    boostedRegressor5.visualize(header=header)
    ax1 = fig.add_subplot(132)
    ax1.plot([0, 1000], [0, 1000], "k-")
    ax1.plot(y_train, boostedRegressor5.predict(X_train), "b.")
    ax1.plot(y_test, boostedRegressor5.predict(X_test), "r.")
    ax1.set_title("Boosted Regressor")
    ax1.set_xlabel("True Values for 50%-DF")
    ax1.set_ylabel("Predicted Values for 50%-DF")

if False:
    randomRegressor5 = RandomForestRegressor(usedTrees=100)
    randomRegressor5.fit(X_train, y_train)
    print("Random Regressor (5):", randomRegressor5.validate(X_test, y_test))
    randomRegressor5.visualize(header=header)
    ax1 = fig.add_subplot(133)
    ax1.plot([0, 1000], [0, 1000], "k-")
    ax1.plot(y_train, randomRegressor5.predict(X_train), "b.")
    ax1.plot(y_test, randomRegressor5.predict(X_test), "r.")
    ax1.set_title("Random Regressor")
    ax1.set_xlabel("True Values for 50%-DF")
    ax1.set_ylabel("Predicted Values for 50%-DF")

plt.tight_layout()
plt.show()