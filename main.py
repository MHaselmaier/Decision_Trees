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
from sklearn.tree import export_graphviz 


with open("dataset.csv", encoding="UTF-8") as inputFile:
    reader = csv.reader(inputFile)

    header = next(reader)

    X = [row for row in reader]
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

fig = plt.figure()
fig.set_figheight(2 * 5)
fig.set_figwidth(3 * 5)

if True:
    regressor = DecisionTreeRegressor(minSamples=5)
    regressor.fit(X_train, y_train)
    #print("Regressor:", regressor.validate(X_test, y_test))
    #regressor.visualize(header=header)

    sklearnRegressor = tree.DecisionTreeRegressor(min_samples_split=5)
    sklearnRegressor.fit(X_train, y_train)
    export_graphviz(sklearnRegressor, out_file ='tree.dot') 

    regressorValidation = statistics.mean([abs(yActual - yPredict) for (yActual, yPredict) in zip(y_test, regressor.predict(X_test))])
    sklearnRegressorValidation = statistics.mean([abs(yActual - yPredict) for (yActual, yPredict) in zip(y_test, sklearnRegressor.predict(X_test))])
    print(regressorValidation, sklearnRegressorValidation)

    ax1 = fig.add_subplot(231)
    ax1.plot([0, 1000], [0, 1000], "k-")
    ax1.plot(y_train, regressor.predict(X_train), "b.")
    ax1.plot(y_test, regressor.predict(X_test), "r.")
    ax1.set_title("Regressor \n(" + str(regressorValidation) + ")")
    ax1.set_xlabel("True Values for 50%-DF")
    ax1.set_ylabel("Predicted Values for 50%-DF")

    ax1 = fig.add_subplot(234)
    ax1.plot([0, 1000], [0, 1000], "k-")
    ax1.plot(y_train, sklearnRegressor.predict(X_train), "b.")
    ax1.plot(y_test, sklearnRegressor.predict(X_test), "r.")
    ax1.set_title("Scikit-Learn Regressor \n(" + str(sklearnRegressorValidation) + ")")
    ax1.set_xlabel("True Values for 50%-DF")
    ax1.set_ylabel("Predicted Values for 50%-DF") 


if True:
    boostedRegressor5 = BoostedDecisionTreeRegressor(usedTrees=50)
    boostedRegressor5.fit(X_train, y_train)
    #print("Boosted Regressor (5):", boostedRegressor5.validate(X_test, y_test))
    #boostedRegressor5.visualize(header=header)

    sklearnBoostedRegressor = GradientBoostingRegressor(min_samples_split=5, n_estimators=50)
    sklearnBoostedRegressor.fit(X_train, y_train)

    regressorValidation = statistics.mean([abs(yActual - yPredict) for (yActual, yPredict) in zip(y_test, boostedRegressor5.predict(X_test))])
    sklearnRegressorValidation = statistics.mean([abs(yActual - yPredict) for (yActual, yPredict) in zip(y_test, sklearnBoostedRegressor.predict(X_test))])
    print(regressorValidation, sklearnRegressorValidation)

    ax1 = fig.add_subplot(232)
    ax1.plot([0, 1000], [0, 1000], "k-")
    ax1.plot(y_train, boostedRegressor5.predict(X_train), "b.")
    ax1.plot(y_test, boostedRegressor5.predict(X_test), "r.")
    ax1.set_title("Boosted Regressor \n(" + str(regressorValidation) + ")")
    ax1.set_xlabel("True Values for 50%-DF")
    ax1.set_ylabel("Predicted Values for 50%-DF")

    ax1 = fig.add_subplot(235)
    ax1.plot([0, 1000], [0, 1000], "k-")
    ax1.plot(y_train, sklearnBoostedRegressor.predict(X_train), "b.")
    ax1.plot(y_test, sklearnBoostedRegressor.predict(X_test), "r.")
    ax1.set_title("Scikit-Learn Boosted Regressor \n(" + str(sklearnRegressorValidation) + ")")
    ax1.set_xlabel("True Values for 50%-DF")
    ax1.set_ylabel("Predicted Values for 50%-DF")

if True:
    randomRegressor5 = RandomForestRegressor(usedTrees=50)
    randomRegressor5.fit(X_train, y_train)
    #print("Random Regressor (5):", randomRegressor5.validate(X_test, y_test))
    #randomRegressor5.visualize(header=header)

    sklearnRandomForestRegressor = ensemble.RandomForestRegressor(min_samples_split=5, n_estimators=50)
    sklearnRandomForestRegressor.fit(X_train, y_train)

    regressorValidation = statistics.mean([abs(yActual - yPredict) for (yActual, yPredict) in zip(y_test, randomRegressor5.predict(X_test))])
    sklearnRegressorValidation = statistics.mean([abs(yActual - yPredict) for (yActual, yPredict) in zip(y_test, sklearnRandomForestRegressor.predict(X_test))])
    print(regressorValidation, sklearnRegressorValidation)

    ax1 = fig.add_subplot(233)
    ax1.plot([0, 1000], [0, 1000], "k-")
    ax1.plot(y_train, randomRegressor5.predict(X_train), "b.")
    ax1.plot(y_test, randomRegressor5.predict(X_test), "r.")
    ax1.set_title("Random Regressor \n(" + str(regressorValidation) + ")")
    ax1.set_xlabel("True Values for 50%-DF")
    ax1.set_ylabel("Predicted Values for 50%-DF")

    ax1 = fig.add_subplot(236)
    ax1.plot([0, 1000], [0, 1000], "k-")
    ax1.plot(y_train, sklearnRandomForestRegressor.predict(X_train), "b.")
    ax1.plot(y_test, sklearnRandomForestRegressor.predict(X_test), "r.")
    ax1.set_title("Scikit-Learn Random Regressor \n(" + str(sklearnRegressorValidation) + ")")
    ax1.set_xlabel("True Values for 50%-DF")
    ax1.set_ylabel("Predicted Values for 50%-DF")

plt.tight_layout()
plt.show()