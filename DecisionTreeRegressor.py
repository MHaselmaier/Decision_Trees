import numpy as np

import math
import statistics
from random import randint

from Tree import Tree


class DecisionTreeRegressor:
    decisionTree = Tree()

    def fit(self, X, y, node=None):
        (feature, splittingPoint) = self.findBestSplittingPoint(X, y)
        
        leftX, leftY, rightX, rightY = self.splitData(X, y, feature, splittingPoint)

        if None == node:
            node = self.decisionTree

        if 5 <= len(leftX) and 5 <= len(rightX):
            node.value = (feature, splittingPoint)
            node.left = Tree(parent=node)
            self.fit(leftX, leftY, node.left)
            node.right = Tree(parent=node)
            self.fit(rightX, rightY, node.right)
        else:
            node.value = statistics.mean(leftY + rightY)


    def findBestSplittingPoint(self, X, y):
        featuresSortedByY = self.sortFeaturesByY(X, y)
        possibleSplittingPointsPerFeature = self.calculatePossibleSplittingPointsPerFeature(featuresSortedByY)
        splittingPointSTDReductions = self.calculateSplittingPointSTDReductions(possibleSplittingPointsPerFeature, X, y)

        print()
        print(featuresSortedByY)
        print(possibleSplittingPointsPerFeature)
        print(splittingPointSTDReductions)
        print()
        
        return self.bestSplittingPoint(splittingPointSTDReductions, possibleSplittingPointsPerFeature)

    def sortFeaturesByY(self, X, y):
        featuresSortedByY = []
        
        flippedX = [list(x) for x in zip(*X)]
        for entry in flippedX:
            featureValueMappedToY = []
            for i in range(len(entry)):
                featureValueMappedToY.append((y[i], entry[i]))
            featuresSortedByY.append([x[1] for x in sorted(featureValueMappedToY)])

        return featuresSortedByY

    def calculatePossibleSplittingPointsPerFeature(self, featuresSortedByY):
        possibleSplittingPointsPerFeature = []

        for feature in featuresSortedByY:
            possibleSplittingPoints = []
            for i in range(10, 91, 10):
                possibleSplittingPoints.append(np.percentile(feature, i))
            possibleSplittingPointsPerFeature.append(possibleSplittingPoints)

        return possibleSplittingPointsPerFeature

    def calculateSplittingPointSTDReductions(self, possibleSplittingPointsPerFeature, X, y):
        splittingPointSTDReductions = []

        std = statistics.stdev(y)

        for i in range(len(possibleSplittingPointsPerFeature)):
            splittingPointSTDReduction = []

            for splittingPoint in possibleSplittingPointsPerFeature[i]:
                leftX, leftY, rightX, rightY = self.splitData(X, y, i, splittingPoint)

                if 2 > len(leftX) or 2 > len(rightX):
                    splittingPointSTDReduction.append(0)
                    continue
                
                leftSTD = statistics.stdev(leftY)
                rightSTD = statistics.stdev(rightY)

                newSTD = (len(leftX) / len(X)) * leftSTD + (len(rightX) / len(X)) * rightSTD
                splittingPointSTDReduction.append(std - newSTD)
            splittingPointSTDReductions.append(splittingPointSTDReduction)

        return splittingPointSTDReductions

    def splitData(self, X, y, feature, splittingPoint):
        leftX = []
        leftY = []
        rightX = []
        rightY = []

        for i in range(len(X)):
            if X[i][feature] <= splittingPoint:
                leftX.append(X[i])
                leftY.append(y[i])
            else:
                rightX.append(X[i])
                rightY.append(y[i])

        return leftX, leftY, rightX, rightY

    def bestSplittingPoint(self, splittingPointSTDReductions, possibleSplittingPointsPerFeature):
        bestSplittingPoints = []

        maxSTDReduction = -math.inf
        for i in range(len(splittingPointSTDReductions)):
            for j in range(len(splittingPointSTDReductions[i])):
                if maxSTDReduction > splittingPointSTDReductions[i][j]:
                    continue
                if maxSTDReduction < splittingPointSTDReductions[i][j]:
                    maxSTDReduction = splittingPointSTDReductions[i][j]
                    bestSplittingPoints.clear()
                
                bestSplittingPoints.append((i, possibleSplittingPointsPerFeature[i][j]))

        return bestSplittingPoints[randint(0, len(bestSplittingPoints) - 1)]

    def predict(self, x):
        node = self.decisionTree
        while node.left is not None or node.right is not None:
            if x[node.value[0]] <= node.value[1]:
                node = node.left
            else:
                node = node.right
        return node.value



X = [[0], [1], [2], [3], [4], [5], [6], [7], [8], [9], 
     [10], [11], [12], [13], [14], [15], [16], [17], [18], [19], 
     [20], [21], [22], [23], [24], [25], [26], [27], [28], [29],
     [30], [31], [32], [33], [34], [35], [36], [37], [38], [39]]
y = [1 for _ in range(10)] + [2 for _ in range(10)] + [3 for _ in range(10)] + [4 for _ in range(10)]

regressor = DecisionTreeRegressor()
regressor.fit(X, y)

print(regressor.predict([[2]]))
print(regressor.predict([[15]]))
print(regressor.predict([[18]]))
print(regressor.predict([[25]]))

regressor.decisionTree.visualize()