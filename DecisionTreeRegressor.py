import numpy as np

import math
import statistics
from random import randint

from Tree import Tree


class DecisionTreeRegressor:
    decisionTree = Tree()

    def fit(self, X, y, node=None):
        self.induction(X, y, self.decisionTree)
        self.pruning(X, y, self.decisionTree)
    
    def induction(self, X, y, node):
        (feature, splittingPoint) = self.findBestSplittingPoint(X, y)
        
        leftX, leftY, rightX, rightY = self.splitData(X, y, feature, splittingPoint)

        if None == node:
            node = self.decisionTree

        if 5 <= len(leftX) and 5 <= len(rightX):
            node.value = (feature, splittingPoint)
            node.left = Tree(parent=node)
            self.induction(leftX, leftY, node.left)
            node.right = Tree(parent=node)
            self.induction(rightX, rightY, node.right)
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

        possibleSplittingPointsPerFeature += self.percentileSplittingPoints(5, featuresSortedByY)

        return possibleSplittingPointsPerFeature

    def percentileSplittingPoints(self, stepsize, featuresSortedByY):
        if 0 >= stepsize or 100 <= stepsize:
            raise ValueError("stepsize must be in interval [1,99]")
        
        possibleSplittingPointsPerFeature = []

        for feature in featuresSortedByY:
            possibleSplittingPoints = []
            for i in range(stepsize, 100 - stepsize + 1, stepsize):
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

    def pruning(self, X, y, node):
        if node is None:
            return

        self.pruning(X, y, node.left)
        self.pruning(X, y, node.right)

        if node.left is None and node.right is None:
            savedParentValue = node.parent.value
            savedParentLeft = node.parent.left
            savedParentRight = node.parent.right

            errorBeforePruning = self.validate(X, y)
            xAtParent, yAtParent = self.calculateXyAtNode(X, y, node.parent)
            node.parent.left = node.parent.right = None
            node.parent.value = statistics.mean(yAtParent)
            errorAfterPrunning = self.validate(X, y)

            if errorBeforePruning < errorAfterPrunning:
                node.parent.value = savedParentValue
                node.parent.left = savedParentLeft
                node.parent.right = savedParentRight
            return

    def calculateXyAtNode(self, X, y, node, currentNode=None):
        nodes = [node]
        currentNode = node
        while currentNode.parent is not None:
            nodes.append(currentNode.parent)
            currentNode = currentNode.parent

        currentNode = nodes.pop()
        nextNode = currentNode
        while currentNode != node:
            nextNode = nodes.pop()

            newX = []
            newY = []
            for i in range(len(X)):
                if nextNode == currentNode.left and X[i][currentNode.value[0]] <= currentNode.value[1]:
                    newX.append(X[i])
                    newY.append(y[i])
                elif nextNode == currentNode.right and X[i][currentNode.value[0]] > currentNode.value[1]:
                    newX.append(X[i])
                    newY.append(y[i])
            X = newX
            y = newY

            currentNode = nextNode

        return X, y

    def predict(self, x):
        node = self.decisionTree
        while node.left is not None or node.right is not None:
            if x[node.value[0]] <= node.value[1]:
                node = node.left
            else:
                node = node.right
        return node.value
    
    def validate(self, X, y):
        sumOfSquaredErrors = 0
        for i in range(len(X)):
            sumOfSquaredErrors += math.pow(y[i] - self.predict(X[i]), 2)
        return sumOfSquaredErrors / len(X)


X = [[i + 10 * j] for j in range(10) for i in range(10)] + [[i + 10 * j] for j in range(10) for i in range(10)] + [[i + 10 * j] for j in range(10) for i in range(10)] + [[i + 10 * j] for j in range(10) for i in range(10)] + [[i + 10 * j] for j in range(10) for i in range(10)]
y = [i // 10 for i in range(100)] + [i // 10 for i in range(100)] + [i // 10 for i in range(100)] + [i // 10 for i in range(100)] + [i // 10 for i in range(100)]

regressor = DecisionTreeRegressor()
regressor.fit(X, y)
regressor.decisionTree.visualize()