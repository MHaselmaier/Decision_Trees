import numpy as np

from abc import ABC, abstractmethod
import random
import statistics
import math

from Tree import Tree


class DecisionTree(ABC):
    def __init__(self, minSamples=5):
        self.minSamples = minSamples

    @abstractmethod
    def fit(self, X, y):
        pass
    
    def induction(self, X, y, node):
        (feature, splittingPoint) = self.findBestSplittingPoint(X, y)
        
        leftX, leftY, rightX, rightY = self.splitData(X, y, feature, splittingPoint)

        if self.minSamples <= len(leftX) and self.minSamples <= len(rightX):
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
        splittingPointErrors = self.calculateSplittingPointErrors(possibleSplittingPointsPerFeature, X, y)
        return self.bestSplittingPoint(splittingPointErrors, possibleSplittingPointsPerFeature)

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

    def calculateSplittingPointErrors(self, possibleSplittingPointsPerFeature, X, y):
        splittingPointErrors = []

        for i in range(len(possibleSplittingPointsPerFeature)):
            splittingPointError = []

            for splittingPoint in possibleSplittingPointsPerFeature[i]:
                leftX, leftY, rightX, rightY = self.splitData(X, y, i, splittingPoint)

                if 2 > len(leftX) or 2 > len(rightX):
                    splittingPointError.append(math.inf)
                    continue
                
                leftYMean = statistics.mean(leftY)
                rightYMean = statistics.mean(rightY)

                error = 0
                for prediction in leftY:
                    error += math.pow(prediction - leftYMean, 2)
                for prediction in rightY:
                    error += math.pow(prediction - rightYMean, 2)

                splittingPointError.append(error)
            splittingPointErrors.append(splittingPointError)

        return splittingPointErrors

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

    def bestSplittingPoint(self, splittingPointErrors, possibleSplittingPointsPerFeature):
        bestSplittingPoints = []

        minError = math.inf
        for i in range(len(splittingPointErrors)):
            for j in range(len(splittingPointErrors[i])):
                if minError < splittingPointErrors[i][j]:
                    continue
                if minError > splittingPointErrors[i][j]:
                    minError = splittingPointErrors[i][j]
                    bestSplittingPoints.clear()
                
                bestSplittingPoints.append((i, possibleSplittingPointsPerFeature[i][j]))

        return bestSplittingPoints[random.randint(0, len(bestSplittingPoints) - 1)]

    def pruning(self, X, y, node):
        if node is None:
            return

        self.pruning(X, y, node.left)
        self.pruning(X, y, node.right)

        if node.left is None and node.right is None and node.parent is not None:
            savedParentValue = node.parent.value
            savedParentLeft = node.parent.left
            savedParentRight = node.parent.right

            errorBeforePruning = self.validate(X, y)
            _, yAtParent = self.calculateXyAtNode(X, y, node.parent)
            node.parent.left = node.parent.right = None
            node.parent.value = statistics.mean(yAtParent)
            errorAfterPrunning = self.validate(X, y)

            if errorBeforePruning < errorAfterPrunning:
                node.parent.value = savedParentValue
                node.parent.left = savedParentLeft
                node.parent.right = savedParentRight
            return

    def calculateXyAtNode(self, X, y, node):
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

    @abstractmethod
    def predict(self, X):
        pass

    def validate(self, X, y):
        sumOfSquaredErrors = 0
        for i in range(len(X)):
            sumOfSquaredErrors += math.pow(y[i] - self.predict(X[i]), 2)
        return sumOfSquaredErrors / len(X)
