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
        feature, splittingPoint = self.findBestSplittingPoint(X, y)
        if feature is None or splittingPoint is None:
            node.left = node.right = None
            node.value = statistics.mean(y), self.calculateMSE(y), len(X)
            return

        leftX, leftY, rightX, rightY = self.splitData(X, y, feature, splittingPoint)
        node.value = feature, splittingPoint, statistics.mean(y), self.calculateMSE(y), len(X)

        #if self.minSamples > len(leftX) and self.minSamples > len(rightX):
         #   node.left = node.right = None
          #  node.value = statistics.mean(y), self.calculateMSE(y), len(X)
           # return

        if self.minSamples > len(leftX):
            node.left = Tree((statistics.mean(leftY), self.calculateMSE(leftY), len(leftX)), node)
        else:
            node.left = Tree(parent=node)
            self.induction(leftX, leftY, node.left)
        if self.minSamples > len(rightX):
            node.right = Tree((statistics.mean(rightY), self.calculateMSE(rightY), len(rightX)), node)
        else:
            node.right = Tree(parent=node)
            self.induction(rightX, rightY, node.right)

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
        possibleSplittingPointsPerFeature = [set() for _ in featuresSortedByY]

        percentileSplittingPoints = self.percentileSplittingPoints(5, featuresSortedByY)
        for i, _ in enumerate(percentileSplittingPoints):
            possibleSplittingPointsPerFeature[i] |= percentileSplittingPoints[i]

        splittingPointsAtEachValue = self.splittingPointsAtEachValue(featuresSortedByY)
        for i, _ in enumerate(splittingPointsAtEachValue):
            possibleSplittingPointsPerFeature[i] |= splittingPointsAtEachValue[i]

        return [list(possibleSplittingPoints) for possibleSplittingPoints in possibleSplittingPointsPerFeature]

    def percentileSplittingPoints(self, stepsize, featuresSortedByY):
        if 0 >= stepsize or 100 <= stepsize:
            raise ValueError("stepsize must be in interval [1,99]")
        
        possibleSplittingPointsPerFeature = []

        for feature in featuresSortedByY:
            possibleSplittingPoints = set()
            for i in range(0, 100 + 1, stepsize):
                possibleSplittingPoints.add(np.percentile(feature, i))
            possibleSplittingPointsPerFeature.append(possibleSplittingPoints)
        
        return possibleSplittingPointsPerFeature
    
    def splittingPointsAtEachValue(self, featuresSortedByY):
        return [set(f) for f in featuresSortedByY]

    def calculateSplittingPointErrors(self, possibleSplittingPointsPerFeature, X, y):
        splittingPointErrors = []

        for i in range(len(possibleSplittingPointsPerFeature)):
            splittingPointError = []

            for splittingPoint in possibleSplittingPointsPerFeature[i]:
                leftX, leftY, rightX, rightY = self.splitData(X, y, i, splittingPoint)

                if 1 > len(leftX) or 1 > len(rightX):
                    splittingPointError.append(float("nan"))
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
        bestSplittingPoints = [(None, None)]

        minError = math.inf
        for i in range(len(splittingPointErrors)):
            for j in range(len(splittingPointErrors[i])):
                if minError < splittingPointErrors[i][j] or math.isnan(splittingPointErrors[i][j]):
                    continue
                if minError > splittingPointErrors[i][j]:
                    minError = splittingPointErrors[i][j]
                    bestSplittingPoints.clear()
                
                bestSplittingPoints.append((i, possibleSplittingPointsPerFeature[i][j]))

        return bestSplittingPoints[random.randint(0, len(bestSplittingPoints) - 1)]

    def calculateMSE(self, y):
        mse = 0

        prediction = statistics.mean(y)
        for i in range(len(y)):
            mse += math.pow(y[i] - prediction, 2)

        return mse / len(y)

    def pruning(self, X, y, node):
        if node is None:
            return

        self.pruning(X, y, node.left)
        self.pruning(X, y, node.right)

        if (node.left is None or node.right is None) and node.parent is not None:
            savedParentValue = node.parent.value
            savedParentLeft = node.parent.left
            savedParentRight = node.parent.right

            errorBeforePruning = self.validate(X, y)
            _, yAtParent = self.calculateXyAtNode(X, y, node.parent)
            node.parent.left = node.parent.right = None
            node.parent.value = statistics.mean(yAtParent), self.calculateMSE(yAtParent), len(yAtParent)
            errorAfterPrunning = self.validate(X, y)

            if errorBeforePruning < errorAfterPrunning:
                node.parent.value = savedParentValue
                node.parent.left = savedParentLeft
                node.parent.right = savedParentRight

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
        # Standard Error of Regression
        distances = [abs(yActual - yPredict) for (yActual, yPredict) in zip(y, self.predict(X))]
        return statistics.mean(distances)
        
        # RMSE:
        #mse = 0
        #for i in range(len(X)):
        #   mse += (math.pow(y[i] - self.predict(X[i]), 2)) / len(X)
        #return math.sqrt(mse)
    
    @abstractmethod
    def visualize(self, name="tree", header=None):
        pass
    
    def valueToText(self, header, value):
        if 3 == len(value):
            return "mse: " + str(value[1]) + "\\nsamples: " + str(value[2]) + "\\nvalue: " + str(value[0])

        return header[value[0]] + "\\n\\<= " + str(value[1]) + "\\nmse: " + str(value[3]) +  "\\nsamples: " + str(value[4]) + "\\nvalue: " + str(value[2])
