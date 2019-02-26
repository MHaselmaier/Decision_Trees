from abc import ABC, abstractmethod
import collections
import math

from DecisionTree import DecisionTree
from Tree import Tree


class BoostedDecisionTreeRegressor(DecisionTree):
    def __init__(self, usedTrees=5):
        self.decisionTrees = [Tree() for _ in range(usedTrees)]

    def fit(self, X, y):
        currentY = y
        for decisionTree in self.decisionTrees:
            self.induction(X, currentY, decisionTree)
            currentY = self.calculateNewY(X, currentY, decisionTree)
        currentY = y
        for decisionTree in self.decisionTrees:
            self.pruning(X, currentY, decisionTree)
            currentY = self.calculateNewY(X, currentY, decisionTree)
    
    def calculateNewY(self, X, y, decisionTree):
        newY = []
        for i in range(len(X)):
            newY.append(y[i] - self.predictionFromGivenTree(X[i], decisionTree))
        return newY

    def predictionFromGivenTree(self, x, decisionTree):
        node = decisionTree
        while node.left is not None or node.right is not None:
            if x[node.value[0]] <= node.value[1]:
                node = node.left
            else:
                node = node.right
        return node.value

    def predict(self, X):
        if not isinstance(X, collections.Iterable):
            raise TypeError("X should be a list of features or a list of lists of features!")
        
        if not isinstance(X[0], collections.Iterable):
                X = [X]
        
        predictions = []
        for x in X:
            prediction = None
            for decisionTree in self.decisionTrees:
                value = self.predictionFromGivenTree(x, decisionTree)

                if prediction is None:
                    prediction = value
                elif value is not None:
                    prediction += value

            predictions.append(prediction)
        return predictions if 1 < len(predictions) else predictions[0]
