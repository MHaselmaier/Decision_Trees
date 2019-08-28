import graphviz

from abc import ABC, abstractmethod
import collections
import statistics
import math
import json

from DecisionTree import DecisionTree
from Tree import Tree


class BoostedDecisionTreeRegressor(DecisionTree):
    def __init__(self, minSamples=5, maxDepth=5, usedTrees=5, learningRate=0.05):
        super().__init__(minSamples=minSamples, maxDepth=maxDepth)
        self.decisionTrees = [Tree() for _ in range(usedTrees + 1)]
        self.learningRate = learningRate

    def fit(self, X, y):
        currentY = y
        self.decisionTrees[0].value = (statistics.mean(y), self.calculateMSE(y), len(X))
        currentY = self.calculateNewY(X, y, 1)
        for i, decisionTree in enumerate(self.decisionTrees[1:], 1):
            self.induction(X, currentY, decisionTree)
            currentY = self.calculateNewY(X, y, i + 1)
    
    def calculateNewY(self, X, y, trees):
        newY = []
        for i in range(len(X)):
            prediction = 0
            for j in range(trees):
                prediction += self.predictionFromGivenTree(X[i], self.decisionTrees[j])
            newY.append(self.learningRate * (y[i] - prediction))
        return newY

    def predictionFromGivenTree(self, x, node):
        while node.left or node.right:
            if x[node.value[0]] <= node.value[1]:
                node = node.left
            else:
                node = node.right
        return node.value[0]

    def prune(self, X, y):
        for decisionTree in self.decisionTrees:
            self.pruning(X, y, decisionTree)

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

    def visualize(self, header=None, name="boosted"):
        dot = graphviz.Digraph(engine="dot", node_attr={'shape': 'record', 'height': '.1'})

        dot.attr(packmode="array")
        for i, tree in enumerate(self.decisionTrees):
            with dot.subgraph(name="tree" + str(i)) as d:
                tree.visualize(name + str(i), d, lambda value : self.valueToText(header, value))

        dot.render(name, view=True, cleanup=True)
    
    def toJSON(self, header, name="boosted"):
        trees = []
        for decisionTree in self.decisionTrees:
            trees.append(json.loads(decisionTree.toJSON(valueToJSON=lambda value: self.valueToJSON(header, value))))

        with open(name + ".json", "w", encoding="utf-8") as f:
            json.dump({"type": "BoostedDecisionTreeRegressor", "trees": trees}, f, ensure_ascii=False, separators=(',', ':'))
