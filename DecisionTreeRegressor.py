import collections
import statistics
import math
import json

from DecisionTree import DecisionTree
from Tree import Tree


class DecisionTreeRegressor(DecisionTree):
    def __init__(self, minSamples=5, maxDepth=math.inf):
        super().__init__(minSamples=minSamples, maxDepth=maxDepth)
        self.decisionTree = Tree()

    def fit(self, X, y):
        self.induction(X, y, self.decisionTree)
        self.pruning(X, y, self.decisionTree)

    def predict(self, X):
        if not isinstance(X, collections.Iterable):
            raise TypeError("X should be a list of features or a list of lists of features!")
        
        if not isinstance(X[0], collections.Iterable):
                X = [X]

        predictions = []
        for x in X:
            node = self.decisionTree
            while node.left or node.right:
                if x[node.value[0]] <= node.value[1]:
                    node = node.left
                else:
                    node = node.right
            predictions.append(node.value[0])
        return predictions if 1 < len(predictions) else predictions[0]

    def visualize(self, header=None, name="regressor"):
        self.decisionTree.visualize(name=name, valueToText=lambda value : self.valueToText(header, value))

    def toJSON(self, header, name="regressor"):
        trees = [json.loads(self.decisionTree.toJSON(valueToJSON=lambda value: self.valueToJSON(header, value)))]
        
        with open(name + ".json", "w", encoding="utf-8") as f:
            json.dump({"type": "DecisionTreeRegressor", "trees": trees}, f, ensure_ascii=False, separators=(',', ':'))

