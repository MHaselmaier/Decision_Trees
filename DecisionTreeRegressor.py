import collections
import statistics
import math

from DecisionTree import DecisionTree
from Tree import Tree


class DecisionTreeRegressor(DecisionTree):
    def __init__(self, minSamples=5):
        super.__init__(minSamples=minSamples)
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
            while node.left is not None or node.right is not None:
                if x[node.value[0]] <= node.value[1]:
                    node = node.left
                else:
                    node = node.right
            predictions.append(node.value)
        return predictions if 1 < len(predictions) else predictions[0]
