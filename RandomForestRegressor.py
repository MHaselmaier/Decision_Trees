import collections
import random

from RandomDecisionTree import RandomDecisionTree
from Tree import Tree


class RandomForestRegressor(RandomDecisionTree):
    def __init__(self, minSamples=5, usedTrees=5, percentageOfDatapointsPerTree=0.7):
        super().__init__(minSamples=5)
        self.decisionTrees = [Tree() for _ in range(usedTrees)]
        self.percentageOfDatapointsPerTree = percentageOfDatapointsPerTree

    def fit(self, X, y):
        for decisionTree in self.decisionTrees:
            datapointsToKeep = random.sample(range(len(y)), int(len(y) * self.percentageOfDatapointsPerTree))
            XToKeep = []
            yToKeep = []
            for i in datapointsToKeep:
                XToKeep.append(X[i])
                yToKeep.append(y[i])

            self.induction(XToKeep, yToKeep, decisionTree)
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
                node = decisionTree
                while node.left is not None or node.right is not None:
                    if x[node.value[0]] <= node.value[1]:
                        node = node.left
                    else:
                        node = node.right
                
                if prediction is None:
                    prediction = node.value
                elif node.value is not None:
                    prediction += node.value
            
            if prediction is not None:
                prediction = prediction / len(self.decisionTrees)

            predictions.append(prediction)
        return predictions if 1 < len(predictions) else predictions[0]
