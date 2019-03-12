import graphviz

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
            #XToKeep = []
            #yToKeep = []
            #for _ in range(len(X)):
            #    chosenDataPoint = random.randint(0, len(X) - 1)
            #    XToKeep.append(X[chosenDataPoint])
            #    yToKeep.append(y[chosenDataPoint])

            self.induction(XToKeep, yToKeep, decisionTree)
            self.pruning(X, y, decisionTree)

    def predict(self, X):
        if not isinstance(X, collections.Iterable):
            raise TypeError("X should be a list of features or a list of lists of features!")
        
        if not isinstance(X[0], collections.Iterable):
                X = [X]

        predictions = []
        for x in X:
            prediction = 0
            amtOfDecisionTrees = len(self.decisionTrees)
            for decisionTree in self.decisionTrees:
                node = decisionTree
                while node.left is not None or node.right is not None:
                    if x[node.value[0]] <= node.value[1]:
                        node = node.left
                    else:
                        node = node.right
                
                if node.value:
                    prediction += node.value[0]
                else:
    	            amtOfDecisionTrees -= 1
            
            predictions.append(prediction / amtOfDecisionTrees)
        return predictions if 1 < len(predictions) else predictions[0]

    def visualize(self, name="random", header=None):
        dot = graphviz.Digraph(engine="dot", node_attr={'shape': 'record', 'height': '.1'})

        dot.attr(packmode="array")
        for i in range(len(self.decisionTrees)):
            with dot.subgraph(name="tree" + str(i)) as d:
                self.decisionTrees[i].visualize(name + str(i), d, lambda value : self.valueToText(header, value))

        dot.render(name, view=True, cleanup=True)
