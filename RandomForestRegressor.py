import graphviz

import collections
import random
import json

from RandomDecisionTree import RandomDecisionTree
from Tree import Tree


class RandomForestRegressor(RandomDecisionTree):
    def __init__(self, minSamples=5, maxDepth=5, usedTrees=5, percentageOfDatapointsPerTree=0.7):
        super().__init__(minSamples=minSamples, maxDepth=maxDepth)
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

    def visualize(self, header=None, name="random"):
        dot = graphviz.Digraph(engine="dot", node_attr={'shape': 'record', 'height': '.1'})

        dot.attr(packmode="array")
        for i, tree in enumerate(self.decisionTrees):
            with dot.subgraph(name="tree" + str(i)) as d:
                tree.visualize(name + str(i), d, lambda value : self.valueToText(header, value))

        dot.render(name, view=True, cleanup=True)
    
    def toJSON(self, header, name="random"):
        trees = []
        for decisionTree in self.decisionTrees:
            trees.append(json.loads(decisionTree.toJSON(valueToJSON=lambda value: self.valueToJSON(header, value))))
        
        with open(name + ".json", "w", encoding="utf-8") as f:
            json.dump({"type": "RandomForestRegressor", "trees": trees}, f, ensure_ascii=False, separators=(',', ':'))
