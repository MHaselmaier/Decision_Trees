import statistics
import math

from Tree import Tree
from DecisionTree import DecisionTree


class DecisionTreeRegressor(DecisionTree):
    decisionTree = Tree()

    def fit(self, X, y):
        self.induction(X, y, self.decisionTree)
        self.pruning(X, y, self.decisionTree)

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
            _, yAtParent = self.calculateXyAtNode(X, y, node.parent)
            node.parent.left = node.parent.right = None
            node.parent.value = statistics.mean(yAtParent)
            errorAfterPrunning = self.validate(X, y)

            if errorBeforePruning < errorAfterPrunning:
                node.parent.value = savedParentValue
                node.parent.left = savedParentLeft
                node.parent.right = savedParentRight
            return

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