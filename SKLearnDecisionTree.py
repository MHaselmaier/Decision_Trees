import json
import graphviz

import sklearn
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor

from DecisionTree import DecisionTree


class SKLearnDecisionTree(DecisionTree):
    def __init__(self, decisionTree):
        if (not isinstance(decisionTree, DecisionTreeRegressor) and
            not isinstance(decisionTree, GradientBoostingRegressor) and
            not isinstance(decisionTree, RandomForestRegressor)):
            raise ValueError("Unsupported sklearn tree!\n" +
                "Please provide DecisionTreeRegressor, GradientBoostingRegressor or RandomForestRegressor.")
        self.decisionTree = decisionTree

    def fit(self, X, y):
        self.decisionTree.fit(X, y)

    def prune(self, X, y):
        return

    def predict(self, X):
        return self.decisionTree.predict(X)

    def visualize(self, header=None, name="sklearn"):
        def tree_to_graph(tree, node_id, name, dot):
            feature = ""
            if header is not None:
                feature = header[tree.feature[node_id]]
            else:
                feature = tree.feature[node_id]
            
            prediction = tree.value[node_id]
            if tree.n_outputs == 1:
                prediction = prediction[0, :]
            if isinstance(self.decisionTree, sklearn.ensemble.GradientBoostingRegressor):
                prediction = ', '.join([str(x * self.decisionTree.learning_rate) for x in prediction])
            else:
                prediction = ', '.join([str(x) for x in prediction])

            if tree.children_left[node_id] == sklearn.tree._tree.TREE_LEAF:
                value = self.decisionTree.criterion + ": " + str(tree.impurity[node_id]) + \
                    "\\nsamples: " + str(tree.n_node_samples[node_id]) + "\\nvalue: " + str(prediction)
                dot.node(name, graphviz.nohtml("<f0> |<f1> " + value + " |<f2>"))
                return
            
            value = feature + "\\n\\<= " + str(tree.threshold[node_id]) + "\\n" + self.decisionTree.criterion + \
                ": " + str(tree.impurity[node_id]) +  "\\nsamples: " + str(tree.n_node_samples[node_id]) + "\\nvalue: " + str(prediction)
            dot.node(name, graphviz.nohtml("<f0> |<f1> " + value + " |<f2>"))

            left_child = tree.children_left[node_id]
            tree_to_graph(tree, left_child, name + "l", dot)
            dot.edge(name + ":f0", name + "l:f1", "True")

            right_child = tree.children_right[node_id]
            tree_to_graph(tree, right_child, name + "r", dot)
            dot.edge(name + ":f2", name + "r:f1", "False")

        dot = graphviz.Digraph(engine="dot", node_attr={'shape': 'record', 'height': '.1'})

        if isinstance(self.decisionTree, sklearn.tree.DecisionTreeRegressor):
            tree_to_graph(self.decisionTree.tree_, 0, name, dot)
        elif isinstance(self.decisionTree, sklearn.ensemble.GradientBoostingRegressor):
            dot.attr(packmode="array")

            with dot.subgraph(name="tree_mean") as d:
                value = self.decisionTree.criterion + ": NaN" + "\\nsamples: " + \
                    str(self.decisionTree.estimators_[0,0].tree_.n_node_samples[0]) + "\\nvalue: " + str(self.decisionTree.init_.mean)
                d.node("tree_mean0", graphviz.nohtml("<f0> |<f1> " + value + " |<f2>"))

            estimators = self.decisionTree.estimators_.flatten()
            for i, tree in enumerate(estimators):
                with dot.subgraph(name="tree" + str(i)) as d:
                    tree_to_graph(tree.tree_, 0, name + str(i), d)
        elif isinstance(self.decisionTree, sklearn.ensemble.RandomForestRegressor):
            dot.attr(packmode="array")

            for i, tree in enumerate(self.decisionTree.estimators_):
                with dot.subgraph(name="tree" + str(i)) as d:
                    tree_to_graph(tree.tree_, 0, name + str(i), d)
        
        dot.render(name, view=True, cleanup=True)

    def toJSON(self, header, name="sklearn"): 
        def node_to_str(tree, node_id):
            value = tree.value[node_id]
            if tree.n_outputs == 1:
                value = value[0, :]
        
            jsonValue = ','.join([str(x) for x in value])
            if isinstance(self.decisionTree, sklearn.ensemble.GradientBoostingRegressor):
                jsonValue = ','.join([str(x * self.decisionTree.learning_rate) for x in value])
        
            if tree.children_left[node_id] == sklearn.tree._tree.TREE_LEAF:
                return '"prediction":%s' % (jsonValue)
            else:
                return '"variable":"%s","splittingPoint":%s' % (header[tree.feature[node_id]], tree.threshold[node_id])
        
        def recurse(tree, node_id):
            left_child = tree.children_left[node_id]
            right_child = tree.children_right[node_id]
        
            js = "{" + node_to_str(tree, node_id)
        
            if left_child != sklearn.tree._tree.TREE_LEAF:
                js = js + ',"children":[' + \
                    recurse(tree, left_child) + "," + \
                    recurse(tree, right_child) + ']'
            
            return js + "}"
        
        js = ""
        if isinstance(self.decisionTree, sklearn.tree.DecisionTreeRegressor):
            js = js + '{"trees":[' + recurse(self.decisionTree.tree_, 0) + "],"\
                        '"type":"DecisionTreeRegressor"}'
        elif isinstance(self.decisionTree, sklearn.ensemble.GradientBoostingRegressor):
            js = js + '{"trees":[' +\
                '{"prediction":%s},' % (self.decisionTree.init_.mean)
            estimators = self.decisionTree.estimators_.flatten()
            for tree in estimators:
                    js = js + recurse(tree.tree_, 0)
                    if tree != estimators[-1]:
                        js = js + ","
            js = js + '],"type":"BoostedDecisionTreeRegressor"}'
        elif isinstance(self.decisionTree, sklearn.ensemble.RandomForestRegressor):
            js = js + '{"trees":['
            for tree in self.decisionTree.estimators_:
                js = js + recurse(tree.tree_, 0)
                if tree != self.decisionTree.estimators_[-1]:
                    js = js + ","
            js = js + '],"type":"RandomForestRegressor"}'
        
        with open(name + ".json", "w", encoding="utf-8") as f:
            json.dump(json.loads(js), f, ensure_ascii=False, separators=(',', ':'))
