import graphviz

class Tree:
    value = None
    __parent = None
    __left = None
    __right = None

    def __init__(self, value=None, parent=None, left=None, right=None):
        self.value = value
        self.parent = parent
        self.left = left
        self.right = right

    @property
    def parent(self):
        return self.__parent

    @parent.setter
    def parent(self, parent):
        if isinstance(parent, Tree) or parent is None:
            self.__parent = parent
        else:
            raise TypeError("parent child should be of type Tree!")

    @property
    def left(self):
        return self.__left
    
    @left.setter
    def left(self, left):
        if isinstance(left, Tree) or left is None:
            self.__left = left
        else:
            raise TypeError("left child should be of type Tree!")

    @property
    def right(self):
        return self.__right
    
    @right.setter
    def right(self, right):
        if isinstance(right, Tree) or right is None:
            self.__right = right
        else:
            raise TypeError("right child should be of type Tree!")

    def visualize(self, name="tree", dot=None, nodeID=0):
        render = dot is None
        if dot is None:
            dot = graphviz.Digraph('g', node_attr={'shape': 'record', 'height': '.1'})
        
        nodes = [(self, nodeID)]
        dot.node(str(nodeID), graphviz.nohtml("<f0> |<f1> " + str(self.value) + "|<f2>"))
        nodeID += 1
        while 0 < len(nodes):
            currentNode, currentID = nodes.pop()

            if currentNode.left is not None:
                dot.node(str(nodeID), graphviz.nohtml("<f0> |<f1> " + str(currentNode.left.value) + "|<f2>"))
                nodes.append((currentNode.left, nodeID))
                dot.edge(str(currentID) + ":f0", str(nodeID) + ":f1", label="True")
                nodeID += 1
            
            if currentNode.right is not None:
                dot.node(str(nodeID), graphviz.nohtml("<f0> |<f1> " + str(currentNode.right.value) + "|<f2>"))
                nodes.append((currentNode.right, nodeID))
                dot.edge(str(currentID) + ":f2", str(nodeID) + ":f1", label="False")
                nodeID += 1
        
        if render:
            dot.render(name + ".gv", view=True, cleanup=True)
            
        return nodeID
