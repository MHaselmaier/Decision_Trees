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

    def visualize(self, name="tree", dot=None, valueToText=lambda value : str(value)):
        render = dot is None
        if dot is None:
            dot = graphviz.Digraph(engine="dot", node_attr={'shape': 'record', 'height': '.1'})
        
        dot.node(name, graphviz.nohtml("<f0> |<f1> " + valueToText(self.value) + " |<f2>"))
        if self.left:
            self.left.visualize(name + "l", dot, valueToText)
            dot.edge(name + ":f0", name + "l:f1", "True")
        if self.right:
            self.right.visualize(name + "r", dot, valueToText)
            dot.edge(name + ":f2", name + "r:f1", "False")
        
        if render:
            dot.render(name, view=True, cleanup=True)
