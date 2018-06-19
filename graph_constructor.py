"""
Constructs an undirected graph
"""

class Undirected_graph:
    """
    Class for the directed graph data structure
    """
    def __init__(self):
        """
        Initializes an empty graph.
        """
        self.graph = {}

    def add_node(self, node):
        """
        Adds a node to the graph
        """
        if node not in self.graph:
            self.graph[node] = {}

    def add_edge(self, first_node, second_node, weight):
        """
        Adds an edge to the graph.
        If the edge already exists, it is replaced.
        """
        if first_node not in self.graph:
            add_node(first_node)
        if second_node not in self.graph:
            add_node(second_node)
        self.graph[first_node][second_node] = weight
        self.graph[second_node][first_node] = weight

    def get_edge(self, first_node, second_node):
        """
        Checks whether or not an edge exists between
        two nodes first_node and second_node
        Returns the weight of the edge, or -1 if nonexistant
        """
        if first_node in self.graph:
            if second_node in self.graph[first_node]:
                return self.graph[first_node][second_node]
        return -1
