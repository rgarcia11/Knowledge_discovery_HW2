"""
Computes page rank
"""
import sys
sys.setrecursionlimit(1500)

def s(vi, graph, alpha, p, convergence, page_rank, n=None):
    """
    Recursively computes the equation
    s(vi) = alpha * sum(vj in adjacent nodes of vi, wji/(sum(vk in adjacent nodes of vj)*wjk)*s(vj)+(1-alpha)*pi)
    s is the rank (result of this function)
    alpha is a damping factor
    v is a vertex
    wij is the weight of a node between vertices vi and vj
    pi is the i-eth element of an array p
        p is initialized as 1/n for every element, where n is the number of nodes
        s is initialized as 1/n also
    """
    points_to_vi = graph.graph[vi]
    print(points_to_vi)
    outer_sum = 0
    for vj in points_to_vi:
        

def page_rank(graph,alpha, convergence, p=None):
    p = {}
    page_rank = {}
    for node in graph.graph:
        p[node] = 1/len(graph.graph)
        page_rank[node] = 1/len(graph.graph)
    page_rank_temp = page_rank
    for iteration in range(1,convergence)
    for vi in graph.graph:
        page_rank[vi] = s(vi, graph, alpha, p, convergence, page_rank_temp)
    return page_rank
