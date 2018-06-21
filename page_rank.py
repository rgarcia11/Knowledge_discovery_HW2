"""
Computes page rank
"""
import sys
sys.setrecursionlimit(1500)

def s(vi, graph, alpha, p, convergence, last_page_rank, n=None):
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
    return alpha*sum(graph.get_edge(vj,vi)/sum(graph.get_edge(vj,vk) for vk in graph.graph[vj])*last_page_rank[vj] for vj in graph.graph[vi])+(1-alpha)*p[vi]

def page_rank(graph,alpha,convergence,p=None):
    p = {}
    page_rank = {}
    last_page_rank = {}
    for node in graph.graph:
        p[node] = 1/len(graph.graph)
        page_rank[node] = 1/len(graph.graph)
        last_page_rank[node] = 1/len(graph.graph)
    for iteration in range(0,convergence):
        for vi in graph.graph:
            page_rank[vi] = s(vi, graph, alpha, p, convergence, last_page_rank)
        #normalization and updating steps
        total_weight = sum(page_rank[vi]**2 for vi in page_rank)**(1/2)
        for vi in page_rank:
            page_rank[vi] = page_rank[vi] / total_weight
            last_page_rank[vi] = page_rank[vi]
    return page_rank
