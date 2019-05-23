import networkx as nx

def jaccard_wt(graph, node):
    """
    The weighted jaccard score, defined above.
    Args:
      graph....a networkx graph
      node.....a node to score potential new edges for.
    Returns:
      A list of ((node, ni), score) tuples, representing the 
                score assigned to edge (node, ni)
                (note the edge order)
    """
    #nodes which are not neighbors
    n_neighbors = [n for n in graph.nodes() if n not in set(graph.neighbors(node)) and n != node]
    j_wt = []
    for n in n_neighbors:
        J = float(sum(1/graph.degree(i) for i in (set(graph.neighbors(node)) & set(graph.neighbors(n))))) / (float(1/(sum(graph.degree(i) for i in set(graph.neighbors(node))))) + float(1/(sum(graph.degree(i) for i in set(graph.neighbors(n))))))        
        j_wt.append(((node, n), J)) 
    return sorted(j_wt, key = lambda x : x[1], reverse = True) 