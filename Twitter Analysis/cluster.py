#!/usr/bin/env python
# coding: utf-8

# In[5]:


import networkx as nx
import matplotlib.pyplot as plt

def read_graph():
    return nx.read_edgelist('users.txt', delimiter=':')

def partition_girvan_newman(graph, depth=0):
    if graph.order() ==1 :
        return [graph.nodes()]
    
    def find_best_edge(graph):
        betweenness = nx.edge_betweenness_centrality(graph)
        return sorted(betweenness.items(), key=lambda x: x[1], reverse=True)[0][0]

    components = [c for c in nx.connected_component_subgraphs(graph)]
    indent = '   ' * depth  
    while len(components) <= 2:
        edge_to_remove = find_best_edge(graph)
        graph.remove_edge(*edge_to_remove)
        components = [c for c in nx.connected_component_subgraphs(graph)]
    result = [c for c in components]
    return result

def write_data(graph, clusters):
    file = open("clusters.txt","w")
    file.write('Original graph has %d nodes and %d edges' % 
          (graph.order(), graph.number_of_edges()))
    file.write('\nNumber of Communities discovered: %d' % len(clusters))
    i=0
    for c in clusters:
        file.write('\ncommunity %d has %d nodes' % ((i+1), c.order()))
        i+=1
    file.write('\nAverage number of users per community: %.2f' %(graph.order()/len(clusters)))
    file.close()
    
def draw_network(graph,filename):
    plt.figure(figsize=(22,14))
    plt.axis('off')
    screen_names =['paulpogba','MarcusRashford','JesseLingard','AnthonyMartial','LukeShaw23','AnderHerrera','D_DeGea']
    nodelbl = {lbl: lbl for lbl in screen_names}
    nx.networkx.draw_networkx(graph, node_color='red', alpha =.5,width = .5,node_size = 100, edge_color ='black', labels = nodelbl)
    plt.savefig(filename)
    plt.show()

def main():
    graph = read_graph()
    graph_copy = graph.copy()
    print('Original graph has %d nodes and %d edges' % 
          (graph.order(), graph.number_of_edges()))
    clusters = partition_girvan_newman(graph_copy)
    print('Communities discovered: %d' % len(clusters))
    i=0
    for c in clusters:
        print('cluster %d has %d nodes' % ((i+1), c.order()))
        i+=1
    print('Average number of users per community: %.2f' %(graph.order()/len(clusters)))
    write_data(graph, clusters)
    print("Answers written in cluster_answers.txt")
    print("Orinal Graph:")
    draw_network(graph,"Original Cluster.png")
    print("Graph after applying community Detection algorithm: ")
    draw_network(graph_copy,"Communities.png")

if __name__ == "__main__":
    main()


# In[ ]:




