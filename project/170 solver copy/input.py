import networkx as nx
import parse
import numpy as np


if __name__ == '__main__':
    G = nx.Graph()

    for i in range(30):
        G.add_node(i)

    for i in range(30):
        for j in range(i+1, 30):
            G.add_edge(i, j, weight=np.around(float(100) * np.random.rand(), decimals=3))

    print([[u, v, wt] for (u, v, wt) in G.edges.data('weight')])
    dir = r"C:\Users\antho\Desktop\170\project-sp21-skeleton-master\samples\30.in"
    parse.write_input_file(G, dir)