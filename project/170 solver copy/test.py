import networkx as nx
from parse import read_input_file, write_output_file
from utils import is_valid_solution, calculate_score
import sys
from os.path import basename, normpath
import glob
import heapq
import os
#from Queue import PriorityQueue

small_dir = r"C:\Users\antho\Desktop\170\project\test input\small"
graph_small = os.listdir(small_dir)

def try_edge(pack, size):
    lst = pack[1]
    G = lst[0]
    c = lst[1]
    k = lst[2]
    print(G.edges())
    G.remove_edge(2, 25)
    G.remove_edge(2, 24)
    G.remove_edge(2, 23)
    G.remove_edge(2, 20)
    G.remove_edge(2, 18)
    G.remove_edge(0, 24)
    G.remove_edge(24, 12)
    G.remove_edge(25, 29)
    path = nx.dijkstra_path(G, 0, size-1)
    path_edges = []
    for i in range(len(path)-1):
        path_edges.append((path[i], path[i+1]))
        print(G.edges[path[i], path[i+1]]['weight'])
    print(path_edges)
    value = nx.dijkstra_path_length(G, 0, size-1)
    print(path)
    print(value)
    a = nx.connectivity.minimum_st_edge_cut(G, 0, size-1)
    print(a)
    b = nx.connectivity.minimum_edge_cut(G, 0, size - 1)
    print(type(b))

if __name__ == '__main__':
    #assert len(sys.argv) == 2
    for small_g in graph_small:
        G = read_input_file(small_dir + '/' + small_g)
        #c, k = solve(G)
        try_edge((0, [G, [], []]), len(G))
        break
        assert is_valid_solution(G, c, k)
        print("Shortest Path Difference: {}".format(calculate_score(G, c, k)))
        write_output_file(G, c, k, 'outputs/small-1.out')