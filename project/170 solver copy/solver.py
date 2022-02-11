import networkx as nx
from parse import read_input_file, write_output_file
from utils import is_valid_solution, calculate_score
import sys
from os.path import basename, normpath
import glob
import heapq
import os
import copy
import random
#from Queue import PriorityQueue

def remove_edge(G, i, j):
    G.remove_edge(i, j)
    if nx.is_connected(G):
        return G
    return False

def remove_node(G, i):
    G.remove_node(i)
    if nx.is_connected(G):
        return G
    return False

def remove_edges(G, edges):
    G.remove_edge_from(edges)
    if nx.is_connected(G):
        return G
    return False

def remove_nodes(G, nodes):
    G.remove_node_from(nodes)
    if nx.is_connected(G):
        return G
    return False

def min_cut(G, size):
    cut_value, partition = nx.minimum_cut(G, 0, size-1)
    return cut_value, partition

'''def try_remove(pack, size, num_c=1, num_k=15):
    #score = pack[0]
    lst = pack[1]
    #G = lst[0]
    c = lst[1]
    k = lst[2]
    if len(c) >= num_c and len(k) >= num_k:
        return pack
    elif len(c) >= num_c:
        return try_edge(pack, size)
    else:
        return try_node(pack, size)'''



def try_node_random(pack, size, datum):
    lst = pack[3]
    G = lst[0]
    C = lst[1]
    K = lst[2]
    path = nx.dijkstra_path(G, 0, size-1)
    if len(path) < 3:
        return None

    index = random.randint(1, len(path)-2)
    G = remove_node(G, path[index])

    if G:
        value = nx.dijkstra_path_length(G, 0, size - 1)
        diff = value - datum
        C.append(path[index])
        opts = len(C) + len(K)
        new_pack = (diff, diff/opts, random.random(), [G, C, K])
        return new_pack

    return None

def try_node_1(pack, size, datum):
    lst = pack[3]
    G = lst[0]
    C = lst[1]
    K = lst[2]
    path = nx.dijkstra_path(G, 0, size-1)
    if len(path) < 3:
        return None
    G = remove_node(G, path[1])
    if G:
        value = nx.dijkstra_path_length(G, 0, size - 1)
        diff = value - datum
        C.append(path[1])
        opts = len(C) + len(K)
        new_pack = (diff, diff/opts, random.random(), [G, C, K])
        return new_pack

    return None


def try_edge_cut(pack, size, datum):
    lst = pack[3]
    G = lst[0]
    C = lst[1]
    K = lst[2]
    path = nx.dijkstra_path(G, 0, size-1)
    path_edges = []

    for i in range(len(path)-1):
        path_edges.append((path[i], path[i+1]))
    min_cut_set = list(nx.connectivity.minimum_edge_cut(G, 0, size - 1))

    edge_to_remove = None
    for i in path_edges:
        if i in min_cut_set:
            edge_to_remove = i
            break

    if edge_to_remove:
        G = remove_edge(G, edge_to_remove[0], edge_to_remove[1])
        if G:
            value = nx.dijkstra_path_length(G, 0, size - 1)
            diff = value - datum
            K.append(edge_to_remove)
            opts = len(C) + len(K)
            new_pack = (diff, diff/opts, random.random(), [G, C, K])
            return new_pack

    return None

'''
def try_edge_cut_2(pack, size, datum):
    lst = pack[3]
    G = lst[0]
    C = lst[1]
    K = lst[2]
    path = nx.dijkstra_path(G, 0, size-1)
    path_edges = []

    for i in range(len(path)-1):
        path_edges.append((path[i], path[i+1]))
    min_cut_set = list(nx.connectivity.minimum_st_edge_cut(G, 0, size - 1))

    edge_to_remove = None
    for i in path_edges:
        if i in min_cut_set:
            edge_to_remove = i
            min_cut_set.remove(i)
            break

    if edge_to_remove:

        weight = float('inf')
        min_edge = None
        for i in min_cut_set:
            temp = G.edges[i[0], i[1]]['weight']
            if temp < weight:
                weight = temp
                min_edge = i
        G = remove_edge(G, edge_to_remove[0], edge_to_remove[1])

        if G and min_edge:
            G = remove_edge(G, min_edge[0], min_edge[1])

        if G:
            value = nx.dijkstra_path_length(G, 0, size - 1)
            diff = value - datum
            K.append(edge_to_remove)
            opts = len(C) + len(K)
            new_pack = (diff, diff/opts, random.random(), [G, C, K])
            return new_pack

    return None
'''

def try_edge_random(pack, size, datum):
    lst = pack[3]
    G = lst[0]
    C = lst[1]
    K = lst[2]
    path = nx.dijkstra_path(G, 0, size-1)
    path_edges = []
    for i in range(len(path)-1):
        path_edges.append((path[i], path[i+1]))

    edge_to_remove = path_edges[random.randint(0, len(path_edges)-1)]

    G = remove_edge(G, edge_to_remove[0], edge_to_remove[1])

    if G:
        value = nx.dijkstra_path_length(G, 0, size - 1)
        diff = value - datum
        K.append(edge_to_remove)
        opts = len(C) + len(K)
        new_pack = (diff, diff/opts, random.random(), [G, C, K])
        return new_pack

    return None

'''
def try_edge_lightest(pack, size, datum):
    lst = pack[3]
    G = lst[0]
    C = lst[1]
    K = lst[2]
    path = nx.dijkstra_path(G, 0, size-1)
    path_edges = []
    weight = float('inf')
    index = 0
    for i in range(len(path)-1):
        path_edges.append((path[i], path[i+1]))
        if G.edges[path[i], path[i+1]]['weight'] < weight:
            weight = G.edges[path[i], path[i+1]]['weight']
            index = i

    edge_to_remove = path_edges[index]

    G = remove_edge(G, edge_to_remove[0], edge_to_remove[1])

    if G:
        value = nx.dijkstra_path_length(G, 0, size - 1)
        diff = value - datum
        K.append(edge_to_remove)
        opts = len(C) + len(K)
        new_pack = (diff, diff/opts, random.random(), [G, C, K])
        return new_pack

    return None
'''



def is_finished(pack, num_c=1, num_k=15):
    lst = pack[3]
    C = lst[1]
    K = lst[2]
    return num_c - len(C), num_k - len(K)

def clean_graph(G):
    edges = list(G.edges)
    print(len(edges))
    for i in edges:
        if i[0] == i[1]:
            G.remove_edge(i[0], i[1])
    print(len(list(G.edges)))
    return G


def solve(G):
    """
    Args:
        G: networkx.Graph
    Returns:
        c: list of cities to remove
        k: list of edges to remove
    """
    size = G.number_of_nodes()
    datum = nx.dijkstra_path_length(G, 0, size - 1)
    pq = []
    G_prime = copy.deepcopy(G)
    G_prime = clean_graph(G_prime)
    cache = (0, 0, 0, [G_prime, [], []])
    heapq.heappush(pq, cache)
    count = 0

    while pq:
        if len(pq) <= 32:
            for i in range(len(pq)):
                cache = heapq.heappop(pq)
                c, k = is_finished(cache)

                if c > 0:
                    tup1 = try_node_random(copy.deepcopy(cache), size, datum)
                    tup2 = try_node_1(copy.deepcopy(cache), size, datum)

                #if k > 1:
                    #tup5 = try_edge_cut_2(copy.deepcopy(cache), size, datum)

                if k > 0:
                    tup3 = try_edge_cut(copy.deepcopy(cache), size, datum)
                    tup4 = try_edge_random(copy.deepcopy(cache), size, datum)

                if tup1:
                    heapq.heappush(pq, tup1)
                    count += 1
                if tup2:
                    heapq.heappush(pq, tup2)
                    count += 1
                if tup3:
                    heapq.heappush(pq, tup3)
                    count += 1
                if tup4:
                    heapq.heappush(pq, tup4)
                    count += 1
                #if tup5:
                    #heapq.heappush(pq, tup5)
                    #count += 1

                tup1, tup2, tup3, tup4, tup5 = None, None, None, None, None

        elif len(pq) > 32:
            while len(pq) > 32:
                heapq.heappop(pq)

        if count > 3000:
            while pq:
                cache = heapq.heappop(pq)

    C = cache[3][1]
    K = cache[3][2]
    print(count)
    return C, K
    pass

small_dir = r"C:\Users\antho\Desktop\170\project\inputs\small"
graph_small = os.listdir(small_dir)

med_dir = r"C:\Users\antho\Desktop\170\project\test input\medium"
graph_med = os.listdir(med_dir)

large_dir = r"C:\Users\antho\Desktop\170\project\test input\large"
graph_large = os.listdir(large_dir)


if __name__ == '__main__':
    for small_g in graph_small:
        input_path = small_dir + '/' + small_g
        G = read_input_file(input_path)
        c, k = solve(G)
        assert is_valid_solution(G, c, k)
        print("Shortest Path Difference: {}".format(calculate_score(G, c, k)))
        output_dir = r"C:\Users\antho\Desktop\170\project\outputs\small"
        write_output_file(G, c, k, output_dir + "//" + small_g[:-3] + '.out')


# Here's an example of how to run your solver.

# Usage: python3 solver.py test.in
'''
if __name__ == '__main__':
    assert len(sys.argv) == 2
    path = sys.argv[1]
    G = read_input_file(path)
    c, k = solve(G)
    assert is_valid_solution(G, c, k)
    print("Shortest Path Difference: {}".format(calculate_score(G, c, k)))
    write_output_file(G, c, k, 'outputs/small-1.out')'''


# For testing a folder of inputs to create a folder of outputs, you can use glob (need to import it)
# if __name__ == '__main__':
#     inputs = glob.glob('inputs/*')
#     for input_path in inputs:
#         output_path = 'outputs/' + basename(normpath(input_path))[:-3] + '.out'
#         G = read_input_file(input_path)
#         c, k = solve(G)
#         assert is_valid_solution(G, c, k)
#         distance = calculate_score(G, c, k)
#         write_output_file(G, c, k, output_path)
