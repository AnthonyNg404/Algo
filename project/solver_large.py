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
    node_to_remove = path[index]
    if node_to_remove:

        if G:
            value = nx.dijkstra_path_length(G, 0, size - 1)
            diff = value - datum
            C.append(node_to_remove)
            opts = len(C)*2 + len(K)
            new_pack = (diff, diff/opts, random.random(), [G, C, K])
            return new_pack

    return None

def try_node_first(pack, size, datum):
    lst = pack[3]
    G = lst[0]
    C = lst[1]
    K = lst[2]
    path = nx.dijkstra_path(G, 0, size-1)
    if len(path) < 3:
        return None
    G = remove_node(G, path[1])
    node_to_remove = path[1]
    if node_to_remove:
        if G:
            value = nx.dijkstra_path_length(G, 0, size - 1)
            diff = value - datum
            C.append(node_to_remove)
            opts = len(C)*2 + len(K)
            new_pack = (diff, diff/opts, random.random(), [G, C, K])
            return new_pack

    return None

def try_node_last(pack, size, datum):
    lst = pack[3]
    G = lst[0]
    C = lst[1]
    K = lst[2]
    path = nx.dijkstra_path(G, 0, size-1)
    if len(path) < 3:
        return None
    G = remove_node(G, path[-2])
    node_to_remove = path[-2]
    if node_to_remove:
        if G:
            value = nx.dijkstra_path_length(G, 0, size - 1)
            diff = value - datum
            C.append(node_to_remove)
            opts = len(C)*2 + len(K)
            new_pack = (diff, diff/opts, random.random(), [G, C, K])
            return new_pack

    return None

def try_node_gain(pack, size, datum):
    lst = pack[3]
    G = lst[0]
    C = lst[1]
    K = lst[2]
    path = nx.dijkstra_path(G, 0, size-1)

    if len(path) < 3:
        return None

    path_edges = []
    for i in range(len(path)-1):
        path_edges.append((path[i], path[i+1]))

    node_to_remove = None
    dis_gain = float('-inf')

    for i in range(len(path_edges)-1):
        len_i = G.edges[path_edges[i][0], path_edges[i][1]]['weight'] + G.edges[path_edges[i+1][0], path_edges[i+1][1]]['weight']
        paths_len = []

        for j in nx.edge_disjoint_paths(G, path_edges[i][0], path_edges[i+1][1]):
            path_len = 0

            for k in range(len(j)-1):
                path_len += G.edges[j[k], j[k+1]]['weight']
            paths_len.append(path_len)

        if len(paths_len)==1:
            continue

        paths_len.sort()
        paths_len = paths_len[1:]
        dis_gain_i = min(paths_len) - len_i

        if dis_gain_i > dis_gain:
            dis_gain = dis_gain_i
            node_to_remove = path_edges[i][1]

    if node_to_remove:
        G = remove_node(G, node_to_remove)
        if G:
            value = nx.dijkstra_path_length(G, 0, size - 1)
            diff = value - datum
            C.append(node_to_remove)
            opts = len(C)*2 + len(K)
            new_pack = (diff, diff/opts, random.random(), [G, C, K])
            return new_pack

    return None


def try_node_lightest(pack, size, datum):
    lst = pack[3]
    G = lst[0]
    C = lst[1]
    K = lst[2]
    path = nx.dijkstra_path(G, 0, size-1)

    if len(path) < 3:
        return None

    path_edges = []
    weight = float('inf')
    index = 0
    for i in range(len(path) - 1):
        path_edges.append((path[i], path[i + 1]))
        if G.edges[path[i], path[i + 1]]['weight'] < weight:
            weight = G.edges[path[i], path[i + 1]]['weight']
            index = i

    edge_to_remove = path_edges[index]

    if edge_to_remove[0] == 0:
        G = remove_node(G, edge_to_remove[1])
        node_to_remove = edge_to_remove[1]
    elif edge_to_remove[1] == size-1:
        G = remove_node(G, edge_to_remove[0])
        node_to_remove = edge_to_remove[0]
    else:
        if G[path[path.index(edge_to_remove[0])-1]][edge_to_remove[0]]['weight'] < G[edge_to_remove[1]][path[path.index(edge_to_remove[1])+1]]['weight']:
            G = remove_node(G, edge_to_remove[0])
            node_to_remove = edge_to_remove[0]
        else:
            G = remove_node(G, edge_to_remove[1])
            node_to_remove = edge_to_remove[1]

    if G:
        value = nx.dijkstra_path_length(G, 0, size - 1)
        diff = value - datum
        C.append(node_to_remove)
        opts = len(C)*2 + len(K)
        new_pack = (diff, diff/opts, random.random(), [G, C, K])
        return new_pack

    return None


def try_node_cut(pack, size, datum):
    lst = pack[3]
    G = lst[0]
    C = lst[1]
    K = lst[2]
    path = nx.dijkstra_path(G, 0, size-1)

    if len(path) < 3:
        return None

    path_edges = []
    for i in range(len(path)-1):
        path_edges.append((path[i], path[i+1]))

    min_cut_set = list(nx.connectivity.minimum_edge_cut(G, 0, size - 1))
    edge_to_remove = None
    for i in path_edges:
        if i in min_cut_set:
            edge_to_remove = i
            break

    if edge_to_remove[0] == 0:
        G = remove_node(G, edge_to_remove[1])
        node_to_remove = edge_to_remove[1]
    elif edge_to_remove[1] == size-1:
        G = remove_node(G, edge_to_remove[0])
        node_to_remove = edge_to_remove[0]
    else:
        if G[path[path.index(edge_to_remove[0])-1]][edge_to_remove[0]]['weight'] < G[edge_to_remove[1]][path[path.index(edge_to_remove[1])+1]]['weight']:
            G = remove_node(G, edge_to_remove[0])
            node_to_remove = edge_to_remove[0]
        else:
            G = remove_node(G, edge_to_remove[1])
            node_to_remove = edge_to_remove[0]

    if G:
        value = nx.dijkstra_path_length(G, 0, size - 1)
        diff = value - datum
        C.append(node_to_remove)
        opts = len(C)*2 + len(K)
        new_pack = (diff, diff/opts, random.random(), [G, C, K])
        return new_pack

    return None


def try_edge_gain(pack, size, datum):
    lst = pack[3]
    G = lst[0]
    C = lst[1]
    K = lst[2]
    path = nx.dijkstra_path(G, 0, size-1)
    path_edges = []
    for i in range(len(path)-1):
        path_edges.append((path[i], path[i+1]))

    edge_to_remove = None
    dis_gain = float('-inf')
    for i in range(len(path_edges)):
        len_i = G.edges[path_edges[i][0], path_edges[i][1]]['weight']
        paths_len = []
        for j in nx.edge_disjoint_paths(G, path_edges[i][0], path_edges[i][1]):
            path_len = 0
            for k in range(len(j)-1):
                path_len += G.edges[j[k], j[k+1]]['weight']
            paths_len.append(path_len)
        if len(paths_len)==1:
            continue
        paths_len.sort()
        paths_len = paths_len[1:]
        dis_gain_i = min(paths_len) - len_i
        if dis_gain_i > dis_gain:
            dis_gain = dis_gain_i
            edge_to_remove = path_edges[i]

    if edge_to_remove:
        G = remove_edge(G, edge_to_remove[0], edge_to_remove[1])
        if G:
            value = nx.dijkstra_path_length(G, 0, size - 1)
            diff = value - datum
            K.append(edge_to_remove)
            opts = len(C)*2 + len(K)
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
            opts = len(C)*2 + len(K)
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
        second_edge = None
        neighbor = list(G.neighbors(edge_to_remove[0]))
        edge_set = [(edge_to_remove[0], j) for j in neighbor if j != edge_to_remove[1]]

        for i in edge_set:
            temp = G.edges[i[0], i[1]]['weight']
            if temp < weight:
                weight = temp
                second_edge = i

        G = remove_edge(G, edge_to_remove[0], edge_to_remove[1])
        if G and second_edge:
            G = remove_edge(G, second_edge[0], second_edge[1])

        if G:
            value = nx.dijkstra_path_length(G, 0, size - 1)
            diff = value - datum
            K.append(edge_to_remove)
            K.append(second_edge)
            opts = len(C)*2 + len(K)
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
        opts = len(C)*2 + len(K)
        new_pack = (diff, diff/opts, random.random(), [G, C, K])
        return new_pack

    return None

def try_edge_first(pack, size, datum):
    lst = pack[3]
    G = lst[0]
    C = lst[1]
    K = lst[2]
    path = nx.dijkstra_path(G, 0, size-1)
    path_edges = []
    for i in range(len(path)-1):
        path_edges.append((path[i], path[i+1]))

    edge_to_remove = path_edges[0]

    G = remove_edge(G, edge_to_remove[0], edge_to_remove[1])

    if G:
        value = nx.dijkstra_path_length(G, 0, size - 1)
        diff = value - datum
        K.append(edge_to_remove)
        opts = len(C)*2 + len(K)
        new_pack = (diff, diff/opts, random.random(), [G, C, K])
        return new_pack

    return None

def try_edge_last(pack, size, datum):
    lst = pack[3]
    G = lst[0]
    C = lst[1]
    K = lst[2]
    path = nx.dijkstra_path(G, 0, size-1)
    path_edges = []
    for i in range(len(path)-1):
        path_edges.append((path[i], path[i+1]))

    edge_to_remove = path_edges[-1]

    G = remove_edge(G, edge_to_remove[0], edge_to_remove[1])

    if G:
        value = nx.dijkstra_path_length(G, 0, size - 1)
        diff = value - datum
        K.append(edge_to_remove)
        opts = len(C)*2 + len(K)
        new_pack = (diff, diff/opts, random.random(), [G, C, K])
        return new_pack

    return None


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
        opts = len(C)*2 + len(K)
        new_pack = (diff, diff/opts, random.random(), [G, C, K])
        return new_pack

    return None


'''
#########################################
#########################################
#########################################
make sure to change num_c and num_k when changing graph size
small graph    num_c = 1 and num_k = 15
medium graph   num_c = 3 and num_k = 50
large graph    num_c = 5 and num_k = 100
#########################################
#########################################
#########################################
'''

def is_finished(pack, num_c=5, num_k=100):
    lst = pack[3]
    C = lst[1]
    K = lst[2]
    return num_c - len(C), num_k - len(K)

def clean_graph(G):
    edges = list(G.edges)
    for i in edges:
        if i[0] == i[1]:
            G.remove_edge(i[0], i[1])
    return G

def prune_pq(pq):

    respq = [i for i in pq if i]
    numspq1 = []
    numspq2 = []
    cachespq = []

    for i in respq:
        if i[0] not in numspq1 and i[1] not in numspq2:
            numspq1.append(i[0])
            numspq2.append(i[1])
            cachespq.append(i)

    return cachespq


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
    count1 = 0
    score = float('inf')

    while pq:
        caches = []
        if len(pq) <= 512:

            #cache = heapq.nlargest(1, pq)[0]
            cache = heapq.nlargest(1, pq)[0]
            if score == cache[0]:
                cache = heapq.heappop(pq)

            '''tries the smallest or the largest#############'''
            #if count1 & 2 == 1:
                #cache = heapq.nlargest(1, pq)[0]
            #else:
                #cache = heapq.heappop(pq)

            '''only pops the smallest###############'''
            #cache = heapq.heappop(pq)

            #score = cache[0]
            c, k = is_finished(cache)

            '''edit the combination of these transitions##########################'''
            if c > 0:
                caches.append(try_node_random(copy.deepcopy(cache), size, datum))
                caches.append(try_node_last(copy.deepcopy(cache), size, datum))
                caches.append(try_node_first(copy.deepcopy(cache), size, datum))
                #caches.append(try_node_gain(copy.deepcopy(cache), size, datum))
                caches.append(try_node_cut(copy.deepcopy(cache), size, datum))
                caches.append(try_edge_random(copy.deepcopy(cache), size, datum))
                caches.append(try_node_lightest(copy.deepcopy(cache), size, datum))

            #if k > 1:
                #caches.append(try_edge_cut_2(copy.deepcopy(cache), size, datum))

            if k > 0:
                caches.append(try_edge_first(copy.deepcopy(cache), size, datum))
                caches.append(try_edge_last(copy.deepcopy(cache), size, datum))
                #caches.append(try_edge_gain(copy.deepcopy(cache), size, datum))
                caches.append(try_edge_cut(copy.deepcopy(cache), size, datum))
                caches.append(try_edge_random(copy.deepcopy(cache), size, datum))
                caches.append(try_edge_random(copy.deepcopy(cache), size, datum))
                caches.append(try_edge_lightest(copy.deepcopy(cache), size, datum))

            res = [i for i in caches if i]
            nums = []
            caches = []

            for i in res:
                if i[0] not in nums:
                    nums.append(i[0])
                    caches.append(i)

            if len(caches) > 0:
                caches.sort(reverse=True)
                for i in range(len(caches)):
                    if caches[i]:
                        heapq.heappush(pq, caches[i])
                        count += 1

        pq = prune_pq(pq)
        '''edit the width of searching#################################'''
        if len(pq) > 64:
            while len(pq) > 64:
                heapq.heappop(pq)

        '''edit the schedule of narrowing##############################'''
        if count > 400:
            while len(pq) > 32:
                heapq.heappop(pq)
        if count > 600:
            while len(pq) > 16:
                heapq.heappop(pq)
        elif count > 800:
            while len(pq) > 8:
                heapq.heappop(pq)
        elif count > 1000:
            while len(pq) > 4:
                heapq.heappop(pq)
        count1 += 1
        #print(cache[0])
        #print(count)
        #print(len(pq))
        score = cache[0]

    print(cache[0])
    C = cache[3][1]
    K = cache[3][2]
    return C, K
    pass

small_dir = r"C:\Users\antho\Desktop\170\project\test input\small"
graph_small = os.listdir(small_dir)

med_dir = r"C:\Users\antho\Desktop\170\project\inputs\medium sp"
graph_med = os.listdir(med_dir)

large_dir = r"C:\Users\antho\Desktop\170\project\inputs\large"
graph_large = os.listdir(large_dir)

if __name__ == '__main__':
    for g in graph_large:
        c, k = None, None
        input_path = large_dir + '/' + g
        G = read_input_file(input_path)
        c, k = solve(G)
        assert is_valid_solution(G, c, k)
        print("Shortest Path Difference: {}".format(calculate_score(G, c, k)))
        output_dir = r"C:\Users\antho\Desktop\170\project\outputs\large"
        write_output_file(G, c, k, output_dir + "//" + g[:-3] + '.out')


#if __name__ == '__main__':
#    for med_g in graph_med:
#        input_path = med_dir + '/' + med_g
#        G = read_input_file(input_path)
#        c, k = solve(G)
#        assert is_valid_solution(G, c, k)
#        print("Shortest Path Difference: {}".format(calculate_score(G, c, k)))
#        output_dir = r"C:\Users\antho\Desktop\170\project\outputs\medium"
#        write_output_file(G, c, k, output_dir + "//" + med_g[:-3] + '.out')


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
