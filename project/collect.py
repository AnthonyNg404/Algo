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
import utils
import parse

def read_output_file(G, path):
    """
    Parses and validates an output file

    Args:
        G: input graph corresponding to input file
        path: str, path to output file
    Returns:
        score: the difference between the new and original shortest path
    """
    H = G.copy()
    if len(H) >= 20 and len(H) <= 30:
        max_cities = 1
        max_roads = 15
    elif len(H) > 30 and len(H) <= 50:
        max_cities = 3
        max_roads = 30
    elif len(H) > 50 and len(H) <= 100:
        max_cities = 5
        max_roads = 100
    else:
        print('Input Graph is not of a valid size')

    assert H.has_node(0), 'Source vertex is missing in input graph'
    assert H.has_node(len(G) - 1), 'Target vertex is missing in input graph'

    cities = []
    removed_edges = []

    with open(path, "r") as fo:

        number_of_cities = fo.readline().strip()
        assert number_of_cities.isdigit(), 'Number of cities is not a digit'
        number_of_cities = int(number_of_cities)

        assert number_of_cities <= max_cities, 'Too many cities being removed from input graph'

        for _ in range(number_of_cities):
            city = fo.readline().strip()
            assert city.isdigit(), 'Specified vertex is not a digit'
            city = int(city)
            assert H.has_node(city), 'Specified vertex is not in input graph'
            cities.append(city)

        number_of_roads = fo.readline().strip()
        assert number_of_roads.isdigit(), 'Number of roads is not a digit'
        number_of_roads = int(number_of_roads)

        for _ in range(number_of_roads):
            road = fo.readline().split()
            assert len(road) == 2, 'An edge must be specified with a start and end vertex'
            assert road[0].isdigit() and road[1].isdigit()
            u = int(road[0])
            v = int(road[1])
            assert H.has_edge(u, v), 'Specified edge is not in input graph'
            removed_edges.append((u,v))

    return utils.calculate_score(G, cities, removed_edges), cities, removed_edges

input_dir = r"C:\Users\antho\Desktop\170\project\inputs\large"
graph_file = os.listdir(input_dir)


output1_dir = r"C:\Users\antho\Desktop\170\project\prepare\large3"
output1_file = os.listdir(output1_dir)
output2_dir = r"C:\Users\antho\Desktop\170\project\prepare\large"
output2_file = os.listdir(output1_dir)

if __name__ == '__main__':
    num_overwrite = 0
    for g, out1, out2 in zip(graph_file, output1_file, output2_file):
        c1, k1, c2, k2  = None, None, None, None
        input_path = input_dir + '/' + g
        out1_path = output1_dir + '/' + out1
        out2_path = output2_dir + '/' + out2
        G = read_input_file(input_path)
        score1, c1, k1 = read_output_file(G, out1_path)
        score2, c2, k2 = read_output_file(G, out2_path)
        print(score1, "     ", score2)
        assert is_valid_solution(G, c1, k1)
        assert is_valid_solution(G, c2, k2)
        if score1 > score2:
            num_overwrite += 1
            write_output_file(G, c1, k1, output2_dir + "//" + g[:-3] + '.out')
    print(num_overwrite)
