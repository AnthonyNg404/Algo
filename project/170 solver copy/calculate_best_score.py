import networkx as nx
from parse import read_input_file, write_output_file
from utils import is_valid_solution, calculate_score
import sys
from os.path import basename, normpath


def main():

    small_dir = r"/Users/xinyu/downloads/inputs/small"
    med_dir = r"/Users/xinyu/downloads/inputs/medium"
    large_dir = r"/Users/xinyu/downloads/inputs/large"
    small_output_dir1 = r"/Users/xinyu/Desktop/outputs_/small"
    small_output_dir2 = r"/Users/xinyu/Desktop/outputs_/small1"

    for i in range(1, 300):
        input_path = small_dir + '/' + 'small-{}.in'.format(i)
        G = read_input_file(input_path)
        c1 = []
        k1 = []
        readc1 = False
        readk1 = False
        countc1 = 0
        countk1 = 0
        output1 = open('/Users/xinyu/Desktop/outputs_/small/small-{}.out'.format(i), 'r')
        while True:
            line = output1.readline()

            if not line: #end of the file
                break

            if readc1 == False: # get the num of city
                countc1 = int(line)
                readc1 = True
                continue

            elif countc1 != 0:
                c1.append(int(line.strip()))
                countc1 -=1
                continue
            elif (countc1 == 0) & readc1 & (readk1 == False):
                countk1 = int(line)
                readk1 = True
                continue
            elif countk1 !=0:
                nums = line.strip().split(" ")
                num = [int(i) for i in nums]

                k1.append(tuple(num))
                continue
                countk1 -=1


        assert is_valid_solution(G, c1, k1)
        score1 = calculate_score(G, c1, k1)

        output2 = open('/Users/xinyu/Desktop/outputs_/small1/small-{}.out'.format(i), 'r')
        c2 = []
        k2 = []
        readc2 = False
        readk2 = False
        countc2 = 0
        countk2 = 0
        while True:
            line = output2.readline()
            if not line:
                break
            elif not readc2:
                countc2 = int(line)
                readc2 = True
                continue
            elif countc2 != 0:
                c2.append(int(line.strip()))
                countc2 -=1
                continue
            elif (countc2 == 0) & readc2 & (readk2 == False):
                countk2 = int(line)
                readk2 = True
                continue
            elif countk2 !=0:
                nums = line.strip().split(" ")
                num = [int(i) for i in nums]
                k2.append(tuple(num))
                continue
                countk2 -=1
        assert is_valid_solution(G, c2, k2)
        score2 = calculate_score(G, c2, k2)
        best_output_dir = r"/Users/xinyu/Desktop/outputs/best_small"
        if (score1 > score2):
            write_output_file(G, c1, k1, best_output_dir + "/" + "small-{}".format(i) + ".out")
        else:
            write_output_file(G, c2, k2, best_output_dir + "/" + "small-{}".format(i) + ".out")
main()
