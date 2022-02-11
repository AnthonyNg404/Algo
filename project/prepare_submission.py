import sys
import os
import json
from parse import validate_file

if __name__ == '__main__':
    outputs_dir = "C:/Users/antho/Desktop/170/project/submit/outputs"
    inputs_dir = "C:/Users/antho/Desktop/170/project/submit/inputs"
    submission_name = "submission1.json"
    submission = {}
    for i in os.listdir(inputs_dir):
        for input_path in os.listdir(inputs_dir+'/'+i):
            print(input_path)
            graph_name = input_path.split('.')[0]
            output_file = outputs_dir + '/' + i + '/' + graph_name + '.out'
            print(output_file)
            if os.path.exists(output_file) and validate_file(output_file):
                output = open(output_file).read()
                print('TTT')
                submission[input_path] = output
    with open(submission_name, 'w') as f:
        f.write(json.dumps(submission))
