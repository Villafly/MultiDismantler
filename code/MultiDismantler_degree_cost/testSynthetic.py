import os
#sys.path.append(os.path.dirname(__file__) + os.sep + '../')
from MultiDismantler_torch import MultiDismantler
from tqdm import tqdm
import torch.backends.cudnn as cudnn
import random
import numpy as np
import torch
import argparse
ap = argparse.ArgumentParser()
ap.add_argument("-o", "--output", required=True,
                help="path to output file")
args = vars(ap.parse_args())
def create_dynamic_table(headers, data):
    table = "|"
    for header in headers:
        table += f" {header} |"
    table += "\n|"
    for _ in headers:
        table += " ------- |"
    table += "\n|"
    for row in data:
        for item in row:
            table += f" {item} |"
    return table
def main():
    dqn = MultiDismantler()
    # data_test_path = '../data/synthetic/uniform_cost/'
    data_test_path = './synthetic/'
    types = ['data_g', 'data_gamma', 'data_k']
    
    data_test_name = ['32','64', '128','256', '512', '1024']
    #data_test_name = ['32']
    model_file = './models/nrange_30_50_iter_100000.ckpt'
    #file_path = './results/MultDismantler_synthetic_cost/'
    ap = argparse.ArgumentParser()
    ap.add_argument("-o", "--output", required=True,
                help="path to output file")
    args = vars(ap.parse_args())
    
    '''
    score_std_list = []
    for dirt in dir_type:
        #file_path = f'./results/MultDismantler_synthetic_cost/{dirt}'       
        for i in tqdm(range(len(data_test_name))):
        #    with open('%s/result_%s.txt'%(file_path,data_test_name[i]), 'w') as fout:
            data_test = data_test_path + data_test_name[i]
            score_mean, score_std, time_mean, time_std, cost_mean = dqn.Evaluate(data_test, data_test_name[i], dirt,model_file,)
            score_std_list.append(['%.4f±%.2f,' % (score_mean , score_std )])
            print(score_mean)
        #   fout.write('%.4f±%.2f,' % (score_mean , score_std ))
        #   fout.write('%.4f' % (cost_mean))
        #   fout.flush()
        #   print('data_test_%s has been tested!' % data_test_name[i])
    markdown_table = create_dynamic_table(data_test_name, score_std_list)
    file_path = args['output'] 
    #if not os.path.exists(file_path):
    #    os.makedirs(file_path)
    output_directory = os.path.dirname(args['output'])
    print("output_directory", output_directory)
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    with open(output_directory + "/markdown.md", 'w', encoding="utf8") as file:
        file.write(markdown_table)
    '''
    for data_type in types:
        file_path = args['output'] + data_type
        if not os.path.exists(file_path):
            os.makedirs(file_path)
        for i in tqdm(range(len(data_test_name))):
            with open('%s/result_%s_unit_cost.txt'%(file_path,data_test_name[i]), 'w') as fout:
                data_test = data_test_path + data_test_name[i]
                score_mean, score_std, time_mean, time_std, cost_mean = dqn.Evaluate(data_test, data_test_name[i], data_type, model_file)
                fout.write('%.4f±%.2f,' % (score_mean , score_std ))
                print('data_test_%s has been tested!' % data_test_name[i])

if __name__=="__main__":
    cudnn.benchmark = False
    cudnn.deterministic = True
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    main()