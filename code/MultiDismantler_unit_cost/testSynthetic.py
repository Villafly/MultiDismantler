7#!/usr/bin/env python2
# -*- coding: utf-8 -*-
import sys,os
#sys.path.append(os.path.dirname(__file__) + os.sep + '../')
from MultiDismantler_torch import MultiDismantler
from tqdm import tqdm
import torch.backends.cudnn as cudnn
import torch
import random
import pickle
import numpy as np
import argparse
print("pwd", os.path.dirname(__file__))
def main():
    dqn = MultiDismantler()
    data_test_path = '../data/synthetic/uniform_cost/'
    data_test_name =['32','64','128','256','512','1024']
    #data_test_name = ['32']
    model_file = './models/g0.5_TORCH-Model_GMM_30_50/nrange_30_50_iter_100000.ckpt'
    types = ['data_g', 'data_gamma', 'data_k']

    ap = argparse.ArgumentParser()
    ap.add_argument("-o", "--output", required=True,help="path to output file")
    args = vars(ap.parse_args())
   
    for data_type in types:
        file_path = args['output'] + data_type
        if not os.path.exists(file_path):
            os.makedirs(file_path)
        for i in tqdm(range(len(data_test_name))):
            with open('%s/result_%s_unit_cost.txt'%(file_path,data_test_name[i]), 'w') as fout:
                data_test = data_test_path + data_test_name[i]
                score_mean, score_std, time_mean, time_std, cost_mean = dqn.Evaluate(data_test, data_test_name[i], data_type, model_file)
                fout.write('%.4f±%.2f,' % (score_mean , score_std ))
                #print("cost_mean", cost_mean)
                #fout.write('%.4f' % (cost_mean))
                #pickle.dump(str(cost_mean), fout)
                #fout.flush()
                print('data_test_%s has been tested!' % data_test_name[i])


if __name__=="__main__":
    # cudnn.benchmark = True
    # cudnn.deterministic = False
    # random.seed(0)
    # np.random.seed(0)
    # torch.manual_seed(0)
    # torch.cuda.manual_seed(0)
    main()

