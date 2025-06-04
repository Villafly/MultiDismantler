import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import argparse
import os


ap = argparse.ArgumentParser()
ap.add_argument("-o", "--output", required=True,
                help="path to output file")
args = vars(ap.parse_args())

dataset = [
           ['Sanremo2016_final_multiplex', 56562, 'Sanremo2016',0.01, 12,'B'],
           ['celegans_connectome_multiplex', 279, 'Celegan', 0.4, 23,'C'],
           ['homo_genetic_multiplex', 18222, 'Homo', 0.08, 12,'A'],
           ]
for datas in dataset:
    plt.figure()
    file_paths = [
        f"../../data/AUDC/AUDC_data/HDA/none_cost/MaxCCList_Strategy_{datas[0]}.txt",
        f"../../data/AUDC/AUDC_data/CI/none_cost/MaxCCList_Strategy_{datas[0]}.txt",
        f"../../data/AUDC/AUDC_data/MinSum/ND/{datas[0]}/{str(datas[4])[0]}-{str(datas[4])[1:]}/{datas[0]}-{str(datas[4])[0]}-{str(datas[4])[1:]}-MinSum-ND-pairs.txt",
        f"../../data/AUDC/AUDC_data/FINDER_ori/none_cost/StepRatio_0.0000/MaxCCList_Strategy_{datas[0]}.txt",
        f"../../data/AUDC/AUDC_data/NIRM_MCC/{datas[0]}/{datas[0]}.txt",
        f"../../data/AUDC/AUDC_data/MultiDismantler/g0.5/StepRatio_0.0000/MaxCCList_Strategy_{datas[0]}.txt",
    ]

    N = datas[1]

    
    values = []

    
    for file_path in file_paths:
        with open(file_path, "r") as file:
            lines = file.readlines()
            values.append([float(line.strip()) for line in lines])

    
    x_values = []
    y_values = []

    for values_list in values:
        total_lines = len(values_list)
        x_values.append([(i / N)  for i in range(0, total_lines)])
        y_values.append(values_list)

   
    font = {'family': 'Times New Roman',
            'weight': 'normal',
            'size': 14}

    plt.text(0.05, 0.95, f'{datas[5]}', transform=plt.gcf().transFigure, fontsize=18, fontweight='normal', fontname='Times New Roman', verticalalignment='top')

    plt.rc('font', **font)
    plt.rc('axes', titlesize=16)  
    plt.rc('axes', labelsize=12)  
    plt.rc('xtick', labelsize=12)  
    plt.rc('ytick', labelsize=12)  

    
    labels = ["HDA", "CI", "MinSum", "FINDER", "NIRM", "MultiDismantler"]
    colors = ['#a6cee3', '#b2df8a', '#fb9a99', '#fdbf6f', '#cab2d6', 'r']
    marker = ['o', 's', '^', 'D', '+', '*']

    for i in range(len(values)):
        plt.plot(x_values[i], y_values[i], label=f'{labels[i]}', c=colors[i], linewidth=2, marker=marker[i], markevery=0.1)

    plt.xlabel('Fraction of removal node costs')
    plt.ylabel('Normalized LMCC of residual network')

   
    legend_font = {'family': 'Times New Roman',
                   'weight': 'normal',
                   'size': 10}

    plt.legend(prop=legend_font)

    title_font = {'family': 'Times New Roman', 'weight': 'bold', 'size': 15}
    plt.title(f'{datas[2]} - unit costs', fontdict=title_font)
    plt.xlim(0, datas[3])
    # xtic = np.arange(0, datas[2], 0.02)
    # x_ticks = np.arange(0, (datas[3]+0.02), 0.02)
    # plt.xticks(x_ticks)
    # plt.xticks(xtic)
    outputpath = f"{args['output']}"
    if not os.path.exists(outputpath):
        os.makedirs(outputpath)
    plt.savefig(outputpath + f"{datas[0]}_unit_cost.png")
    plt.close()
