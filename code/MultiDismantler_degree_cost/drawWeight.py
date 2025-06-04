import matplotlib.pyplot as plt
import matplotlib
import argparse
import os

ap = argparse.ArgumentParser()
ap.add_argument("-o", "--output", required=True,
                help="path to output file")
args = vars(ap.parse_args())

def read_values(file_path):
    with open(file_path, 'r') as file:
        return [float(line.strip()) for line in file]

dataset = [
           
           ['homo_genetic_multiplex', 18222, 'Homo', 0.6, 12,'D'],
           ['Sanremo2016_final_multiplex', 56562, 'Sanremo2016',0.4, 12,'E'],
           ['celegans_connectome_multiplex', 279, 'Celegan', 0.6, 23,'F'],
          
           ]


for data in dataset:

    file_paths = [
        f"../../data/AUDC/AUDC_data/HDA/cost_degree/cost_{data[0]}_{data[4]}.txt",
        f"../../data/AUDC/AUDC_data/HDA/cost_degree/MaxCCList_Strategy_{data[0]}_{data[4]}.txt",
        f"../../data/AUDC/AUDC_data/CI/cost_degree/cost_{data[0]}_{data[4]}.txt",
        f"../../data/AUDC/AUDC_data/CI/cost_degree/MaxCCList_Strategy_{data[0]}_{data[4]}.txt",
        f"../../data/AUDC/AUDC_data/MinSum/cost/cost_degree/cost_{data[0]}_{data[4]}.txt",
        f"../../data/AUDC/AUDC_data/MinSum/ND/{data[0]}/{str(data[4])[0]}-{str(data[4])[1:]}/{data[0]}-{str(data[4])[0]}-{str(data[4])[1:]}-MinSum-ND-pairs.txt",
        f"../../data/AUDC/AUDC_data/FINDER_ori/cost_degree_100000/StepRatio_0.0000/Cost_{data[0]}.txt",
        f"../../data/AUDC/AUDC_data/FINDER_ori/cost_degree_100000/StepRatio_0.0000/MaxCCList_Strategy_{data[0]}.txt",
        f"../../data/AUDC/AUDC_data/NIRM_cost_degree/cost_{data[0]}_{data[4]}.txt",
        f"../../data/AUDC/AUDC_data/NIRM_MCC/{data[0]}/{data[0]}.txt",
        f"../../data/AUDC/AUDC_data/MultiDismantler/g0.5-degree/StepRatio_0.0000/Cost_{data[0]}_{data[4]}.txt",
        f"../../data/AUDC/AUDC_data/MultiDismantler/g0.5-degree/StepRatio_0.0000/MaxCCList_Strategy_{data[0]}_{data[4]}.txt"

    ]

    # 设置字体和字体大小
    font = {'family': 'Times New Roman',
            'weight': 'normal',
            'size': 14}

    plt.text(0.05, 0.95, f'{data[5]}', transform=plt.gcf().transFigure, fontsize=18, fontweight='normal', fontname='Times New Roman',
             verticalalignment='top')

    plt.rc('font', **font)
    plt.rc('axes', titlesize=16) 
    plt.rc('axes', labelsize=12)  
    plt.rc('xtick', labelsize=12)  
    plt.rc('ytick', labelsize=12)  

    
    labels = ["HDA", "CI", "MinSum", "FINDER", "NIRM", "MultiDismantler"]
    colors = ['#a6cee3', '#b2df8a', '#fb9a99', '#fdbf6f', '#cab2d6', 'r']
    marker = ['o', 's', '^', 'D', '+', '*']

  
    for i in range(0, len(file_paths), 2):
        x_values = read_values(file_paths[i])
        y_values = read_values(file_paths[i + 1])
        plt.plot(x_values[:-1], y_values, label=labels[i // 2], c=colors[i // 2], linewidth=2, marker=marker[i // 2],
                 markevery=0.1)

    plt.xlabel('Fraction of removal node costs')
    plt.ylabel('Normalized LMCC of residual network')

    legend_font = {'family': 'Times New Roman',
                   'weight': 'normal',
                   'size': 10}

    plt.legend(prop=legend_font)

    title_font = {'family': 'Times New Roman', 'weight': 'bold', 'size': 15}
    plt.title(f'{data[2]} - degree costs', fontdict=title_font)

    plt.xlim(0, data[3])
    # plt.xticks(np.arange(0, 0.025, 0.005))
    # plt.savefig('Fb&Tt(random costs)')
    outputpath = f"{args['output']}"
    if not os.path.exists(outputpath):
        os.makedirs(outputpath)
    plt.savefig(outputpath + f"{data[0]}_degree_cost.png")
    plt.close()
