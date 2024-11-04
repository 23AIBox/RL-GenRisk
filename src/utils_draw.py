import pandas as pd
import numpy as np
from matplotlib import rcParams
rcParams['font.family'] = 'sans-serif'
rcParams['font.sans-serif'] = ['Arial']
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import random
from scipy import stats
import networkx as nx
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

def draw_2_box(lst_1, lst_2, name, top_n, save_path, save_check):
    u_statistic, p_value = stats.mannwhitneyu(lst_1, lst_2)
    print(p_value)
    plt.figure(figsize=(4, 6), dpi=100)

    sns.boxplot(data=[lst_1, lst_2], width=0.6, saturation=1, palette="Set2", fliersize=1, linewidth=3)
    plt.ylabel(name, fontsize=25)

    max_y = max(max(lst_1), max(lst_2))
    y_offset = 0.1 * max_y  

    plt.ylim(-1, max_y + y_offset * 3)

    x_pos = [0, 1]
    y_pos = max_y + y_offset

    plt.plot([0, 0.3], [y_pos, y_pos], color='black', linewidth=1.5)
    plt.plot([0.7, 1], [y_pos, y_pos], color='black', linewidth=1.5)
    plt.plot([0, 0], [y_pos, y_pos - y_offset / 2], color='black', linewidth=1.5)
    plt.plot([1, 1], [y_pos, y_pos - y_offset / 2], color='black', linewidth=1.5)
    
    if p_value<0.001:
        logo = "***"
    elif p_value > 0.001 and p_value < 0.01:
        logo = "**"
    elif p_value > 0.01 and p_value < 0.05:
        logo = "*"
    else:
        logo = "ns"
    plt.text((x_pos[0] + x_pos[1]) / 2, y_pos - y_offset / 2, logo, color='#8B0000', ha='center', va='bottom', fontsize=30)

    plt.gca().spines['bottom'].set_linewidth(2)
    plt.gca().spines['left'].set_linewidth(2)
    plt.gca().spines['top'].set_linewidth(2)
    plt.gca().spines['right'].set_linewidth(2)
    plt.xticks([0, 1], [top_n, 'Random'], fontsize=25)
    plt.yticks(fontsize=20)
    if save_check == True:
        plt.savefig(save_path, bbox_inches = 'tight')

df_ranking = pd.read_csv('Ranking_List.txt', sep='\t')
df_ranking

K = 20
gene_topK = []
topK_STPL = []
topK_CS = []

for i in range(K):
    gene_topK.append(df_ranking.iloc[i]['Gene'])
    topK_STPL.append(df_ranking.iloc[i]['Average shortest path length to known risk genes'])
    topK_CS.append(df_ranking.iloc[i]['Average cosine similarity with known risk genes'])
Random_STPL = []
Random_CS = []

ALL_STPL = df_ranking['Average shortest path length to known risk genes'].tolist()
ALL_CS = df_ranking['Average cosine similarity with known risk genes'].tolist()


for i in range(100): 
    sampled_data_STPL = np.random.choice(ALL_STPL, size=K)
    sampled_data_CS = np.random.choice(ALL_CS, size=K)
    for x in sampled_data_STPL:
        Random_STPL.append(x)
    for x in sampled_data_CS:
        Random_CS.append(x)

draw_2_box(topK_STPL, ALL_STPL, "Average shortest path length", "Top "+str(K), "avg_shortest_path_length.pdf", True)
draw_2_box(topK_CS, ALL_CS, "Average cosine similarity", "Top "+str(K), "avg_cosine_similarity.pdf", True)



def draw_top_k_proportion(gene_ranking, num):
    filename_1_intogen = '../data/sta_ccRCC_IntOGen.txt'
    filename_1_ngc = '../data/sta_ccRCC_NCG.txt'
    filename_1_all = '../data/sta_ccRCC_Merged.txt'

    df_gold_genes_intogen = pd.read_csv(filename_1_intogen, header=None)
    lst_gold_genes_intogen = df_gold_genes_intogen[0].tolist()
    
    df_gold_genes_ngc = pd.read_csv(filename_1_ngc, header=None)
    lst_gold_genes_ngc = df_gold_genes_ngc[0].tolist()

    df_gold_genes_all = pd.read_csv(filename_1_all, header=None)
    lst_gold_genes_all = df_gold_genes_all[0].tolist()

    gold_standards = {
        'IntOGen': lst_gold_genes_intogen,
        'NCG': lst_gold_genes_ngc,
        'Merged': lst_gold_genes_all
    }

    topK_genes = gene_ranking[:num]
    overlap_matrix = []
    k_lst = [20, 30, 40, 50, 60, 70, 80, 90, 100]
    dict_topk_acc = {}
    dict_topk_acc['IntOGen'] = {}
    dict_topk_acc['NCG'] = {}
    dict_topk_acc['Merged'] = {}
    
    for k in k_lst:
        dict_topk_acc['IntOGen'][k] = 0
        dict_topk_acc['NCG'][k] = 0
        dict_topk_acc['Merged'][k] = 0

    for k in k_lst:
        cnt_intogen = 0
        cnt_ngc = 0
        cnt_merged = 0
        for i in range(k):
            if topK_genes[i] in lst_gold_genes_intogen:
                cnt_intogen+=1
            if topK_genes[i] in lst_gold_genes_ngc:
                cnt_ngc+=1
            if topK_genes[i] in lst_gold_genes_all:
                cnt_merged+=1
        dict_topk_acc['IntOGen'][k] = cnt_intogen/k
        dict_topk_acc['NCG'][k] = cnt_ngc/k
        dict_topk_acc['Merged'][k] = cnt_merged/k
        
        data = dict_topk_acc
    
    bar_width = 0.25
    index = np.array(list(data['IntOGen'].keys()))
    indices = np.arange(len(index))
    intogen_values = list(data['IntOGen'].values())
    ngc_values = list(data['NCG'].values())
    merged_values = list(data['Merged'].values())

    plt.figure(figsize=(12, 6))
    color_g = (194/255, 207/255, 162/255)
    color_p = (164/255, 121/255, 158/255)
    color_q = (94/255, 166/255, 156/255)
    plt.bar(indices - bar_width, intogen_values, bar_width, label='IntOGen', color = color_p)
    plt.bar(indices, ngc_values, bar_width, label='NCG',color = color_g)
    plt.bar(indices + bar_width, merged_values, bar_width, label='Merged',color=color_q)
    plt.xlabel('Top K Identified Risk Genes', fontsize=20)
    plt.ylabel('Percentage', fontsize=20)
    plt.xticks(indices, index, fontsize=20)
    plt.yticks(fontsize=20)
    plt.legend(frameon=False,fontsize=15)

    plt.savefig("TopK_known_Proportion.pdf", bbox_inches='tight')
    plt.show()
gene_ranking = df_ranking['Gene'].tolist()
draw_top_k_proportion(gene_ranking, 100)

