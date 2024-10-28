import pandas as pd
import numpy as np
from matplotlib import rcParams
rcParams['font.family'] = 'sans-serif'
rcParams['font.sans-serif'] = ['Arial']
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

from scipy import stats
import networkx as nx
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

df_gold = pd.read_csv('sta_KIRC_all.txt',header=None)
lst_gold = df_gold[0].tolist()
print(len(lst_gold))
df_q_value = pd.read_csv('Ranking_List.txt', header=None, sep='\t')
gene_ranking = df_q_value[0].tolist()
df_HPRD = pd.read_csv('HPRD.txt', sep=' ',header=None)
edge_list = np.array(df_HPRD)
G_hprd = nx.Graph()
G_hprd.add_edges_from(edge_list)

shortest_path_length = {}
for x in lst_gold:
    shortest_path_length[x] = []
lst_top20_STPL_gold = []
lst_all_STPL_gold = []

lst_target = lst_gold

# users can change K in this place. 
K = 20

for i in range(K):
    tmp_gene = gene_ranking[i]
    for x in lst_target:
        if x == tmp_gene:
            continue
        if nx.has_path(G_hprd, tmp_gene, x):
            length = nx.shortest_path_length(G_hprd, tmp_gene, x)
            lst_top20_STPL_gold.append(length)
        else:
            lst_top20_STPL_gold.append(15)

all_lst = [i for i in range(len(gene_ranking))]
for i in all_lst:
    tmp_gene = gene_ranking[i]
    for x in lst_target:
        if x == tmp_gene:
            continue
        if nx.has_path(G_hprd, tmp_gene, x):
            length = nx.shortest_path_length(G_hprd, tmp_gene, x)
            lst_all_STPL_gold.append(length)
        else:
            lst_all_STPL_gold.append(15)
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
    plt.xticks([0, 1], [top_n, 'ALL'], fontsize=25)
    plt.yticks(fontsize=20)
    if save_check == True:
        plt.savefig(save_path, bbox_inches = 'tight')

draw_2_box(lst_top20_STPL_gold, lst_all_STPL_gold,'STPL', 'top K', 'STPL.pdf', False)



embedding = np.load('Embedding.npy')
df_gene_idx = pd.read_csv('Embedding_Gene_idx.txt', sep='\t', header=None)
dict_embedding = {}
for i in range(len(emb_gene_idx)):
    dict_embedding[emb_gene_idx[i]] = embedding[i]
rom sklearn.metrics.pairwise import cosine_similarity
target_genes = lst_gold
lst_cosine_top20 = []

for i in range(K):
    tmp_gene = gene_ranking[i]
    emb_tmp_gene = dict_embedding[tmp_gene].reshape(1, -1)
    for tgt_gene in target_genes:
        if tgt_gene == tmp_gene:
            continue
        emb_tgt_gene = dict_embedding[tgt_gene].reshape(1, -1)
        lst_cosine_top20.append(cosine_similarity(emb_tmp_gene, emb_tgt_gene)[0][0])
lst_cosine_ALL = []
for i in range(len(gene_ranking)):
    tmp_gene = gene_ranking[i]
    emb_tmp_gene = dict_embedding[tmp_gene].reshape(1, -1)
    for tgt_gene in target_genes:
        if tgt_gene == tmp_gene:
            continue
        emb_tgt_gene = dict_embedding[tgt_gene].reshape(1, -1)
        lst_cosine_ALL.append(cosine_similarity(emb_tmp_gene, emb_tgt_gene)[0][0])


draw_2_box(lst_cosine_top20, lst_cosine_ALL,'Cosine Similarity', 'top 20', 'cosine_similarity.pdf', False)

# The proportion of known risk genes in Top K genes
def draw_top_k_proportion(lst_topK, num):
    filename_1_intogen = 'sta_ccRCC_intogen.txt'
    filename_1_ngc = 'sta_ccRCC_NGC.txt'
    filename_1_all = 'sta_ccRCC_Merged.txt'

    df_gold_genes_intogen = pd.read_csv(filename_1_intogen, header=None)
    lst_gold_genes_intogen = df_gold_genes_intogen[0].tolist()
    
    df_gold_genes_ngc = pd.read_csv(filename_1_ngc, header=None)
    lst_gold_genes_ngc = df_gold_genes_ngc[0].tolist()

    df_gold_genes_all = pd.read_csv(filename_1_all, header=None)
    lst_gold_genes_all = df_gold_genes_all[0].tolist()

    print(len(lst_gold_genes_intogen))
    print(len(lst_gold_genes_ngc))
    print(len(lst_gold_genes_all))

    gold_standards = {
        'IntOGen': lst_gold_genes_intogen,
        'NGC': lst_gold_genes_ngc,
        'Merged': lst_gold_genes_all
    }

    topK_genes = lst_topK[:num]
    overlap_matrix = []
    k_lst = [20, 30, 40, 50, 60, 70, 80, 90, 100]
    dict_topk_acc = {}
    dict_topk_acc['IntOGen'] = {}
    dict_topk_acc['NGC'] = {}
    dict_topk_acc['Merged'] = {}
    
    for k in k_lst:
        dict_topk_acc['IntOGen'][k] = 0
        dict_topk_acc['NGC'][k] = 0
        dict_topk_acc['Merged'][k] = 0

    for k in k_lst:
        cnt_intogen = 0
        cnt_ngc = 0
        cnt_merged = 0
        for i in range(k):
            if top100_genes[i] in lst_gold_genes_intogen:
                cnt_intogen+=1
            if top100_genes[i] in lst_gold_genes_ngc:
                cnt_ngc+=1
            if top100_genes[i] in lst_gold_genes_all:
                cnt_merged+=1
        dict_topk_acc['IntOGen'][k] = cnt_intogen/k
        dict_topk_acc['NGC'][k] = cnt_ngc/k
        dict_topk_acc['Merged'][k] = cnt_merged/k
        
        data = dict_topk_acc
    
    bar_width = 0.25
    index = np.array(list(data['IntOGen'].keys()))
    indices = np.arange(len(index))
    intogen_values = list(data['IntOGen'].values())
    ngc_values = list(data['NGC'].values())
    merged_values = list(data['Merged'].values())

    plt.figure(figsize=(12, 6))
    color_g = (194/255, 207/255, 162/255)
    color_p = (164/255, 121/255, 158/255)
    color_q = (94/255, 166/255, 156/255)
    plt.bar(indices - bar_width, intogen_values, bar_width, label='IntOGen', color = color_p)
    plt.bar(indices, ngc_values, bar_width, label='NGC',color = color_g)
    plt.bar(indices + bar_width, merged_values, bar_width, label='Merged',color=color_q)
    plt.xlabel('Top K Identified Risk Genes', fontsize=20)
    plt.ylabel('Percentage', fontsize=20)
    plt.xticks(indices, index, fontsize=20)
    plt.yticks(fontsize=20)
    plt.legend(frameon=False,fontsize=15)

    plt.savefig("TopK_known_Proportion.pdf", bbox_inches='tight')
    plt.show()


draw_top_k_proportion(gene_ranking, K)

    
    
