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

degree_centrality = nx.degree_centrality(G_hprd)
closeness_centrality = nx.closeness_centrality(G_hprd)
neighbor_degree = nx.average_neighbor_degree(G_hprd)

def draw_3_box(lst_1, lst_2, lst_3, name, top_n, save_path, save_check):
    u_statistic_1, p_value_1 = stats.mannwhitneyu(lst_1, lst_2)
    u_statistic_2, p_value_2 = stats.mannwhitneyu(lst_2, lst_3)
    u_statistic_3, p_value_3 = stats.mannwhitneyu(lst_1, lst_3)
    
    print("P-value between top_n and KRG:", p_value_1)
    print("P-value between KRG and ALL:", p_value_2)
    print("P-value between top_n and ALL:", p_value_3)

    plt.figure(figsize=(6, 6), dpi=100)

    sns.boxplot(data=[lst_1, lst_2, lst_3], width=0.6, saturation=1, palette="Set2", fliersize=1, linewidth=3)
    plt.ylabel(name, fontsize=25)

    max_y = max(max(lst_1), max(lst_2), max(lst_3))
    y_offset = 0.1 * max_y  

    plt.ylim(0, max_y + y_offset * 5)

    def draw_pvalue_line(x1, x2, y_pos, p_value, line_offset=0.05):
        plt.plot([x1, x1, x2, x2], [y_pos, y_pos + line_offset, y_pos + line_offset, y_pos], color='black', linewidth=1.5)
        if p_value < 0.001:
            logo = "***"
        elif p_value < 0.01:
            logo = "**"
        elif p_value < 0.05:
            logo = "*"
        else:
            logo = "ns"
        plt.text((x1 + x2) / 2, y_pos + line_offset, logo, color='#8B0000', ha='center', va='bottom', fontsize=20)

    draw_pvalue_line(0, 1, max_y + y_offset, p_value_1, line_offset=y_offset * 0.5)

    draw_pvalue_line(1, 2, max_y + y_offset * 1.8, p_value_2, line_offset=y_offset * 0.5)

    draw_pvalue_line(0, 2, max_y + y_offset * 3.0, p_value_3, line_offset=y_offset * 0.5)

    plt.gca().spines['bottom'].set_linewidth(2)
    plt.gca().spines['left'].set_linewidth(2)
    plt.gca().spines['top'].set_linewidth(2)
    plt.gca().spines['right'].set_linewidth(2)
    plt.xticks([0, 1, 2], [top_n, 'KRG', 'ALL'], fontsize=25)
    plt.yticks(fontsize=20)

    if save_check == True:
        plt.savefig(save_path, bbox_inches='tight')

    plt.show()
def normalized(lst_tmp):
    max_val = np.nanmax(lst_tmp)
    for i in range(len(lst_tmp)):
        lst_tmp[i] = lst_tmp[i]/max_val
    return lst_tmp


# degree_centrality
# neighbor_degree
# closeness_centrality

lst_top20_degree_centrality = []
lst_top20_neighbor_degree = []
lst_top20_closeness_centrality = []

lst_gold_degree_centrality = []
lst_gold_neighbor_degree = []
lst_gold_closeness_centrality = []


lst_all_degree_centrality = []
lst_all_neighbor_degree = []
lst_all_closeness_centrality = []

for i in range(20):
    tmp_gene = gene_ranking[i]
    lst_top20_degree_centrality.append(degree_centrality[tmp_gene])
    lst_top20_neighbor_degree.append(neighbor_degree[tmp_gene])
    lst_top20_closeness_centrality.append(closeness_centrality[tmp_gene])

for i in range(len(lst_gold)):
    tmp_gene = lst_gold[i]
    lst_gold_degree_centrality.append(degree_centrality[tmp_gene])
    lst_gold_neighbor_degree.append(neighbor_degree[tmp_gene])
    lst_gold_closeness_centrality.append(closeness_centrality[tmp_gene])
    
    
lst_all_degree_centrality = list(degree_centrality.values())
lst_all_neighbor_degree = list(neighbor_degree.values())
lst_all_closeness_centrality = list(closeness_centrality.values())


draw_3_box(normalized(lst_top20_degree_centrality), normalized(lst_gold_degree_centrality), normalized(lst_all_degree_centrality), 'Degree Centrality', 'Top 20', 'Degree.pdf',False)




shortest_path_length = {}
for x in lst_gold:
    shortest_path_length[x] = []
lst_top20_STPL_gold = []
lst_all_STPL_gold = []

lst_target = lst_gold

for i in range(20):
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

draw_2_box(lst_top20_STPL_gold, lst_all_STPL_gold,'STPL', 'top 20', 'STPL.pdf', False)




embedding = np.load('Embedding.npy')
df_gene_idx = pd.read_csv('Embedding_Gene_idx.txt', sep='\t', header=None)
dict_embedding = {}
for i in range(len(emb_gene_idx)):
    dict_embedding[emb_gene_idx[i]] = embedding[i]
rom sklearn.metrics.pairwise import cosine_similarity
target_genes = lst_gold
lst_cosine_top20 = []

for i in range(20):
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


init_embedding = np.load('init_embedding.npy')
dict_init_embedding = {}
for i in range(len(emb_gene_idx)):
    dict_init_embedding[emb_gene_idx[i]] = init_embedding[i]

target_genes = lst_gold
lst_init_cosine_top20 = []

for i in range(20):
    tmp_gene = gene_ranking[i]
    emb_tmp_gene = dict_init_embedding[tmp_gene].reshape(1, -1)
    for tgt_gene in target_genes:
        if tgt_gene == tmp_gene:
            continue
        emb_tgt_gene = dict_init_embedding[tgt_gene].reshape(1, -1)
        lst_init_cosine_top20.append(cosine_similarity(emb_tmp_gene, emb_tgt_gene)[0][0])
lst_init_cosine_ALL = []

for i in range(len(gene_ranking)):
    tmp_gene = gene_ranking[i]
    emb_tmp_gene = dict_init_embedding[tmp_gene].reshape(1, -1)
    for tgt_gene in target_genes:
        if tgt_gene == tmp_gene:
            continue
        emb_tgt_gene = dict_init_embedding[tgt_gene].reshape(1, -1)
        lst_init_cosine_ALL.append(cosine_similarity(emb_tmp_gene, emb_tgt_gene)[0][0])


draw_2_box(lst_cosine_top20, lst_cosine_ALL,'Cosine Similarity', 'top 20', 'cosine_similarity.pdf', False)