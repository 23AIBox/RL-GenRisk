from inputall import *
from DQN import *
import matplotlib.pyplot as plt
import os
import csv
import numpy as np
import networx as nx
import random
from sklearn import preprocessing
os.environ['CUDA_VISIBLE_DEVICES'] = '4,5'

import torch
torch.set_num_threads(4)
gene_sta = []

def seed_torch(seed=42):
    seed = int(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False


with open('../data/GeneID.csv', 'r') as f:
    reader = csv.reader(f)
    for i in reader:
        if i[0] == 'Gene Symbol':
            continue
        gene_sta.append(i[0])
test_sta_total = []

test_sta_number=len(test_sta_total)
test_sta=[]
test_sta2=[]
sample_index=random.sample(range(0, test_sta_number), int(test_sta_number*0.7))

def Normalized(feature):
    X_scaler = preprocessing.StandardScaler()
    X_train = X_scaler.fit_transform(feature)
    return X_train
def Normalized_minmax(feature):
    X_scaler = preprocessing.MinMaxScaler()
    X_train=copy.deepcopy(feature)
    for i in range(len(feature[0])):
        X_train[:,[i]] = X_scaler.fit_transform(feature[:,[i]])
    return X_train
def get_feature(net,weights,gene_name,gene_final):
    nodes_size = net.shape[0]
    feature = np.zeros((nodes_size, 3))
    i=0
    for gene in list(gene_name):
        feature[i][0] = np.sum(net[i])
        feature[i][1] = weights[gene]
        if gene in list(gene_final.keys()):
            feature[i][2] = len(gene_final[gene])
        i=i+1
    #feature = Normalized_minmax(feature)
    feature = Normalized(feature)

    return feature

def get_feature1(net,actions,weights,gene_name,gene_final):
    nodes_size = net.shape[0]
    feature = np.zeros((nodes_size, 3))
    i=0
    for gene in list(gene_name):
        if i not in actions:
            feature[i][0] = np.sum(net[i])
            feature[i][1] = weights[gene]
            if gene in list(gene_final.keys()):
                feature[i][2] = len(gene_final[gene])
        i=i+1
    feature = Normalized(feature)
    return feature


def laplacian(net):
    lap = copy.deepcopy(net)
    lap = lap * (-1)
    for i in range(net.shape[0]):
        lap[i][i] = np.sum(net[i])
    return lap

def evaluate(gene2):
    gene_sta = []
    with open('../data/GeneID.csv', 'r') as f:
        reader = csv.reader(f)
        for i in reader:
            gene_sta.append(i[0])

    gene1_num = []
    num = 0
    for i in list(gene2.keys()):
        if i in gene_sta:
            num = num + 1
    num1 = 0
    num2 =0
    for i in list(gene2.keys()):
        if i in test_sta:
            num1 = num1 + 1
        if i in test_sta2:
            num2 = num2 + 1

    return num,num1,num2
def get_data_output():
    df_HPRD = pd.read_csv('../data/HPRD.txt', sep=' ',header=None)
    edge_list = np.array(df_HPRD)
    G_hprd = nx.Graph()
    G_hprd.add_edges_from(edge_list)
    nodelist = list(G_hprd.nodes())
    df_gold = pd.read_csv('../data/sta_ccRCC_Merged.txt',header=None)
    lst_gold = df_gold[0].tolist()
    lst_gold = np.intersect1d(nodelist, lst_gold)
    embedding = np.load('../data/Embedding.npy')
    df_gene_idx = pd.read_csv('../data/Embedding_Gene_idx.txt', sep='\t', header=None)
    emb_gene_idx = df_gene_idx[0].tolist()
    dict_embedding = {}
    for i in range(len(emb_gene_idx)):
        dict_embedding[emb_gene_idx[i]] = embedding[i]
    return G_hprd, lst_gold, dict_embedding

def one_side_ttest(value, lst_data):
    from scipy import stats
    data = lst_data  
    t_stat, p_value = stats.ttest_1samp(data, value)
    p_value_lesser = p_value / 2 if t_stat > 0 else 1 - p_value / 2 
    p_value_greater= p_value / 2 if t_stat < 0 else 1 - p_value / 2
    return p_value_greater, p_value_lesser

def get_average_STPL(gene_ranking, G_hprd, lst_gold):
    dict_average_STPL = {}
    dict_average_STPL_p = {}
    for x in gene_ranking:
        dict_average_STPL[x] = 0
        dict_average_STPL_p[x] = 0
    lst_data = []
    for x in gene_ranking:
        lst_lengths = []
        for y in lst_gold:
            if x == y:
                continue
            if nx.has_path(G_hprd, x, y):
                length = nx.shortest_path_length(G_hprd, y, x)
                lst_lengths.append(length)
            else:
                lst_lengths.append(15) # use 15 as max value
        dict_average_STPL[x] = np.nanmean(lst_lengths)
        lst_data.append(np.nanmean(lst_lengths))
    for x in gene_ranking:
        p_greater, p_lesser = one_side_ttest(dict_average_STPL[x], lst_data)
        if p_lesser < 1e-160:
            p_lesser = "< 1e-160"
        dict_average_STPL_p[x] = p_lesser
    return dict_average_STPL, dict_average_STPL_p
        
def get_average_CS(gene_ranking, lst_gold, dict_embedding):
    from sklearn.metrics.pairwise import cosine_similarity
    dict_average_CS = {}
    dict_average_CS_p = {}
    for x in gene_ranking:
        dict_average_CS[x] = 0
        dict_average_CS_p[x] = 0
    lst_data = []
    for x in gene_ranking:
        lst_lengths = []
        tmp_emb = dict_embedding[x].reshape(1, -1)
        for y in lst_gold:
            if x == y:
                continue    
        tgt_emb = dict_embedding[y].reshape(1, -1)
        sim = cosine_similarity(tmp_emb, tgt_emb)[0][0]

        dict_average_CS[x] = np.nanmean(sim)
        lst_data.append(np.nanmean(sim))

    for x in gene_ranking:
        p_greater, p_lesser = one_side_ttest(dict_average_CS[x], lst_data)
        if p_greater < 1e-160:
            p_greater = "< 1e-160"
        dict_average_CS_p[x] = p_greater
    return dict_average_CS, dict_average_CS_p
        

def run(gene_final,score_alpha):

    i = 0
    gene_sort = {}
    log_path="log"
    os.makedirs(log_path, exist_ok=True)
    f2 = open(log_path+"/log_"+cancer+".txt", "w")
    save_path = "agent_ccRCC.th"

    RL.load(save_path)

    RL.clear_mem()
    RL.epsilon = 1
    feature = get_feature(network,weights,gene_name,gene_final)

    patient = []
    for gene in list(gene_final.keys()):
        patient.extend(gene_final[gene])
    pat_num = len(set(patient))
    RL.pat_num = pat_num

    action_sel=[i for i in range(RL.n_actions)]
    action_index= RL.choose_action(feature,action_sel,RL.actions_index)
    f = open("Ranking_List.txt", "w")
    result=[]
    for i in range(len(action_index)):
        result.append([gene_name[i],action_index[i]])
    result.sort(key=lambda x: x[1],reverse=True)
    gene_ranking = []
    for i in range(len(action_index)):
        gene_ranking.append(result[i][0])
    G_hprd, lst_gold, dict_embedding = get_data_output()
    dict_average_STPL, dict_average_STPL_p  = get_average_STPL(gene_ranking, G_hprd, lst_gold)
    dict_average_CS, dict_average_CS_p = get_average_CS(gene_ranking, lst_gold, dict_embedding)
    print('Gene' + "\t" + "Average shortest path length to known risk genes" +"\t" +"p-value (Average shortest path length)" + "\t" +"Average cosine similarity with known risk genes" +"\t" + "p-value (Average cosine similarity)" + "\n")
    for i in range(len(action_index)):
        x = result[i][0]
        print(result[i][0]+"\t"+str(dict_average_STPL[x])+"\t"+str(dict_average_STPL_p[x])+"\t"+str(dict_average_CS[x]) + "\t"+str(dict_average_CS_p[x]) , file = f)
    print(action_index,action_index.shape)
    f.close()
    exit()

    return gene_sort


if __name__ == "__main__":
    import sys
    cancer = 'KIRC'
    seed = 1
    weightnumber = 50
    seed_torch(seed)
    f = open("test_" + cancer + ".txt", "w")
    for i in range(test_sta_number):
        if i in sample_index:
            test_sta.append(test_sta_total[i])
        else:
            test_sta2.append(test_sta_total[i])
            print(test_sta_total[i], file=f)
    f.close()
    with open("../data/sta_ccRCC_Merged.txt", 'r') as file_to_read:
        for line in file_to_read.readlines():
            gene_temp = line.split()
            test_sta_total.append(gene_temp[0])
    file_to_read.close()
    train_patient_data, test_patient_data,patients = getInput(cancer)
    gene_data = getGene(patients)
    network, gene_final, gene_name = getNetworkall(gene_data)
    weights = getWeight(gene_name)

    feature = get_feature(network,weights,gene_name,gene_final)

    len_gene = len(gene_name)
    score_alpha=0.5
    RL = DeepQNetwork(len_gene, network[:, :], feature[:,:],64,
                      train_patient_data=train_patient_data,
                      test_patient_data=test_patient_data,
                      gene_sta = test_sta,
                      weights = weights,
                      learning_rate=0.001,
                      reward_decay=0.9,
                      e_greedy=0.95,
                      replace_target_iter=100,
                      memory_size=3000,
                      score_alpha=score_alpha
                      # output_graph=True
                      )
    gene_sort = run(gene_final,score_alpha)
    


