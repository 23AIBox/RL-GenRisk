import pandas as pd
import numpy as np
import random

# 转化数据 以患者为样本，划分训练集，测试集
def getInput(cancer):
    filename = "nCOP-master/Inputs/"+cancer+".txt"
    patients = {}
    with open(filename, 'r') as file_to_read:
        for line in file_to_read.readlines():
            gene_temp = line.split()
            if gene_temp[0] in ['TTN','MUC16','SYNE1','NEB','MUC19','CCDC168','FSIP2','OBSCN','GPR98']:
                continue
            for patient in gene_temp[1:]:
                if patient not in list(patients.keys()):
                    patients[patient] = [gene_temp[0]]
                else:
                    patients[patient].append(gene_temp[0])
    patient_num = len(patients.keys())
    train_size = int(patient_num*0.8)
    train_data = {}
    test_data = {}
    patients_name = list(patients.keys())
    for i in range(patient_num):
        if i < train_size:
            patient = patients_name[i]
            train_data[patient] = patients[patient]
        else:
            patient = patients_name[i]
            test_data[patient] = patients[patient]
    print('总患者样本数', len(patients.keys()))
    print('训练集患者样本数', len(train_data.keys()))
    print('测试集患者样本数', len(test_data.keys()))
    return patients, test_data,patients

# 导入权重矩阵
def getWeight(gene_name):
    filename = 'nCOP-master/Inputs/weights.txt'
    weights = {}
    with open(filename, 'r') as file_to_read:
        for line in file_to_read.readlines():
            gene_temp = line.split()
            gene=gene_temp[0]
            if gene in list(gene_name):
                weight=gene_temp[1]
                weights[gene] = float(weight)
    weights_value = list(weights.values())
    # for gene in list(weights.keys()):
    #     weight = weights[gene]
    #     # print(type(weight),type(max_weight))
    #     weights[gene] = weight/sum(weights_value)
    return weights

# 生成基因及其覆盖的样本字典 基因：覆盖样本
def getGene(patients):
    gene_dic = {}
    for patient in list(patients.keys()):
        genes = patients[patient]
        for gene in genes:
            if gene not in list(gene_dic.keys()):
                gene_dic[gene] = [patient]
            else:
                gene_dic[gene].append(patient)
    return gene_dic

# 生成随机样本的基因及其覆盖的样本字典 基因：覆盖样本
def random_getGene(patients,gene_name):
    gene_dic = {}
    for patient in list(patients.keys()):
        genes = patients[patient]
        for gene in genes:
            if gene in list(gene_name):
                if gene not in list(gene_dic.keys()):
                    gene_dic[gene] = [patient]
                else:
                    gene_dic[gene].append(patient)
    return gene_dic



# 随机抽选85%的患者样本
def random_patient(patients):
    ran_patient = {}
    num = int(len(patients)*0.85)
    a = random.sample(patients.keys(), num)
    for i in a:
        ran_patient[i]=patients[i]
    return ran_patient

# 导入网络
# set1 基因名称
def getNetwork(gene):
    set1 = []
    net = {}
    gene_new = {}
    filename = 'nCOP-master/Inputs/HPRD.txt'
    with open(filename, 'r') as file_to_read:
        for line in file_to_read.readlines():
            gene_temp = line.split()
            gene1 = gene_temp[0]
            gene2 = gene_temp[1]
            if gene1 in list(net.keys()):
                net[gene1].append(gene2)
            else:
                net[gene1] = [gene2]
            if gene1 not in list(gene_new.keys()) :
                if gene1 in list(gene.keys()):
                    gene_new[gene1] = gene[gene1]
                else:
                    gene_new[gene1] = []
            if gene2 not in list(gene_new.keys()):
                if gene2 in list(gene.keys()):
                    gene_new[gene2] = gene[gene2]
                else:
                    gene_new[gene2] = []
            if gene1 not in set1:
                set1.append(gene1)
            if gene2 not in set1:
                set1.append(gene2)

    gen_len = len(set1)
    print(len(set1))
    print(len(gene_new))
    network = pd.DataFrame(np.zeros((gen_len, gen_len)), index=list(set1), columns=list(set1))
    for gene1 in list(net.keys()):
        for gene2 in net[gene1]:
            network[gene1][gene2] = 1
            network[gene2][gene1] = 1

    return np.array(network), gene_new, set1
