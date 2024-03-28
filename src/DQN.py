import numpy as np
import pandas as pd
import copy
# np.random.seed(1)
# tf.set_random_seed(1)
import matplotlib.pyplot as plt
import matplotlib as mpl
import torch
from sklearn import preprocessing
from replay_buffer import ReplayBuffer
from qfunction import Q_Fun
mpl.use('Agg')

# Deep Q Network off-policy
class DeepQNetwork:
    def __init__(
            self,
            # 多少个动作
            n_actions,
            net_ori,
            fea_ori,
            # 环境特征维度
            embedding_size,
            train_patient_data,
            test_patient_data,
            gene_sta,
            weights,
            score_alpha,
            pat_num = 0,
            learning_rate=0.001,
            # 奖励衰减率
            reward_decay=0.95,
            # 贪心策略值
            e_greedy=0.95,
            replace_target_iter=100,
            memory_size=500,
            batch_size=128,
            e_greedy_increment=0.001,
            output_graph=False,
    ):
        # 原始的网络
        self.net_ori = copy.deepcopy(net_ori)
        self.fea_ori = copy.deepcopy(fea_ori)
        print(np.sum(self.net_ori))
        self.train_patient_data = train_patient_data
        self.test_patient_data = test_patient_data
        self.train_cover = []
        self.test_cover = []
        self.weights = weights
        self.gene_sta = gene_sta
        # # 原始的拉普拉斯矩阵
        # self.laplacian_ori = self.laplacian(self.net_ori)
        # 已经被选择过的节点
        self.actions = []
        # 已选节点位置为1
        self.actions_index = np.ones(n_actions)
        self.n_actions = n_actions
        self.embedding_size = embedding_size
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon_max = e_greedy
        # 隔多少步进行更新
        self.replace_target_iter = replace_target_iter
        self.memory_size = memory_size
        self.batch_size = batch_size
        # 更改贪心策略值，丰富样本
        self.epsilon_increment = e_greedy_increment
        self.epsilon = 0.9
        # 上一步的奖励
        self.score_be = 0
        self.score_sta = 0
        self.score_pat = 0
        # total learning step
        self.learn_step_counter = 0
        # 记录初始节点的嵌入
        self.a_ori = np.zeros((1, self.embedding_size))
        # initialize zero memory [s, a, r, s_]

        self.memory_counter = 0
        # consist of [target_net, evaluate_net]
        self.tau = 0.001

        self.gene_ori = []
        self.reward_all = 0
        self.reward_list = []
        self.train_sta = 0
        self.train_before = 0

        # n-step 步数
        self.n_step = 4
        # 记录loss
        self.cost_his = []
        self.cost_his_emb = []
        self.cost_his_q = []
        T=3
        ALPHA=0.001
        self.sync_target_frames=100
        self.Q = Q_Fun(self.embedding_size, self.embedding_size, T, ALPHA,self.net_ori)
        self.Q_target=Q_Fun(self.embedding_size, self.embedding_size, T, ALPHA,self.net_ori)
        self.memory = ReplayBuffer(self.memory_size,self.n_actions)
        self.score_alpha=score_alpha
        self.embedding=None

        for name, param in self.Q.named_parameters():
            if param.requires_grad:
                print(name)
    # 计算拉普拉斯公式,用于计算图嵌入loss
    def laplacian(self, net):
        lap = copy.deepcopy(net)
        lap=lap*(-1)
        for i in range(net.shape[0]):
            lap[i][i]=np.sum(net[i])
        # for i in range(net.shape[0]):
        #     for j in range(net.shape[1]):
        #         if i == j:
        #             lap[i][j] = np.sum(net[i])
        #         else:
        #             lap[i][j] = -net[i][j]
        return lap

    # 将数据转换为n-step数据
    def getBatch(self):
        selete = [x for x in range(self.memory_size-self.n_step) if self.memory_temp[x] == 1]
        if self.memory_counter > self.memory_size:
            sample_index = np.random.choice(selete, size=self.batch_size)
        else:
            sample_index = np.random.choice(selete, size=self.batch_size)
        batch_r, q_next, batch_s_ = np.zeros((self.batch_size, 1)),np.zeros((self.batch_size, 1)), np.zeros((self.batch_size, self.embedding_size))
        batch_a = self.memory_ar[sample_index, :self.embedding_size]
        batch_s = self.memory_s[sample_index, :]
        batch_fea = self.memory_fea[sample_index, :, :]
        batch_net = self.memory_net[sample_index, :, :]
        betch_lap = self.memory_lap[sample_index, :, :]
        for i in range(self.batch_size):
            index = sample_index[i]
            reward = 0
            for j in range(self.n_step):
                reward = reward + self.memory_ar[index+self.n_step, self.embedding_size]
            batch_r[i,:] = reward
            # print(self.memory_ar[index+self.n_step, self.embedding_size+1].shape)
            action_list = self.memory_actions[index+self.n_step,:]
            network = self.memory_net[index+self.n_step,:,:]
            feature = self.memory_fea[index+self.n_step,:,:]
            s_ = self.memory_s_[index+self.n_step,:]
            # print(action_list.shape)
            # print(network.shape)
            # print(feature.shape)
            # print(s_.shape)
            q_next[i,:] = self.get_train_Q(action_list,network,feature,s_)
            # print(q_next[i,:].shape)
            batch_s_[i,:] = self.memory_s_[index+self.n_step, :]
        return batch_fea,batch_net,betch_lap,batch_s,batch_a,batch_r,q_next,batch_s_
            


    # 存储数据
    def store_transition(self, s, a, r, s_, network_new1, feature_new1, lap):

        if not hasattr(self, 'memory_counter'):
            self.memory_counter = 0
        # print(s)
        # print(s_)
        transition = np.hstack(([a, r]))

        # replace the old memory with new memory
        index = self.memory_counter % self.memory_size
        self.memory_ar[index, :] = transition
        self.memory_s[index, :] = s
        self.memory_s_[index, :] = s_
        self.memory_fea[index, :] = feature_new1
        self.memory_net[index, :] = network_new1
        self.memory_lap[index, :] = lap
        self.memory_actions[index,:]=self.actions_index[:]
        if len(self.actions) >= 150-self.n_step:
            self.memory_temp[index] = 0
        else:
            self.memory_temp[index] = 1
        self.memory_counter += 1

    # 计算当前节点嵌入
    # def getEmbedding(self, feature_new, network_new):
    #     embedding = self.sess.run(self.emb_node,
    #                               feed_dict={self.feature_ori: feature_new,
    #                                          self.net: network_new})
    #     return embedding

    # 计算状态
    def getState(self, feature_new, network_new):
        embedding = self.sess.run(self.emb_node,
                                  feed_dict={self.feature_ori: feature_new,
                                             self.net: network_new})
        s=np.sum(embedding,axis=0,keepdims=True)
        return s
    # # 计算状态
    # def getState(self):
    #     node_emb_ori = self.sess.run(self.emb_node, feed_dict={self.feature_ori: self.fea_ori,
    #                                         self.net: self.net_ori})
    #     # print(type(self.actions_index))
    #     # print(type(node_emb_ori))
    #     s = np.dot(np.expand_dims(self.actions_index, 0),node_emb_ori)/np.sum(self.actions_index)
    #
    #     return s


    # 计算当前可选动作
    def getAction(self,  network_new,action_selold,action_index):
        # 找到可选动作编号

        # 寻找已选动作的相邻节点
        # for i in list(self.actions):
        #     for j in range(network_new.shape[0]):
        #         if self.net_ori[i][j] > 0 and j not in action_sel and j not in self.actions:
        #             action_sel.append(j)
        #             action_index[j]=1

        #print('相邻节点个数', len(action_sel))
        # 如果没有相邻节点，在初始的5个节点中选择
        # if len(action_sel) == 0:
        #     for i in self.gene_ori:
        #         if i not in self.actions:
        #             action_sel.append(i)
        #             action_selold.append(i)
        #             action_index[i]=1
        #             break
        # else:

        i=self.actions[-1]
        action_selold.remove(i)
        action_index[i]=0
        for j in range(network_new.shape[0]):
            if self.net_ori[i][j] > 0 and j not in action_selold and j not in self.actions:
                action_selold.append(j)
                action_index[j]=1
        # print(action_selold)
        # print("!!!!!!!!!!!!!")
        if len(action_selold) == 0:
            for i in self.gene_ori:
                if i not in self.actions:
                    action_selold.append(i)
                    action_index[i]=1
        # 如果初始节点也选择完，那么随机进行选择
        return action_selold,action_index
    
    # 计算训练集的Q值
    def get_train_Q(self,action_list,network,feature,s_):
        # 找到已选节点的邻居
        nei_list = np.dot(action_list,self.net_ori)
        node_emb_all = self.sess.run(self.emb_node, feed_dict={self.feature_ori: feature,
                                            self.net: network})
        # 找到可选动作编号
        action_sel = []                             
        for i in range(self.n_actions):
            if nei_list[i]>0 and action_list[i]==0:
                action_sel.append(i)
        # 汇总可选动作的嵌入
        action_emb = np.zeros((len(action_sel), self.embedding_size))
        for i in range(len(action_sel)):
            index = action_sel[i]
            action_emb[i, :] = node_emb_all[index, :]
        # 计算所有可选动作的Q值
        s = np.expand_dims(s_, 0)
        s_all = np.repeat(s, len(action_sel), axis=0)

        actions_value = self.sess.run(self.q_target_, feed_dict={self.s_: s_all,
                                                                self.a_: action_emb})
        actions_value = np.squeeze(actions_value)
        qt = np.max(actions_value)
        return qt


    # 选择动作
    def choose_action(self, state,action_sel,action_index):
        # print(action_sel)
        
        # 当相邻节点数不为0时
        if len(action_sel) > 0:
            # s = np.expand_dims(s, 0)

            state = torch.tensor(state, dtype=torch.float32).to(self.Q.device)
            action_index=torch.LongTensor(action_index).to(self.Q.device)

            actions_value,self.embedding = self.Q(self.embedding,state,action_index)
            actions_value=actions_value.detach().numpy()
            return actions_value
            self.embedding=self.embedding.detach()
            # print (actions_value)
            # print (action_sel)
            # print (actions_value[action_sel])
            # print ("-----------------------")
            # print (actions_value)
            # print(actions_value)
            # print(actions_value)
            # print(actions_value)
            # print(action_sel)
            action = np.argmax(actions_value[action_sel])

            action_index = action_sel[action]
            # print(action_index,actions_value[action_index])
            if np.random.uniform() < self.epsilon or self.epsilon==1:
                self.actions.append(action_index)
                self.actions_index[action_index] = 0
            else:
                if len(action_sel)==1:
                    self.actions.append(action_index)
                else:
                    while True:
                        action1 = np.random.randint(0, len(action_sel))
                        action_index = action_sel[action1]
                        if action1 != action:
                            self.actions.append(action_index)
                            self.actions_index[action_index] = 0

                            break
                # print('在相邻节点中随机选择动作')
                # print(action_index)
                # print('----------')
        else:
            while True:
                action_index = np.random.randint(0, self.n_actions)
                if action_index not in self.actions:
                    self.actions.append(action_index)
                    self.actions_index[action_index] = 0
                    break
                    # self.epsilon = min(0.95,self.epsilon + self.e_greedy_increment)
        return action_index

    # 计算下一状态时的最大Q值，用于计算cost
    def getQt(self, feature_new, network_new,s_):
        all_embedding = self.getEmbedding(feature_new, network_new)
        action_sel, action_emb = self.getAction(all_embedding, network_new)
        # print('下一状态可选节点个数', len(action_sel))
        qt = -100
        if len(action_sel)!= 0:
            s = np.expand_dims(s_, 0)
            s_all = np.repeat(s, len(action_sel), axis=0)
            actions_value = self.sess.run(self.q_target_, feed_dict={self.s_: s_all,
                                                                  self.a_: action_emb})
            actions_value = np.squeeze(actions_value)
            qt = np.max(actions_value)
        return qt


    # 计算患者未覆盖比例-reward计算
    def get_reward(self, gene_num, gene_name):
        weight_sum = 0
        gene_name = list(gene_name)
        # num_gene = len(gene_name)-len(self.actions)+1
        patient_num = []
        gene_sta_num=0
        for i in self.actions:
            gene = gene_name[i]
            if gene in self.gene_sta:
                gene_sta_num=gene_sta_num+1
            if gene not in list(gene_num.keys()):
                weight_sum = weight_sum + self.weights[gene]
            else:
                patient_num.extend(gene_num[gene])
                weight_sum = weight_sum + self.weights[gene]
        # print('患者数为：',len(set(patient)),'平均患者数为：',len(patient_num)/num_gene)
        return weight_sum*self.n_actions/150,(self.pat_num-len(set(patient_num)))/self.pat_num,(len(self.actions)-gene_sta_num)/len(self.actions)

    # 计算患者覆盖率
    def getAcc(self, actions, patients, gene_name):
        cover_num = 0
        patients_num = len(patients.keys())
        for patient in patients:
            genes = patients[patient]
            for j in actions:
                if list(gene_name)[j] in genes:
                    cover_num += 1
                    break
        return cover_num / patients_num

    def Normalized_minmax(self,feature):
        X_scaler = preprocessing.MinMaxScaler()
        X_train = copy.deepcopy(feature)
        for i in range(len(feature[0])):
            X_train[:, [i]] = X_scaler.fit_transform(feature[:, [i]])
        return X_train

    # 特征归一化
    def Normalized(self,feature):
        X_scaler = preprocessing.StandardScaler()
        X_train = X_scaler.fit_transform(feature)
        return X_train

    def get_feature(self,net,actions,weights,gene_name,gene_final):
        nodes_size = net.shape[0]
        feature = np.zeros((nodes_size, 3))
        i=0
        for gene in list(gene_name):
            feature[i][0] = np.sum(net[i])
            feature[i][1] = weights[gene]
            if gene in list(gene_final.keys()):
                feature[i][2] = len(gene_final[gene])
            i=i+1
        feature = self.Normalized_minmax(feature)
        #feature = self.Normalized(feature)
        return feature
    '''
    与环境进行交互,计算后一状态、回报
    输入动作，基因覆盖的样本数，当前网络，当前初始特征
    '''
    def step(self,network, action, gene_num, gene_name,weights):


        # 将该节点移除网络
        # print (action,len(gene_num))

        # for i in range(len(gene_num)):
        #     for j in range(len(gene_num)):
        #         if i == action or j == action:
        #             net_new[i][j] = 0

        # feature_new = self.get_feature(network,self.actions,weights,gene_name,gene_num)
        # print(feature_new[0][2])
        # all_embedding = self.getEmbedding(feature_new, net_new)
        # s_ = self.getState(feature_new,net_new)
        actions = self.actions[:]
        node_prop = len(actions)
        weight_sum,patient_num,gene_sta_num = self.get_reward(gene_num, gene_name)
        # score_new = gene_sta_num
        score_new = self.score_alpha*weight_sum+(1-self.score_alpha)*patient_num*3000

        score = self.score_be-score_new
        score_st = self.score_sta-gene_sta_num
        score_pa = self.score_pat-patient_num
        reward = 0
        if score>0 or (self.score_be==0 and score==0):
            reward=reward+score_pa*50
        if score_st>0 or (self.score_sta==0 and score_st==0):
            reward=reward+2

        self.reward_all = self.reward_all + reward
        self.score_be = score_new
        self.score_sta = gene_sta_num
        self.score_pat = patient_num
        # print('节点个数：', node_prop, '; 患者覆盖个数：', patient_num,"无用基因比例: ",gene_sta_num, '; 得分为',  score_new)
        # print('奖励为：', reward)
        done = 0
        if len(actions) == 999:
            self.embedding=None
            done = 1
            print(self.actions)
            train_acc = self.getAcc(actions, self.train_patient_data, gene_name)
            self.train_cover.append(train_acc)
            test_acc = self.getAcc(actions, self.test_patient_data, gene_name)
            self.test_cover.append(test_acc)
            print('train_acc', train_acc, 'test_acc', test_acc)
            self.actions = []
            self.reward_list.append(reward)
            self.reward_all = 0
            # print('数据池中数据量为：',self.memory_counter)
        return  reward, done, actions
    def remember(self, *args):
        self.memory.store_transition(*args)
        self.memory_counter+=1
    def clear_mem(self):
        self.memory.clear()
    # 学习
    def learn(self):
        # sample batch memory from all memory
        # if self.memory_counter > self.memory_size:
        #     sample_index = np.random.choice(self.memory_size, size=self.batch_size)
        # else:
        #     sample_index = np.random.choice(self.memory_counter, size=self.batch_size)
        state, action, reward_sum, action_index ,sel_action= self.memory.sample_buffer(self.batch_size)
        # graph_batch.to(self.Q.device)
        self.Q.optimizer.zero_grad()
        mu=None
        action_temp = copy.deepcopy(action)
        action_index_new = copy.deepcopy(action_index)
        for i in range(self.batch_size):
            action_temp_real=int(action_temp[i])
            action_index_new[i,action_temp_real]=0

        state = torch.tensor(state, dtype=torch.float32).to(self.Q.device)
        action=torch.LongTensor(action).to(self.Q.device)
        reward_sum = torch.tensor(reward_sum, dtype=torch.float32).to(self.Q.device)
        new_state = state.clone()
        action_index = torch.LongTensor(action_index).to(self.Q.device)
        action_index_new = torch.LongTensor(action_index_new).to(self.Q.device)
        sel_action = torch.LongTensor(sel_action).to(self.Q.device)

        # mu = graph_batch.x_attr
        # state = graph_batch.state
        # new_state = graph_batch.new_state
        # edge_index = graph_batch.edge_index
        # action = graph_batch.action
        # done = graph_batch.done
        # reward_sum = graph_batch.reward
        # batch_index = graph_batch.batch
        # action_index=graph_batch.action_index

        # num_nodes = self.n_actions




        temp=self._max_Q(mu,new_state, action_index_new,sel_action).unsqueeze(-1)
        # print(temp)
        # print("-----------------------------")
        # exit()
        y_target = torch.add(reward_sum , temp*self.gamma).detach()

        y_pred,_ = self.Q(mu,state, action_index,batch_flag=True)
        y_pred=y_pred.gather(1,action)
        # print (reward_sum)
        # print (y_target)
        # print (y_pred)
        # print ("-----------------------")
        loss = torch.mean(torch.pow(y_target - y_pred, 2))
        loss.backward()
        self.Q.optimizer.step()

        # print(q_target.shape)
        # train eval network


        print('loss为', loss.item())
        self.cost_his.append(loss.item())
        # self.cost_his_emb.append(self.cost_emb)
        # self.cost_his_q.append(self.cost_q)
        # self.sess.run(self.replace_target_op)
        # self.sess.run(self.replace_target_op_eb)
        

        # increasing epsilon
        
        self.epsilon = self.epsilon + self.epsilon_increment if self.epsilon < self.epsilon_max else self.epsilon_max
        self.learn_step_counter += 1
        if self.learn_step_counter % self.sync_target_frames==0:
            self.Q_target.load_state_dict(self.Q.state_dict())

    def _max_Q(self, mu, state, action_index,sel_action,batch_flag=True):
        Q,_ = self.Q_target(mu, state, action_index,batch_flag=True) #[batch_size*N]
        # batch_index = args[4]
        # return scatter_max(Q, batch_index)[0].detach()
        Q_sel=torch.mul(Q,sel_action)
        # Q_sel=Q.gather(1,action)
        value,index=torch.max(Q_sel,1)
        return value
    def save(self, path,cancer):
        torch.save(self.Q.state_dict(),("{}/agent_"+cancer+".th").format(path))

    def load(self, path):
        self.Q.load_state_dict(torch.load(path))
    def plot_cost(self, i):
        plt.figure(12)
        plt.subplot(221)
        plt.plot(np.arange(len(self.cost_his_emb[-200:])), self.cost_his_emb[-200:], label='cost_emb')
        plt.subplot(222)
        plt.plot(np.arange(len(self.cost_his_q[-200:])), self.cost_his_q[-200:], label='cost_q')
        plt.subplot(212)
        plt.plot(np.arange(len(self.cost_his[-200:])), self.cost_his[-200:],label='cost_all')

        plt.ylabel('Cost')
        plt.xlabel('training steps')
        plt.savefig(str(i) + "loss.png")
        plt.clf()

    def plot_reward(self, i,score_PBRM1,score_MUC4,score_VHL):
        #plt.figure(1)
        #plt.subplot(131)
        #plt.plot(np.arange(len(score_PBRM1)), score_PBRM1, label='PBRM1')
        #plt.ylabel('score')
        #plt.xlabel('PBRM1')
        #plt.subplot(132)
        #plt.plot(np.arange(len(score_MUC4)), score_MUC4, label='MUC4')
        #plt.ylabel('score')
        #plt.xlabel('MUC4')
        plt.figure(1)
        plt.plot(np.arange(len(score_VHL)), score_VHL, label='VHL')
        plt.ylabel('score')
        plt.xlabel('step')

        plt.savefig(str(i) + "score.png")
        plt.clf()

    def plot_cost_finnal(self, i):
        plt.figure(12)
        plt.subplot(221)
        plt.plot(np.arange(len(self.cost_his_emb)), self.cost_his_emb, label='cost_emb')
        plt.subplot(222)
        plt.plot(np.arange(len(self.cost_his_q)), self.cost_his_q, label='cost_q')
        plt.subplot(212)
        plt.plot(np.arange(len(self.cost_his)), self.cost_his,label='cost_all')

        plt.ylabel('Cost')
        plt.xlabel('training steps')
        plt.savefig(str(i) + "loss.png")
        plt.clf()