"""
Created on Mar 1, 2020
Pytorch Implementation of LightGCN in
Xiangnan He et al. LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation

@author: Jianbai Ye (gusye@mail.ustc.edu.cn)

Define models here
"""
import world
import torch
from dataloader import BasicDataset
from torch import nn
import numpy as np
import torch.nn.functional as F


class BasicModel(nn.Module):    
    def __init__(self):
        super(BasicModel, self).__init__()
    
    def getUsersRating(self, users):
        raise NotImplementedError
    
class PairWiseModel(BasicModel):
    def __init__(self):
        super(PairWiseModel, self).__init__()
    def bpr_loss(self, users, pos, neg):
        """
        Parameters:
            users: users list 
            pos: positive items for corresponding users
            neg: negative items for corresponding users
        Return:
            (log-loss, l2-loss)
        """
        raise NotImplementedError
    
class PureMF(BasicModel):
    def __init__(self, 
                 config:dict, 
                 dataset:BasicDataset):
        super(PureMF, self).__init__()
        self.num_users  = dataset.n_users
        self.num_items  = dataset.m_items
        self.latent_dim = config['latent_dim_rec']
        self.f = nn.Sigmoid()
        self.__init_weight()
        
    def __init_weight(self):
        self.embedding_user = torch.nn.Embedding(
            num_embeddings=self.num_users, embedding_dim=self.latent_dim)
        self.embedding_item = torch.nn.Embedding(
            num_embeddings=self.num_items, embedding_dim=self.latent_dim)
        print("using Normal distribution N(0,1) initialization for PureMF")
        
    def getUsersRating(self, users):
        users = users.long()
        users_emb = self.embedding_user(users)
        items_emb = self.embedding_item.weight
        scores = torch.matmul(users_emb, items_emb.t())
        return self.f(scores)
    
    def bpr_loss(self, users, pos, neg):
        users_emb = self.embedding_user(users.long())
        pos_emb   = self.embedding_item(pos.long())
        neg_emb   = self.embedding_item(neg.long())
        pos_scores= torch.sum(users_emb*pos_emb, dim=1)
        neg_scores= torch.sum(users_emb*neg_emb, dim=1)
        loss = torch.mean(nn.functional.softplus(neg_scores - pos_scores))
        reg_loss = (1/2)*(users_emb.norm(2).pow(2) + 
                          pos_emb.norm(2).pow(2) + 
                          neg_emb.norm(2).pow(2))/float(len(users))
        return loss, reg_loss
        
    def forward(self, users, items):
        users = users.long()
        items = items.long()
        users_emb = self.embedding_user(users)
        items_emb = self.embedding_item(items)
        scores = torch.sum(users_emb*items_emb, dim=1)
        return self.f(scores)

class LightGCN(BasicModel):
    def __init__(self, 
                 config:dict, 
                 dataset:BasicDataset):
        super(LightGCN, self).__init__()
        self.config = config
        self.dataset : dataloader.BasicDataset = dataset
        self.__init_weight()
        self.sample = 4
        self.dropnode_rate = 0.5
        self.tem = 0.5
        self.lam = 1.

    def __init_weight(self):
        self.num_users  = self.dataset.n_users
        self.num_items  = self.dataset.m_items
        self.latent_dim = self.config['latent_dim_rec']
        self.n_layers = self.config['lightGCN_n_layers']
        self.keep_prob = self.config['keep_prob']
        self.A_split = self.config['A_split']
        self.embedding_user = torch.nn.Embedding(
            num_embeddings=self.num_users, embedding_dim=self.latent_dim)
        self.embedding_item = torch.nn.Embedding(
            num_embeddings=self.num_items, embedding_dim=self.latent_dim)
        if self.config['pretrain'] == 0:
#             nn.init.xavier_uniform_(self.embedding_user.weight, gain=1)
#             nn.init.xavier_uniform_(self.embedding_item.weight, gain=1)
#             print('use xavier initilizer')
# random normal init seems to be a better choice when lightGCN actually don't use any non-linear activation function
            nn.init.normal_(self.embedding_user.weight, std=0.1)
            nn.init.normal_(self.embedding_item.weight, std=0.1)
            world.cprint('use NORMAL distribution initilizer')
        else:
            self.embedding_user.weight.data.copy_(torch.from_numpy(self.config['user_emb']))
            self.embedding_item.weight.data.copy_(torch.from_numpy(self.config['item_emb']))
            print('use pretarined data')
        self.f = nn.Sigmoid()
        self.Graph = self.dataset.getSparseGraph()
        print(f"lgn is already to go(dropout:{self.config['dropout']})")

        # print("save_txt")
    def __dropout_x(self, x, keep_prob):
        size = x.size()
        index = x.indices().t()
        values = x.values()
        random_index = torch.rand(len(values)) + keep_prob
        random_index = random_index.int().bool()
        index = index[random_index]
        values = values[random_index]/keep_prob
        g = torch.sparse.FloatTensor(index.t(), values, size)
        return g
    
    def __dropout(self, keep_prob):
        if self.A_split:
            graph = []
            for g in self.Graph:
                graph.append(self.__dropout_x(g, keep_prob))
        else:
            graph = self.__dropout_x(self.Graph, keep_prob)
        return graph




    def propagate(self, feature):
        #feature = F.dropout(feature, args.dropout, training=training)
        x = feature
        y = feature
        for i in range(self.n_layers):
            x = torch.spmm(self.Graph, x).detach_()
            y.add_(x)
        return y.div_(self.n_layers+1.0).detach_()

    def rand_prop(self, features, training):
        n = features.shape[0]
        drop_rate = self.dropnode_rate
        drop_rates = torch.FloatTensor(np.ones(n) * drop_rate)
        if training:
            masks = torch.bernoulli(1. - drop_rates).unsqueeze(1)
            features = masks.cuda() * features
        else:
            features = features * (1. - drop_rate)
        features = self.propagate(features)
        return features
        
    def consis_loss(self, logps):
        ps = [torch.exp(p) for p in logps]
        sum_p = 0.
        for p in ps:
            sum_p = sum_p + p
        avg_p = sum_p/len(ps)
        #p2 = torch.exp(logp2)
        
        sharp_p = (torch.pow(avg_p, 1./self.tem) / torch.sum(torch.pow(avg_p, 1./self.tem), dim=1, keepdim=True)).detach()
        loss = 0.
        for p in ps:
            loss += torch.mean((p-sharp_p).pow(2).sum(1))
        loss = loss/len(ps)
        return self.lam * loss

    def computer(self, training):
        """
        propagate methods for lightGCN
        """       
        users_emb = self.embedding_user.weight
        items_emb = self.embedding_item.weight
        all_emb = torch.cat([users_emb, items_emb])

        if training:
            user_list, item_list = [], []
            for k in range(self.sample):
                light_out = self.rand_prop(all_emb, training=True)
                users, items = torch.split(light_out, [self.num_users, self.num_items])
                user_list.append(users)
                item_list.append(items)
            return user_list, item_list
        else:
            light_out = self.rand_prop(all_emb, training=True)
            users, items = torch.split(light_out, [self.num_users, self.num_items])
            return users, items
    


    def getUsersRating(self, users):
        # for inference
        all_users, all_items = self.computer(False)
        users_emb = all_users[users.long()]
        items_emb = all_items
        rating = self.f(torch.matmul(users_emb, items_emb.t()))
        return rating
    
    def getEmbedding(self, users, pos_items, neg_items):
        # for training
        all_users_list, all_items_list = self.computer(True)
        all_embs = []
        for all_users, all_items in zip(all_users_list, all_items_list):
            users_emb = all_users[users]
            pos_emb = all_items[pos_items]
            neg_emb = all_items[neg_items]
            users_emb_ego = self.embedding_user(users)
            pos_emb_ego = self.embedding_item(pos_items)
            neg_emb_ego = self.embedding_item(neg_items)
            all_embs.append([users_emb, pos_emb, neg_emb, users_emb_ego, pos_emb_ego, neg_emb_ego])
        return all_embs
    
    def bpr_loss(self, users, pos, neg):
        all_embs = self.getEmbedding(users.long(), pos.long(), neg.long())
        # (users_emb, pos_emb, neg_emb, 
        # userEmb0,  posEmb0, negEmb0) = self.getEmbedding(users.long(), pos.long(), neg.long())
        loss = 0.
        pos_list, neg_list = [], []
        for users_emb, pos_emb, neg_emb, userEmb0, posEmb0, negEmb0 in all_embs:
            reg_loss = (1/2)*(userEmb0.norm(2).pow(2) + 
                            posEmb0.norm(2).pow(2)  +
                            negEmb0.norm(2).pow(2))/float(len(users))
            pos_scores = torch.mul(users_emb, pos_emb)
            pos_list.append(torch.log_softmax(pos_scores), dim=-1)
            pos_scores = torch.sum(pos_scores, dim=1)
            neg_scores = torch.mul(users_emb, neg_emb)
            neg_list.append(torch.log_softmax(neg_scores), dim=-1)
            neg_scores = torch.sum(neg_scores, dim=1)
            loss += torch.mean(torch.nn.functional.softplus(neg_scores - pos_scores))
        loss += self.consis_loss(pos_list) + self.consis_loss(neg_scores)

        # #bipartite_loss = torch.mean(torch.square(userEmb0 - posEmb0)) + torch.mean(torch.square(userEmb0 - negEmb0))
        # ue = F.softmax(userEmb0, dim=-1)
        # pve = F.softmax(posEmb0, dim=-1)
        # nve = F.softmax(negEmb0, dim=-1)
        # bipartite_loss = torch.mean(ue * torch.log(pve)) + torch.mean(ue * torch.log(nve)) + torch.mean(pve * torch.log(ue)) + torch.mean(nve * torch.log(ue))
        # loss += self.config['ceweight'] * bipartite_loss
        
        return loss, reg_loss
       
    def forward(self, users, items):
        # compute embedding
        all_users, all_items = self.computer()
        print('forward!!!!!!!!!!!!!!!!!!!')
        #all_users, all_items = self.computer()
        users_emb = all_users[users]
        items_emb = all_items[items]
        inner_pro = torch.mul(users_emb, items_emb)
        gamma     = torch.sum(inner_pro, dim=1)
        return gamma
