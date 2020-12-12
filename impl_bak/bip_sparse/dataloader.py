"""
Created on Mar 1, 2020
Pytorch Implementation of LightGCN in
Xiangnan He et al. LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation

@author: Shuxian Bi (stanbi@mail.ustc.edu.cn),Jianbai Ye (gusye@mail.ustc.edu.cn)
Design Dataset here
Every dataset's index has to start at 0
"""
import os
from os.path import join
import sys
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from scipy.sparse import csr_matrix
import scipy.sparse as sp
import world
from world import cprint
from time import time

class BasicDataset(Dataset):
    def __init__(self):
        print("init dataset")
    
    @property
    def n_users(self):
        raise NotImplementedError
    
    @property
    def m_items(self):
        raise NotImplementedError
    
    @property
    def trainDataSize(self):
        raise NotImplementedError
    
    @property
    def testDict(self):
        raise NotImplementedError
    
    @property
    def allPos(self):
        raise NotImplementedError
    
    def getUserItemFeedback(self, users, items):
        raise NotImplementedError
    
    def getUserPosItems(self, users):
        raise NotImplementedError
    
    def getUserNegItems(self, users):
        """
        not necessary for large dataset
        it's stupid to return all neg items in super large dataset
        """
        raise NotImplementedError
    
    def getSparseGraph(self):
        """
        build a graph in torch.sparse.IntTensor.
        Details in NGCF's matrix form
        A = 
            |I,   R|
            |R^T, I|
        """
        raise NotImplementedError

class LastFM(BasicDataset):
    """
    Dataset type for pytorch \n
    Incldue graph information
    LastFM dataset
    """
    def __init__(self, path="../data/lastfm"):
        # train or test
        cprint("loading [last fm]")
        self.mode_dict = {'train':0, "test":1}
        self.mode    = self.mode_dict['train']
        # self.n_users = 1892
        # self.m_items = 4489
        trainData = pd.read_table(join(path, 'data1.txt'), header=None)
        # print(trainData.head())
        testData  = pd.read_table(join(path, 'test1.txt'), header=None)
        # print(testData.head())
        trustNet  = pd.read_table(join(path, 'trustnetwork.txt'), header=None).to_numpy()
        # print(trustNet[:5])
        trustNet -= 1
        trainData-= 1
        testData -= 1
        self.trustNet  = trustNet
        self.trainData = trainData
        self.testData  = testData
        self.trainUser = np.array(trainData[:][0])
        self.trainUniqueUsers = np.unique(self.trainUser)
        self.trainItem = np.array(trainData[:][1])
        # self.trainDataSize = len(self.trainUser)
        self.testUser  = np.array(testData[:][0])
        self.testUniqueUsers = np.unique(self.testUser)
        self.testItem  = np.array(testData[:][1])
        self.Graph = None
        print(f"LastFm Sparsity : {(len(self.trainUser) + len(self.testUser))/self.n_users/self.m_items}")
        
        # (users,users)
        self.socialNet    = csr_matrix((np.ones(len(trustNet)), (trustNet[:,0], trustNet[:,1]) ), shape=(self.n_users,self.n_users))
        # (users,items), bipartite graph
        self.UserItemNet  = csr_matrix((np.ones(len(self.trainUser)), (self.trainUser, self.trainItem) ), shape=(self.n_users,self.m_items)) 
        
        # pre-calculate
        self._allPos = self.getUserPosItems(list(range(self.n_users)))
        self.allNeg = []
        allItems    = set(range(self.m_items))
        for i in range(self.n_users):
            pos = set(self._allPos[i])
            neg = allItems - pos
            self.allNeg.append(np.array(list(neg)))
        self.__testDict = self.__build_test()

    @property
    def n_users(self):
        return 1892
    
    @property
    def m_items(self):
        return 4489
    
    @property
    def trainDataSize(self):
        return len(self.trainUser)
    
    @property
    def testDict(self):
        return self.__testDict

    @property
    def allPos(self):
        return self._allPos

    def getSparseGraph(self):
        if self.Graph is None:
            user_dim = torch.LongTensor(self.trainUser)
            item_dim = torch.LongTensor(self.trainItem)
            
            first_sub = torch.stack([user_dim, item_dim + self.n_users])
            second_sub = torch.stack([item_dim+self.n_users, user_dim])
            index = torch.cat([first_sub, second_sub], dim=1)
            data = torch.ones(index.size(-1)).int()
            self.Graph = torch.sparse.IntTensor(index, data, torch.Size([self.n_users+self.m_items, self.n_users+self.m_items]))
            dense = self.Graph.to_dense()
            D = torch.sum(dense, dim=1).float()
            D[D==0.] = 1.
            D_sqrt = torch.sqrt(D).unsqueeze(dim=0)
            dense = dense/D_sqrt
            dense = dense/D_sqrt.t()
            index = dense.nonzero()
            data  = dense[dense >= 1e-9]
            assert len(index) == len(data)
            self.Graph = torch.sparse.FloatTensor(index.t(), data, torch.Size([self.n_users+self.m_items, self.n_users+self.m_items]))
            self.Graph = self.Graph.coalesce().to(world.device)
        return self.Graph

    def __build_test(self):
        """
        return:
            dict: {user: [items]}
        """
        test_data = {}
        for i, item in enumerate(self.testItem):
            user = self.testUser[i]
            if test_data.get(user):
                test_data[user].append(item)
            else:
                test_data[user] = [item]
        return test_data
    
    def getUserItemFeedback(self, users, items):
        """
        users:
            shape [-1]
        items:
            shape [-1]
        return:
            feedback [-1]
        """
        # print(self.UserItemNet[users, items])
        return np.array(self.UserItemNet[users, items]).astype('uint8').reshape((-1, ))
    
    def getUserPosItems(self, users):
        posItems = []
        for user in users:
            posItems.append(self.UserItemNet[user].nonzero()[1])
        return posItems
    
    def getUserNegItems(self, users):
        negItems = []
        for user in users:
            negItems.append(self.allNeg[user])
        return negItems
            
    
    
    def __getitem__(self, index):
        user = self.trainUniqueUsers[index]
        # return user_id and the positive items of the user
        return user
    
    def switch2test(self):
        """
        change dataset mode to offer test data to dataloader
        """
        self.mode = self.mode_dict['test']
    
    def __len__(self):
        return len(self.trainUniqueUsers)

class Loader(BasicDataset):
    """
    Dataset type for pytorch \n
    Incldue graph information
    gowalla dataset
    """

    def __init__(self,config = world.config,path="../data/gowalla"):
        # train or test
        cprint(f'loading [{path}]')
        self.split = config['A_split']
        self.folds = config['A_n_fold']
        self.twohop = config['twohop']
        self.num_layers = config['lightGCN_n_layers']
        self.mode_dict = {'train': 0, "test": 1}
        self.mode = self.mode_dict['train']
        self.n_user = 0
        self.m_item = 0
        train_file = path + '/train.txt'
        test_file = path + '/test.txt'
        self.path = path
        trainUniqueUsers, trainItem, trainUser = [], [], []
        testUniqueUsers, testItem, testUser = [], [], []
        self.traindataSize = 0
        self.testDataSize = 0

        with open(train_file) as f:
            for l in f.readlines():
                if len(l) > 0:
                    l = l.strip('\n').split(' ')
                    items = [int(i) for i in l[1:]]
                    uid = int(l[0])
                    trainUniqueUsers.append(uid)
                    trainUser.extend([uid] * len(items))
                    trainItem.extend(items)
                    self.m_item = max(self.m_item, max(items))
                    self.n_user = max(self.n_user, uid)
                    self.traindataSize += len(items)
        self.trainUniqueUsers = np.array(trainUniqueUsers)
        self.trainUser = np.array(trainUser)
        self.trainItem = np.array(trainItem)

        with open(test_file) as f:
            for l in f.readlines():
                if len(l) > 0:
                    l = l.strip('\n').split(' ')
                    items = [int(i) for i in l[1:]]
                    uid = int(l[0])
                    testUniqueUsers.append(uid)
                    testUser.extend([uid] * len(items))
                    testItem.extend(items)
                    self.m_item = max(self.m_item, max(items))
                    self.n_user = max(self.n_user, uid)
                    self.testDataSize += len(items)
        self.m_item += 1
        self.n_user += 1
        self.testUniqueUsers = np.array(testUniqueUsers)
        self.testUser = np.array(testUser)
        self.testItem = np.array(testItem)
        
        self.Graph_user, self.Graph_item = None, None
        print(f"{self.trainDataSize} interactions for training")
        print(f"{self.testDataSize} interactions for testing")
        print(f"{world.dataset} Sparsity : {(self.trainDataSize + self.testDataSize) / self.n_users / self.m_items}")

        # (users,items), bipartite graph
        self.UserItemNet = csr_matrix((np.ones(len(self.trainUser)), (self.trainUser, self.trainItem)),
                                      shape=(self.n_user, self.m_item))
        self.users_D = np.array(self.UserItemNet.sum(axis=1)).squeeze()
        self.users_D[self.users_D == 0.] = 1
        self.items_D = np.array(self.UserItemNet.sum(axis=0)).squeeze()
        self.items_D[self.items_D == 0.] = 1.
        # pre-calculate
        self._allPos = self.getUserPosItems(list(range(self.n_user)))
        self.__testDict = self.__build_test()
        print(f"{world.dataset} is ready to go")

    @property
    def n_users(self):
        return self.n_user
    
    @property
    def m_items(self):
        return self.m_item
    
    @property
    def trainDataSize(self):
        return self.traindataSize
    
    @property
    def testDict(self):
        return self.__testDict

    @property
    def allPos(self):
        return self._allPos

    def _split_A_hat(self,A):
        A_fold = []
        fold_len = (self.n_users + self.m_items) // self.folds
        for i_fold in range(self.folds):
            start = i_fold*fold_len
            if i_fold == self.folds - 1:
                end = self.n_users + self.m_items
            else:
                end = (i_fold + 1) * fold_len
            A_fold.append(self._convert_sp_mat_to_sp_tensor(A[start:end]).coalesce().to(world.device))
        return A_fold

    def _convert_sp_mat_to_sp_tensor(self, X):
        coo = X.tocoo().astype(np.float32)
        row = torch.Tensor(coo.row).long()
        col = torch.Tensor(coo.col).long()
        index = torch.stack([row, col])
        data = torch.FloatTensor(coo.data)
        return torch.sparse.FloatTensor(index, data, torch.Size(coo.shape))

    def sparsify_propagation(self, adj, hop_thres):
        adj_valid = (adj > hop_thres)
        adjx, adjy = adj_valid.nonzero()
        adj_data = np.array(adj[adj_valid])[0]
        norm_adj = csr_matrix((adj_data, (adjx, adjy)), shape=(adj.shape[0], adj.shape[1]))
        return norm_adj

    def generate_sparse_normadj(self, layer, hop_threshold, previous, origins):
        # previous: [uu_k-1, uv_k-1, vv_k-1, vu_k-1]; origins: [Suu, Suv, Svv, Svu]
        uu_path = self.path + '/s_pre_adj_uu_' + str(self.twohop) + '_' + str(layer) + '.npz'
        uv_path = self.path + '/s_pre_adj_uv_' + str(self.twohop) + '_' + str(layer) + '.npz'
        vv_path = self.path + '/s_pre_adj_vv_' + str(self.twohop) + '_' + str(layer) + '.npz'
        vu_path = self.path + '/s_pre_adj_vu_' + str(self.twohop) + '_' + str(layer) + '.npz'
        try:
            # Svu = norm_user (V X U), Suv = norm_item (U X V)
            norm_uu = sp.load_npz(uu_path) # Suu        (U X U)
            norm_uv = sp.load_npz(uv_path) # Suv        (U X V)
            norm_vv = sp.load_npz(vv_path) # Svv        (V X V)
            norm_vu = sp.load_npz(vu_path) # Svu        (V X U)
        except:
            print('generate normadj at layer#', layer)
            # Su = norm_user, Sv = norm_item
            if layer == 0:
                # save (Sv, Sv * Su) for users and (Su, Su * Sv) for items
                norm_uu = origins[1].dot(origins[3])
                norm_uu = self.sparsify_propagation(norm_uu, hop_threshold)
                norm_vv = origins[3].dot(origins[1])
                norm_vv = self.sparsify_propagation(norm_vv, hop_threshold)
                norm_uv = origins[1]
                norm_vu = origins[3]
                print('#Layer 0:::', hop_threshold, len(norm_uu.nonzero()[0]), len(norm_vu.nonzero()[0]), len(norm_vv.nonzero()[0]), len(norm_uv.nonzero()[0]))
            else:
                # norm_uu = org_uu * prv_uu + org_uv * prv_vu
                norm_uu = origins[0].dot(previous[0]) + origins[1].dot(previous[3])
                norm_uu = self.sparsify_propagation(norm_uu, hop_threshold)
                # norm_uv = org_uu * prv_uv + org_uv * prv_vv
                norm_uv = origins[0].dot(previous[1]) + origins[1].dot(previous[2])
                norm_uv = self.sparsify_propagation(norm_uv, hop_threshold)
                # norm_vv = org_vv * prv_vv + org_vu * prv_uv
                norm_vv = origins[2].dot(previous[2]) + origins[3].dot(previous[1])
                norm_vv = self.sparsify_propagation(norm_vv, hop_threshold)
                # norm_vu = org_vv * prv_vu + org_vu * prv_uu
                norm_vu = origins[2].dot(previous[3]) + origins[3].dot(previous[0])
                norm_vu = self.sparsify_propagation(norm_vu, hop_threshold)
                print('#Layer', layer, ':::', hop_threshold, len(norm_uu.nonzero()[0]), len(norm_vu.nonzero()[0]), len(norm_vv.nonzero()[0]), len(norm_uv.nonzero()[0]))
            sp.save_npz(uu_path, norm_uu)
            sp.save_npz(uv_path, norm_uv)
            sp.save_npz(vv_path, norm_vv)
            sp.save_npz(vu_path, norm_vu)
        return norm_uu, norm_uv, norm_vv, norm_vu

    def convert_spmat_to_Graph(self, mats):
        tmp_Graphs = []
        for mat in mats:
            tmp_Graph = self._convert_sp_mat_to_sp_tensor(mat)
            tmp_Graphs.append(tmp_Graph.coalesce().to(world.device))
        return tmp_Graphs
        
    def getSparseGraph(self):
        print("loading adjacency matrix")
        s = time()
        print("generating 1")
        R = self.UserItemNet.tolil()

        adj_user = R.T.todok()
        rowsum_user = np.array(adj_user.sum(axis=0))
        D_user = np.power(rowsum_user, -0.5).flatten()
        D_user[np.isinf(D_user)] = 0
        Dmat_user = sp.diags(D_user)

        adj_item = R.todok()
        rowsum_item = np.array(adj_item.sum(axis=0))
        D_item = np.power(rowsum_item, -0.5).flatten()
        D_item[np.isinf(D_item)] = 0
        Dmat_item = sp.diags(D_item)

        norm_user = Dmat_item.dot(adj_user).dot(Dmat_user)
        norm_item = Dmat_user.dot(adj_item).dot(Dmat_item)
        print("generating 2")

        origins = self.generate_sparse_normadj(0, self.twohop, [], [None, norm_item, None, norm_user])
        previous = origins
        self.Graphs = [self.convert_spmat_to_Graph(previous)]
        print("generating 3")
        for layer in range(1, self.num_layers):
            # thres = self.twohop / (4 * layer)
            previous = self.generate_sparse_normadj(layer, self.twohop / layer, previous, origins)
            self.Graphs.append(self.convert_spmat_to_Graph(previous))

        print("generating 4")
        end = time()
        print(f"costing {end-s}s, saved norm_mat...")
        return self.Graphs

    def __build_test(self):
        """
        return:
            dict: {user: [items]}
        """
        test_data = {}
        for i, item in enumerate(self.testItem):
            user = self.testUser[i]
            if test_data.get(user):
                test_data[user].append(item)
            else:
                test_data[user] = [item]
        return test_data

    def getUserItemFeedback(self, users, items):
        """
        users:
            shape [-1]
        items:
            shape [-1]
        return:
            feedback [-1]
        """
        # print(self.UserItemNet[users, items])
        return np.array(self.UserItemNet[users, items]).astype('uint8').reshape((-1,))

    def getUserPosItems(self, users):
        posItems = []
        for user in users:
            posItems.append(self.UserItemNet[user].nonzero()[1])
        return posItems

    # def getUserNegItems(self, users):
    #     negItems = []
    #     for user in users:
    #         negItems.append(self.allNeg[user])
    #     return negItems
