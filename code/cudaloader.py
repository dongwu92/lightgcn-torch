import os
import world
import numpy as np
import torch
import utils


class CudaLoader:
    def __init__(self, dataset, num_epochs):
        self.dataset = dataset
        self.num_epochs = num_epochs
        self.train_temp = '../data/' + world.dataset + '/train_temp/'
        if not os.path.exists(self.train_temp):
            os.mkdir(self.train_temp)
        self.test_temp = '../data/' + world.dataset + '/test_temp/'
        if not os.path.exists(self.test_temp):
            os.mkdir(self.test_temp)
        # load all data into cuda
        if len(os.listdir(self.train_temp)) >= self.num_epochs:
            self._load_train_cuda()
    
    def _load_train_cuda(self):
        self.epochs_data = {}
        for i in range(self.num_epochs):
            users, posItems, negItems = self._sample_train_data_at(i)
            self.epochs_data[i] = [users, posItems, negItems]
        # TODO test

    def _sample_train_data_at(self, epoch):
        epoch_path = self.train_temp + 'S_epoch_' + str(epoch) + '.npy'
        if os.path.exists(epoch_path):
            S = np.load(epoch_path)
        else:
            allusers = list(range(self.dataset.n_users))
            S, sam_time = utils.UniformSample_original(allusers, self.dataset)
            print(f"BPR[sample time][{sam_time[0]:.1f}={sam_time[1]:.2f}+{sam_time[2]:.2f}]")
            np.save(S, epoch_path)
        users = torch.Tensor(S[:, 0]).long()
        posItems = torch.Tensor(S[:, 1]).long()
        negItems = torch.Tensor(S[:, 2]).long()
        users = users.to(world.device)
        posItems = posItems.to(world.device)
        negItems = negItems.to(world.device)
        return users, posItems, negItems
    
    def get_train_data_at(self, epoch):
        if epoch not in self.epochs_data:
            users, posItems, negItems = self._sample_train_data_at(epoch)
            self.epochs_data[epoch] = [users, posItems, negItems]
        return self.epochs_data[epoch]