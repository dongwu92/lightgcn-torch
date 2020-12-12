import scipy.sparse as sp
import numpy as np
from dataloader import Loader
from scipy.sparse import csr_matrix

loader = Loader()
R = loader.UserItemNet.tolil()
twohop = 0.004

adj_user = R.T.todok()
print("generating 2")
rowsum_user = np.array(adj_user.sum(axis=0))
D_user = np.power(rowsum_user, -0.5).flatten()
D_user[np.isinf(D_user)] = 0
Dmat_user = sp.diags(D_user)
print("generating 3")

adj_item = R.todok()
print("generating 4")
rowsum_item = np.array(adj_item.sum(axis=0))
D_item = np.power(rowsum_item, -0.5).flatten()
D_item[np.isinf(D_item)] = 0
Dmat_item = sp.diags(D_item)
print("generating 5")

norm_user = Dmat_item.dot(adj_user).dot(Dmat_user)
norm_item = Dmat_user.dot(adj_item).dot(Dmat_item)

def sparsify_propagation(adj, hop_thres):
    adj_valid = (adj > hop_thres)
    adjx, adjy = adj_valid.nonzero()
    adj_data = np.array(adj[adj_valid])[0]
    norm_adj = csr_matrix((adj_data, (adjx, adjy)), shape=(adj.shape[0], adj.shape[1]))
    return norm_adj

Suu = norm_item.dot(norm_user)
# Suu_valid = (Suu>twohop)
# suux, suuy = Suu_valid.nonzero()
# suu_data = np.array(Suu[Suu_valid])[0]
# norm_uu = csr_matrix((suu_data, (suux, suuy)), shape=(R.shape[0], R.shape[0]))
norm_uu = sparsify_propagation(Suu, twohop)

Svv = norm_user.dot(norm_item)
# Svv_valid = (Svv>twohop)
# svvx, svvy = Svv_valid.nonzero()    
# svv_data = np.array(Svv[Svv_valid])[0]
# norm_vv = csr_matrix((svv_data, (svvx, svvy)), shape=(R.shape[1], R.shape[1]))
norm_vv = sparsify_propagation(Svv, twohop)
