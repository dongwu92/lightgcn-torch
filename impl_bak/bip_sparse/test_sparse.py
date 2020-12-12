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

# Suu = norm_item.dot(norm_user)
# Suu_valid = (Suu>twohop)
# suux, suuy = Suu_valid.nonzero()
# suu_data = np.array(Suu[Suu_valid])[0]
# norm_uu = csr_matrix((suu_data, (suux, suuy)), shape=(R.shape[0], R.shape[0]))
# norm_uu = sparsify_propagation(Suu, twohop)

# Svv = norm_user.dot(norm_item)
# Svv_valid = (Svv>twohop)
# svvx, svvy = Svv_valid.nonzero()    
# svv_data = np.array(Svv[Svv_valid])[0]
# norm_vv = csr_matrix((svv_data, (svvx, svvy)), shape=(R.shape[1], R.shape[1]))
# norm_vv = sparsify_propagation(Svv, twohop)

def generate_sparse_normadj(layer, hop_threshold, previous, origins):
    print('generate normadj at layer#', layer)
    # Su = norm_user, Sv = norm_item
    if layer == 0:
        # save (Sv, Sv * Su) for users and (Su, Su * Sv) for items
        norm_uu = origins[1].dot(origins[3])
        norm_uu = sparsify_propagation(norm_uu, hop_threshold)
        norm_vv = origins[3].dot(origins[1])
        norm_vv = sparsify_propagation(norm_vv, hop_threshold)
        norm_uv = origins[1]
        norm_vu = origins[3]
        print('#Layer 0:::', hop_threshold, len(norm_uu.nonzero()[0]), len(norm_vu.nonzero()[0]), len(norm_vv.nonzero()[0]), len(norm_uv.nonzero()[0]))
    else:
        # norm_uu = org_uu * prv_uu + org_uv * prv_vu
        norm_uu = origins[0].dot(previous[0]) + origins[1].dot(previous[3])
        norm_uu = sparsify_propagation(norm_uu, hop_threshold)
        # norm_uv = org_uu * prv_uv + org_uv * prv_vv
        norm_uv = origins[0].dot(previous[1]) + origins[1].dot(previous[2])
        norm_uv = sparsify_propagation(norm_uv, hop_threshold)
        # norm_vv = org_vv * prv_vv + org_vu * prv_uv
        norm_vv = origins[2].dot(previous[2]) + origins[3].dot(previous[1])
        norm_vv = sparsify_propagation(norm_vv, hop_threshold)
        # norm_vu = org_vv * prv_vu + org_vu * prv_uu
        norm_vu = origins[2].dot(previous[3]) + origins[3].dot(previous[0])
        norm_vu = sparsify_propagation(norm_vu, hop_threshold)
        print('#Layer', layer, ':::', hop_threshold, len(norm_uu.nonzero()[0]), len(norm_vu.nonzero()[0]), len(norm_vv.nonzero()[0]), len(norm_uv.nonzero()[0]))
    return norm_uu, norm_uv, norm_vv, norm_vu

origins = generate_sparse_normadj(0, twohop, [], [None, norm_item, None, norm_user])
previous = origins
