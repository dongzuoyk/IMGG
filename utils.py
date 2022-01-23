import scanpy as sc
import torch
import random
from annoy import AnnoyIndex
import numpy as np

def data_preprocess(adata, n_top_genes=2000, key='batch', min_genes=600, min_cells=3):
    print('Establishing Adata for Next Step...')
    adata = adata[:, [gene for gene in adata.var_names 
                  if not str(gene).startswith(tuple(['ERCC', 'MT-', 'mt-']))]]
    sc.pp.filter_cells(adata, min_genes=min_genes)
    sc.pp.filter_genes(adata, min_cells=min_cells)
    sc.pp.highly_variable_genes(adata, n_top_genes=n_top_genes, flavor='seurat_v3', batch_key=key)
    hv_genes = adata.var['highly_variable'].tolist()
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    
    hv_adata = adata[:, hv_genes]
    print('PreProcess Done.')
    return hv_adata



def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    random.seed(seed)


def normalize(data: np.float32) -> np.float32:
    norm = data  # (np.exp2(data)-1)
    return norm  # / np.array([np.sqrt(np.sum(np.square(norm), axis=1))]).T


def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Linear") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)



def acquire_pairs(adata_X, adata_Y, k, metric,index_dir):

    X = adata_X.X
    Y = adata_Y.X
                  
    f = X.shape[1]
    t1 = AnnoyIndex(f, metric)
    t2 = AnnoyIndex(f, metric)

    for i in range(len(X)):
        t1.add_item(i, X[i])
    for i in range(len(Y)):
        t2.add_item(i, Y[i])
    t1.build(10)
    t2.build(10)

    mnn_mat = np.bool8(np.zeros((len(X), len(Y))))
    sorted_mat = np.array([t2.get_nns_by_vector(item, k) for item in X])
    for i in range(len(sorted_mat)):
        mnn_mat[i,sorted_mat[i]] = True
    _ = np.bool8(np.zeros((len(X), len(Y))))
    sorted_mat = np.array([t1.get_nns_by_vector(item, k) for item in Y])
    for i in range(len(sorted_mat)):
        _[sorted_mat[i],i] = True
    mnn_mat = np.logical_and(_, mnn_mat)
    pairs = [(x, y) for x, y in zip(*np.where(mnn_mat>0))]
    

    pairs = [(index_dir[adata_X.obs_names[x]], index_dir[adata_Y.obs_names[y]]) for x, y in pairs]

    return pairs



import pandas as pd

from sklearn.metrics import silhouette_score,adjusted_rand_score
from sklearn.cluster import KMeans

def compute_ASW(arr, batch_label, cell_label):

    batch_score = silhouette_score(arr, batch_label)
    celltype_score = silhouette_score(arr, cell_label)


    return batch_score,celltype_score

def compute_ARI(adata):
    
    X = adata.obsm['X_pca']
    Y_batchcluster = adata.obs['batch'].tolist()
    Y_batchnum = len(set(Y_batchcluster))

    Y_cellcluster = adata.obs['celltype'].tolist()
    Y_cellnum = len(set(Y_cellcluster))

    km_batch = KMeans(n_clusters=Y_batchnum, init='k-means++', max_iter=30)
    km_batch.fit(X)
    y_batch = km_batch.predict(X)
    batch_ARI = adjusted_rand_score(Y_batchcluster,y_batch)


    km_cell = KMeans(n_clusters=Y_cellnum, init='k-means++', max_iter=30)
    km_cell.fit(X)
    y_cell = km_cell.predict(X)
    cell_ARI = adjusted_rand_score(Y_cellcluster,y_cell)

    return batch_ARI, cell_ARI