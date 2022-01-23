import networkx as nx
import numpy as np
import scanpy as sc
from torch.utils.data import Dataset
from annoy import AnnoyIndex
import random
import math
from itertools import combinations
from collections import Counter
import time

from utils import acquire_pairs

import multiprocessing
import operator

from anndata import AnnData

def create_pairs_dict(pairs,pairs_dict):
    for x,y in pairs:
        if x not in pairs_dict.keys():
            pairs_dict[x] = [y]
        else:
            pairs_dict[x].append(y)
    return pairs_dict

        


class ScDataset(Dataset):
    def __init__(self, ppd_adata, all_adata, batch_key, mnn_times=5, len_weight=None, self_nbs=None, other_nbs=None,overlap=True,sample_method='max',batch_num=2,exclude_list=[],under_sample=False,under_sample_num=20000,balance_sampling=False):
        
        self.same_batch_pairs = {}
        self.diff_batch_pairs = {}
        
        self.ppd_adata = ppd_adata
        self.pca_adata = AnnData(ppd_adata.obsm['X_pca'],ppd_adata.obs)
        self.all_adata = all_adata

        self.batch_list = self.pca_adata.obs[batch_key].unique().tolist()
        self.index_list = self.pca_adata.obs.index.tolist()
        self.index_dir = {}
        for i in range(len(self.index_list)):
            self.index_dir[self.index_list[i]] = i

        self.sample_num = int(len(self.pca_adata)/len(self.pca_adata.obs[batch_key].unique()))
        if self.sample_num > 3000:
            self.sample_num = 3000
        print("Number of samples per batch: "+str(self.sample_num))
        
        self.balance_sampling = balance_sampling
        start_time = time.time()
        
        # Initialization
        all_list = [i for i in range(len(self.pca_adata))]
        exc_list = {}
        
        same_batch_tag = {}
        
        for batch1,batch2 in combinations(self.batch_list, 2):
            self.same_batch_pairs[batch1] = []
            self.same_batch_pairs[batch2] = []
            self.diff_batch_pairs[str(batch1)+"_"+str(batch2)] = []
            exc_list[str(batch1)+"_"+str(batch2)] = []
            same_batch_tag[batch1] = True
            same_batch_tag[batch2] = True
        # Find nearest neighbors
        for i in range(mnn_times):
            
            print("Times: " + str(i))
            tmp_sample = {}
            for batch1,batch2 in combinations(self.batch_list, 2):
                print(batch1 + "<——>" + batch2)
                
                # Remove paired cells in two batches
                new_list = list(set(exc_list[str(batch1)+"_"+str(batch2)])^set(all_list))

                
                batch_1 = self.pca_adata[new_list,:][self.pca_adata[new_list,:].obs[batch_key]==batch1]
                batch_2 = self.pca_adata[new_list,:][self.pca_adata[new_list,:].obs[batch_key]==batch2]
                
                if other_nbs == None:
                    other_nbs = 1
                
                if self_nbs == None:
                    _ = int(max(min(len(batch_1),len(batch_2)) / 200 ,1))
                    if _ >= 15:
                        self_nbs = 15
                    else:
                        self_nbs = _
                    
                
                if batch1 not in tmp_sample.keys():
                    tmp_1 = np.arange(len(batch_1))
                    np.random.shuffle(tmp_1)
                    tmp_sample[batch1] = tmp_1
                    if same_batch_tag[batch1]:
                        self.same_batch_pairs[batch1] += acquire_pairs(batch_1[tmp_1[:self.sample_num]],batch_1[tmp_1[:self.sample_num]],self_nbs,'angular',self.index_dir)
                        if len(batch_1) <=3000:
                            same_batch_tag[batch1] = False
                    
                if batch2 not in tmp_sample.keys():
                    tmp_2 = np.arange(len(batch_2))
                    np.random.shuffle(tmp_2)
                    tmp_sample[batch2] = tmp_2
                    if same_batch_tag[batch2]:
                        self.same_batch_pairs[batch2] += acquire_pairs(batch_2[tmp_2[:self.sample_num]],batch_2[tmp_2[:self.sample_num]],self_nbs,'angular',self.index_dir)
                        if len(batch_2) <= 3000:
                            same_batch_tag[batch2] = False
                
                index_1 = tmp_sample[batch1]
                index_1 = index_1[index_1<len(batch_1)]
                index_2 = tmp_sample[batch2]
                index_2 = index_2[index_2<len(batch_2)]

                self.diff_batch_pairs[str(batch1)+"_"+str(batch2)] += acquire_pairs(batch_1[index_1[:self.sample_num]],batch_2[index_2[:self.sample_num]],other_nbs,'angular',self.index_dir)

                                    
            for key in self.diff_batch_pairs.keys():
                for k,v in self.diff_batch_pairs[key]:
                    exc_list[key] += [k,v]
        
            
#         # Find the nearest neighbors in the same batch
#         for batch_name in self.batch_list:
#             self.same_batch_pairs[batch_name] = acquire_pairs(self.pca_adata[self.pca_adata.obs[batch_key]==batch_name],self.pca_adata[self.pca_adata.obs[batch_key]==batch_name],self_nbs,'angular',self.index_dir)

        end_time = time.time()
        print("Find MNNs time-consuming :" + str(end_time-start_time))
        
        # Construct connected graphs of similar cells across batches
        self.nodes = []
        self.edges = []
        for pair_name in self.diff_batch_pairs.keys():
            for x,y in self.diff_batch_pairs[pair_name]:
                self.nodes.append(x)
                self.nodes.append(y)
                self.edges.append((x,y,1))
        self.nodes = list(set(self.nodes))
        
        print("The percentage of MNN paired cells in the total cells : "+str(len(self.nodes)/len(self.pca_adata)))
        self.G = nx.Graph()
        self.G.add_nodes_from(self.nodes)
        self.G.add_weighted_edges_from(self.edges)
        self.C = sorted(nx.connected_components(self.G), key=len, reverse=True)

        self.pairs_dict = {} # key:cell id, value: its neighbors id in the same batch
        for key in self.same_batch_pairs.keys():
            self.pairs_dict = create_pairs_dict(self.same_batch_pairs[key],self.pairs_dict)

        for i in range(len(self.pca_adata)):
            if i not in self.pairs_dict.keys():
                self.pairs_dict[i] = [i]

        self.init_dataset(batch_key,len_weight,overlap,sample_method,batch_num,exclude_list,under_sample,under_sample_num)

    def init_dataset(self, batch_key, len_weight, overlap, sample_method, batch_num, exclude_list, under_sample, under_sample_num):
        
        # Sampled subgraph nodes list，each subgraph contains cells from different batches
        graph_list = [] 
        for graph in self.C:
            
            tmp_dict = {} # key：batch name，value：cell id list
            for node in graph:
                try:
                    tmp_dict[self.pca_adata[node].obs[batch_key].tolist()[0]] += [node]
                except:
                    tmp_dict[self.pca_adata[node].obs[batch_key].tolist()[0]] = []
                    tmp_dict[self.pca_adata[node].obs[batch_key].tolist()[0]] += [node]
                    
            # Batches contained in the current connected graph
            in_batch = list(tmp_dict.keys())
            
            # Batches not contained in the current connected graph
            out_batch = list(set(self.batch_list)^set(in_batch))
            
            # List of cell's id sampled in the current connected graph
            tmp_graph_list = []
            if out_batch == []:
                sample_times = int(np.mean([len(cell_id_list) for cell_id_list in tmp_dict.values()]))
                for i in range(sample_times):
                    tmp_graph_list.append([random.choice(cell_id_list) for cell_id_list in tmp_dict.values()])
            else:
                new_tmp_dict = tmp_dict.copy()
                # Find the connected graph where the k-nearest neighbor of the cell in the current connected graph is located
                # Observe whether the connected graph contains cells that do not contain batches in the current connected graph
                for batch_name in tmp_dict.keys():
                    for cell_id in tmp_dict[batch_name]:
                        neighbors = self.pairs_dict[cell_id]
                        if len(neighbors)==1:
                            continue
                        for nb in neighbors:
                            try:
                                nb_graph_nodes = list(self.G.adj[nb])
                                for nb_node in nb_graph_nodes:
                                    if self.pca_adata[nb_node].obs[batch_key].tolist()[0] in out_batch:
                                        try:
                                            new_tmp_dict[self.pca_adata[nb_node].obs[batch_key].tolist()[0]] += [nb_node]
                                        except:
                                            new_tmp_dict[self.pca_adata[nb_node].obs[batch_key].tolist()[0]] = []
                                            new_tmp_dict[self.pca_adata[nb_node].obs[batch_key].tolist()[0]] += [nb_node]
                            except:
                                continue
                sample_times = int(np.mean([len(cell_id_list) for cell_id_list in new_tmp_dict.values()]))
                for i in range(sample_times):
                    tmp_graph_list.append([random.choice(cell_id_list) for cell_id_list in new_tmp_dict.values()])

            graph_list += tmp_graph_list
        
        print("The final number of subgraphs :" + str(len(graph_list)))
        len_list = [len(g) for g in graph_list]
        print(Counter(len_list))
        
        self.source_id = []
        self.target_id = []
        
        for graph in graph_list:
            
            if overlap:
                # max graph sample method
                if sample_method == 'max' and len(graph) != batch_num:
                    continue
                # mean graph sample method
                elif sample_method == 'mean' and len(graph) < 1/2 * batch_num:
                    continue
                # free graph sample method
                elif sample_method == 'free' and len(graph) in exclude_list:
                    continue
            else:
                if (len(graph)==5) and (random.random()<=0.5):
                    continue
                elif (len(graph)==4) and (random.random()<=0.6):
                    continue
                elif (len(graph)==3) and (random.random()<=0.7):
                    continue
                elif (len(graph)==2) and (random.random()<=0.8):
                    continue
                    
            self.source_graph = []
            self.target_graph = []

            for node in graph:
                for i in range(len(graph) * len_weight):
                    self.source_graph.append(random.choice(self.pairs_dict[node]))
                    
            self.target_graph = [[] for i in range(len(self.source_graph))]

            for i in range(len(self.source_graph)):
                for node in graph:
                    self.target_graph[i] += [random.choice(self.pairs_dict[node])]

            self.source_id += self.source_graph
            self.target_id += self.target_graph
        
        if under_sample:
            random.seed(10)
            self.source_id = random.sample(self.source_id, under_sample_num)
            random.seed(10)
            self.target_id = random.sample(self.target_id, under_sample_num)
        
        self.source = self.ppd_adata[self.source_id].X.toarray().squeeze()
        
        
        t3 = time.time()
        
        # Multithreading acceleration
        if self.balance_sampling:
            self.target = [self.ppd_adata[l].X.mean(axis=0).toarray().squeeze() for l in self.target_id]
        else:
            p = multiprocessing.Pool(4)
            self.target = p.map(self.do_something,self.target_id)
            p.close()
            p.join()
        t4 = time.time()
        self.datasize = len(self.source_id)
        print("Dataset size : " + str(len(self.source_id)))
        print("Time-consuming to build dataset: " + str(t4-t3))
        
    def do_something(self,arr_list):
        data = self.ppd_adata[arr_list].X.mean(axis=0).toarray().squeeze()
        return data

#     def __len__(self):
#         return math.ceil(len(self.source_id)/1024)*1024

#     def __getitem__(self, index):
        
#         return self.source[index % len(self.source_id)],self.target[index % len(self.source_id)]
    
    def __len__(self):
        return 10*1024
        # return math.ceil(len(self.source_id)/1024)*1024

    def __getitem__(self, index):
        
        ind = random.randint(0,self.datasize-1)
        return self.source[ind],self.target[ind]