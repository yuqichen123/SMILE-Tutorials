# Tutorial 2 plus: Integrating adjacent DLPFC slices with one slice has annotation
This tutorial demonstrates semi-SMILE's ablility to integrate two adjacent slices (151674 and 151675). The slices are sampled from human dorsolateral prefrontal cortex (DLPFC) and the processed data can be downloaded from <https://figshare.com/articles/dataset/DLPFC_slices_and_reference_scRNA-seq_data/27987548>


```python
import warnings
warnings.filterwarnings('ignore')
```
```python
from stSMILE import SMILE
```

```python
import scanpy as sc
import anndata as ad
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import squidpy as sq
import scipy.sparse as sp
from scipy import sparse
from scipy.sparse import csr_matrix
import math
import torch
import torch.nn as nn
import time
import torch.nn.functional as F
from itertools import chain
from scanpy import read_10x_h5
import torch.optim as optim
import sklearn
from sklearn.neighbors import kneighbors_graph
import gudhi
import networkx as nx
from torch_geometric.nn import GCNConv
import random
import os
import json 
import matplotlib.image as mpimg
```

## Load data


```python
section_ids = ['151674','151675']
```


```python
def label_to_int(adataA, label_list, label_name):
    adata_label = np.array(adataA.obs[label_name].copy())
    for i in range(len(label_list)):
        need_index = np.where(adataA.obs[label_name]==label_list[i])[0]
        if len(need_index):
            adata_label[need_index] = i
    adataA.obs['ref'] = adata_label
    return adataA
```


```python
adata_l = []
for i in range(len(section_ids)):
    adata_i = sc.read_h5ad('./dataset/DLPFC/DLPFC_'+ section_ids[i]+'_ST_final.h5ad')
    adata_i.obs_names = [x+'_'+section_ids[i] for x in adata_i.obs_names]
    adata_l.append(adata_i)
```


```python
# convert label to int
label_list = ['Layer1', 'Layer2', 'Layer3', 'Layer4', 'Layer5', 'Layer6', 'WM']
```


```python
for i in range(len(section_ids)):
    adata_l[i] = label_to_int(adata_l[i], label_list, 'Ground Truth')
```


```python
adata0_sc = sc.read_h5ad('./dataset/DLPFC/DLPFC_sc_final.h5ad') 
```


```python
adata0_sc
```
    AnnData object with n_obs × n_vars = 19764 × 3010
        obs: 'orig.ident', 'nCount_RNA', 'nFeature_RNA', 'cell_type', 'cell_subtype', 'subject', 'condition', 'batch', 'n_genes', 'ref'
        var: 'features', 'n_cells', 'n_counts'
        uns: 'rank_genes_groups'
        obsm: 'X_pca'
        obsp: 'adj_f'




```python
label0_list = adata0_sc.obs['cell_subtype'].tolist()
adata0_label_new = adata0_sc.obs['cell_subtype'].tolist()
for i in range(len(label0_list)):
    need_index = np.where(adata0_sc.obs['cell_subtype'] == label0_list[i])[0]    
    if len(need_index):
        for p in range(len(need_index)):
            adata0_label_new[need_index[p]] = i  
```


```python
adata0_sc.obs['ref'] = pd.Series(adata0_label_new, index = adata0_sc.obs['cell_subtype'].index)
adata0_sc.obs['ref'] = adata0_sc.obs['ref'].astype(str)
adata0_sc.obs['ref'] = adata0_sc.obs['ref'].astype('category')
adata0_sc.obs['Ground Truth'] = adata0_sc.obs['cell_subtype']
```


```python
adata_l.append(adata0_sc)
len(adata_l)
```

    3



## Run SMILE

```python
tag_l = ['ST_ref','ST','single cell']
in_features = len(adata_l[0].var.index)
hidden_features = 512
out_features = 50
feature_method = 'GCNConv'
alpha = 0.001
beta = 1 
lamb = 0.01 
theta = 0.001 
gamma = 10 # reconstruct 
spatial_regularization_strength= 0.9
lr=1e-3
subepochs=100
epochs=200
max_patience=50
min_stop=20
random_seed=2024
gpu=0
regularization_acceleration=True
edge_subset_sz=1000000
add_topology = True
add_feature = False
add_image = False
add_sc = True
multiscale = True
anchor_type = None
anchors_all = False
use_rep_anchor = 'embedding'
align_method = 'MMD'
anchor_size=8000
iter_comb= None
n_clusters_l = [7]
edge_weights = [1,0.1,0.1]
class_rep = 'reconstruct'
```


```python
adata_l = SMILE(adata_l, tag_l, section_ids, multiscale,  n_clusters_l, in_features, feature_method, hidden_features, out_features, iter_comb, anchors_all, use_rep_anchor, alpha, beta, lamb, theta, gamma,edge_weights, add_topology, add_feature, add_image, add_sc, spatial_regularization_strength, lr=lr, subepochs=subepochs, epochs=epochs, class_rep = class_rep)
```

    Pretraining to extract embeddings of spots...
    epoch   0: train spatial C loss: 1.9644, train F loss: 38.8634,
    epoch  10: train spatial C loss: 0.4374, train F loss: 30.2184,
    epoch  20: train spatial C loss: 0.2435, train F loss: 24.9033,
    epoch  30: train spatial C loss: 0.1987, train F loss: 21.5896,
    epoch  40: train spatial C loss: 0.1731, train F loss: 19.4320,
    epoch  50: train spatial C loss: 0.1590, train F loss: 17.9552,
    epoch  60: train spatial C loss: 0.1484, train F loss: 16.8943,
    epoch  70: train spatial C loss: 0.1384, train F loss: 16.0920,
    epoch  80: train spatial C loss: 0.1306, train F loss: 15.4624,
    epoch  90: train spatial C loss: 0.1193, train F loss: 14.9472,
    spatial data classification: Avg Accuracy = 95.405775%
    Training classifier...
    Training classifier...
    epoch   0: overall loss: 9.3433,sc classifier loss: 3.5075,representation loss: 0.5836,within spatial regularization loss: 0.1017
    epoch  10: overall loss: 3.1604,sc classifier loss: 2.2548,representation loss: 0.0906,within spatial regularization loss: 0.0657
    epoch  20: overall loss: 2.2735,sc classifier loss: 1.7549,representation loss: 0.0518,within spatial regularization loss: 0.0760
    epoch  30: overall loss: 1.8128,sc classifier loss: 1.3729,representation loss: 0.0440,within spatial regularization loss: 0.0916
    epoch  40: overall loss: 1.4819,sc classifier loss: 1.0856,representation loss: 0.0396,within spatial regularization loss: 0.1009
    epoch  50: overall loss: 1.2719,sc classifier loss: 0.8814,representation loss: 0.0390,within spatial regularization loss: 0.1067
    epoch  60: overall loss: 1.1300,sc classifier loss: 0.7338,representation loss: 0.0396,within spatial regularization loss: 0.1048
    epoch  70: overall loss: 1.0427,sc classifier loss: 0.6323,representation loss: 0.0410,within spatial regularization loss: 0.1096
    epoch  80: overall loss: 0.9401,sc classifier loss: 0.5597,representation loss: 0.0380,within spatial regularization loss: 0.1060
    epoch  90: overall loss: 0.8475,sc classifier loss: 0.5016,representation loss: 0.0346,within spatial regularization loss: 0.1101
    epoch 100: overall loss: 0.7889,sc classifier loss: 0.4497,representation loss: 0.0339,within spatial regularization loss: 0.1111
    epoch 110: overall loss: 0.7915,sc classifier loss: 0.4107,representation loss: 0.0381,within spatial regularization loss: 0.1129
    epoch 120: overall loss: 0.7123,sc classifier loss: 0.3794,representation loss: 0.0333,within spatial regularization loss: 0.1124
    epoch 130: overall loss: 0.6761,sc classifier loss: 0.3556,representation loss: 0.0320,within spatial regularization loss: 0.1119
    epoch 140: overall loss: 0.6583,sc classifier loss: 0.3323,representation loss: 0.0326,within spatial regularization loss: 0.1123
    epoch 150: overall loss: 0.7115,sc classifier loss: 0.3278,representation loss: 0.0384,within spatial regularization loss: 0.1064
    epoch 160: overall loss: 0.8094,sc classifier loss: 0.3252,representation loss: 0.0484,within spatial regularization loss: 0.1026
    epoch 170: overall loss: 0.6518,sc classifier loss: 0.2978,representation loss: 0.0354,within spatial regularization loss: 0.1112
    epoch 180: overall loss: 0.5971,sc classifier loss: 0.2805,representation loss: 0.0317,within spatial regularization loss: 0.1119
    epoch 190: overall loss: 0.5805,sc classifier loss: 0.2646,representation loss: 0.0316,within spatial regularization loss: 0.1114
    single cell data classification: Avg Accuracy = 93.326253%


    R[write to console]:                    __           __ 
       ____ ___  _____/ /_  _______/ /_
      / __ `__ \/ ___/ / / / / ___/ __/
     / / / / / / /__/ / /_/ (__  ) /_  
    /_/ /_/ /_/\___/_/\__,_/____/\__/   version 6.1.1
    Type 'citation("mclust")' for citing this R package in publications.
    


    fitting ...
      |======================================================================| 100%
    Identifying anchors...
    Processing datasets (0, 1)
    0.8442973272034443
    The ratio of filtered mnn pairs: 0.8366676770738608
    Aligning by anchors...
    epoch 100: total loss:14.5369, train F loss: 14.5250, train C loss: 1.9445, train D loss: 0.0120
    epoch 110: total loss:14.1393, train F loss: 14.1331, train C loss: 0.2266, train D loss: 0.0063
    epoch 120: total loss:13.7923, train F loss: 13.7883, train C loss: 0.1629, train D loss: 0.0040
    epoch 130: total loss:13.4869, train F loss: 13.4834, train C loss: 0.1286, train D loss: 0.0034
    epoch 140: total loss:13.2132, train F loss: 13.2100, train C loss: 0.1162, train D loss: 0.0031
    epoch 150: total loss:12.9538, train F loss: 12.9508, train C loss: 0.1033, train D loss: 0.0030
    epoch 160: total loss:12.7292, train F loss: 12.7259, train C loss: 0.0968, train D loss: 0.0033
    epoch 170: total loss:12.5311, train F loss: 12.5278, train C loss: 0.0962, train D loss: 0.0033
    epoch 180: total loss:12.3626, train F loss: 12.3591, train C loss: 0.0824, train D loss: 0.0036
    epoch 190: total loss:12.2245, train F loss: 12.2207, train C loss: 0.0805, train D loss: 0.0038
    spatial data classification: Avg Accuracy = 98.294359%
    Updating classifier...
    Training classifier...
    torch.Size([3635, 33])
    torch.Size([3566, 33])
    epoch   0: overall loss: 15.6703,sc classifier loss: 3.5437,representation loss: 1.2127,within spatial regularization loss: 0.0959
    epoch  10: overall loss: 3.9806,sc classifier loss: 2.2373,representation loss: 0.1743,within spatial regularization loss: 0.0616
    epoch  20: overall loss: 2.7211,sc classifier loss: 1.7246,representation loss: 0.0997,within spatial regularization loss: 0.0602
    epoch  30: overall loss: 2.2522,sc classifier loss: 1.3839,representation loss: 0.0868,within spatial regularization loss: 0.0750
    epoch  40: overall loss: 1.9173,sc classifier loss: 1.1276,representation loss: 0.0790,within spatial regularization loss: 0.0827
    epoch  50: overall loss: 1.6932,sc classifier loss: 0.9299,representation loss: 0.0763,within spatial regularization loss: 0.0861
    epoch  60: overall loss: 1.5083,sc classifier loss: 0.7830,representation loss: 0.0725,within spatial regularization loss: 0.0883
    epoch  70: overall loss: 1.4289,sc classifier loss: 0.6747,representation loss: 0.0754,within spatial regularization loss: 0.0884
    epoch  80: overall loss: 1.3784,sc classifier loss: 0.6001,representation loss: 0.0778,within spatial regularization loss: 0.0909
    epoch  90: overall loss: 1.2288,sc classifier loss: 0.5419,representation loss: 0.0687,within spatial regularization loss: 0.0888
    epoch 100: overall loss: 1.1528,sc classifier loss: 0.4884,representation loss: 0.0664,within spatial regularization loss: 0.0904
    epoch 110: overall loss: 1.1057,sc classifier loss: 0.4501,representation loss: 0.0656,within spatial regularization loss: 0.0917
    epoch 120: overall loss: 1.0087,sc classifier loss: 0.4157,representation loss: 0.0593,within spatial regularization loss: 0.0917
    epoch 130: overall loss: 0.9706,sc classifier loss: 0.3865,representation loss: 0.0584,within spatial regularization loss: 0.0934
    epoch 140: overall loss: 0.9384,sc classifier loss: 0.3609,representation loss: 0.0577,within spatial regularization loss: 0.0938
    epoch 150: overall loss: 0.9120,sc classifier loss: 0.3381,representation loss: 0.0574,within spatial regularization loss: 0.0950
    epoch 160: overall loss: 0.8995,sc classifier loss: 0.3188,representation loss: 0.0581,within spatial regularization loss: 0.0945
    epoch 170: overall loss: 0.9075,sc classifier loss: 0.3011,representation loss: 0.0606,within spatial regularization loss: 0.0950
    epoch 180: overall loss: 0.9742,sc classifier loss: 0.2911,representation loss: 0.0683,within spatial regularization loss: 0.0979
    epoch 190: overall loss: 0.8948,sc classifier loss: 0.2773,representation loss: 0.0617,within spatial regularization loss: 0.0923
    single cell data classification: Avg Accuracy = 92.880994%



```python
adata_concat_st = ad.concat(adata_l[0:len(section_ids)], label="slice_name", keys=section_ids)
```

```python
sc.tl.pca(adata_concat_st)
adata_concat_st.obsm['X_pca_old'] = adata_concat_st.obsm['X_pca'].copy()
adata_concat_st.obsm['X_pca'] = adata_concat_st.obsm['embedding'].copy()
sc.pp.neighbors(adata_concat_st)  
sc.tl.umap(adata_concat_st)
```


```python
sc.tl.leiden(adata_concat_st, random_state=666, key_added="leiden", resolution=0.18)
len(list(set(adata_concat_st.obs['leiden'].tolist())))
```
    7



## Results and visualizations
```python
from stSMILE import analysis
analysis.mclust_R(adata_concat_st, num_cluster=7, used_obsm="embedding")
```

    fitting ...
      |======================================================================| 100%





    AnnData object with n_obs × n_vars = 7201 × 3010
        obs: 'in_tissue', 'array_row', 'array_col', 'Ground Truth', 'n_genes', 'image_cluster', 'dbscan_cluster_new', 'ref', 'st_pre', 'slice_name', 'leiden', 'mclust'
        uns: 'pca', 'neighbors', 'umap', 'leiden'
        obsm: 'X_pca', 'features', 'features_summary_scale0.5_0.5', 'features_summary_scale0.5_1', 'features_summary_scale0.5_2', 'features_summary_scale1_0.5', 'features_summary_scale1_1', 'features_summary_scale1_2', 'features_summary_scale2_0.5', 'features_summary_scale2_1', 'features_summary_scale2_2', 'spatial', 'embedding', 'hidden_spatial', 'reconstruct', 'deconvolution', 'X_pca_old', 'X_umap'
        varm: 'PCs'
        obsp: 'distances', 'connectivities'




```python
plt.rcParams["figure.figsize"] = (4, 4)
sc.pl.umap(adata_concat_st,color=["mclust",'Ground Truth',"slice_name"], wspace=0.4, save = 'DLPFC_umap_cluster_semiSMILE.pdf')  
```

    WARNING: saving figure to file figures/umapDLPFC_umap_cluster_semiSMILE.pdf



    
![png](run_semi_SMILE_on_DLPFC_data_files/run_semi_SMILE_on_DLPFC_data_30_1.png)
    



```python
import matplotlib.pyplot as plt
from sklearn.metrics import adjusted_rand_score as ari_score
from sklearn.metrics import normalized_mutual_info_score as nmi_score
# split to each data
Batch_list = []
for section_id in section_ids:
    Batch_list.append(adata_concat_st[adata_concat_st.obs['slice_name'] == section_id])

spot_size = 200
title_size = 12
ARI_list = []
NMI_list = []
for bb in range(len(section_ids)):
    ARI_list.append(round(ari_score(Batch_list[bb].obs['Ground Truth'], Batch_list[bb].obs['mclust']), 2))
    NMI_list.append(round(nmi_score(Batch_list[bb].obs['Ground Truth'], Batch_list[bb].obs['mclust']), 2))

fig, ax = plt.subplots(2, 1, figsize=(3.5, 7), gridspec_kw={'wspace': 0.05, 'hspace': 0.1})
_sc_0 = sc.pl.spatial(Batch_list[0], img_key=None, color=['mclust'], title=[''],
                      legend_loc=None, legend_fontsize=12, show=False, ax=ax[0], frameon=False,
                      spot_size=spot_size)
_sc_0[0].set_title("ARI=" + str(ARI_list[0])+",NMI=" + str(NMI_list[0]), size=title_size)
_sc_1 = sc.pl.spatial(Batch_list[1], img_key=None, color=['mclust'], title=[''],
                      legend_loc=None, legend_fontsize=12, show=False, ax=ax[1], frameon=False,
                      spot_size=spot_size)
_sc_1[0].set_title("ARI=" + str(ARI_list[1])+",NMI=" + str(NMI_list[1]), size=title_size)
plt.savefig("DLPFC_spatial_semiSMILE.pdf") 
plt.show()
```


    
![png](run_semi_SMILE_on_DLPFC_data_files/run_semi_SMILE_on_DLPFC_data_31_0.png)
    



```python
adata_l[0].uns['deconvolution']
```


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>AAACAAGTATCTCCCA-1_151674_151674</th>
      <th>AAACAATCTACTAGCA-1_151674_151674</th>
      <th>AAACACCAATAACTGC-1_151674_151674</th>
      <th>AAACAGAGCGACTCCT-1_151674_151674</th>
      <th>AAACAGCTTTCAGAAG-1_151674_151674</th>
      <th>AAACAGGGTCTATATT-1_151674_151674</th>
      <th>AAACAGTGTTCCTGGG-1_151674_151674</th>
      <th>AAACATTTCCCGGATT-1_151674_151674</th>
      <th>AAACCCGAACGAAATC-1_151674_151674</th>
      <th>AAACCGGGTAGGTACC-1_151674_151674</th>
      <th>...</th>
      <th>TTGTGTATGCCACCAA-1_151674_151674</th>
      <th>TTGTGTTTCCCGAAAG-1_151674_151674</th>
      <th>TTGTTAGCAAATTCGA-1_151674_151674</th>
      <th>TTGTTCAGTGTGCTAC-1_151674_151674</th>
      <th>TTGTTGTGTGTCAAGA-1_151674_151674</th>
      <th>TTGTTTCACATCCAGG-1_151674_151674</th>
      <th>TTGTTTCATTAGTCTA-1_151674_151674</th>
      <th>TTGTTTCCATACAACT-1_151674_151674</th>
      <th>TTGTTTGTATTACACG-1_151674_151674</th>
      <th>TTGTTTGTGTAAATTC-1_151674_151674</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Ex_5_L5</th>
      <td>5.331036e-03</td>
      <td>7.324228e-06</td>
      <td>1.937497e-09</td>
      <td>5.294757e-03</td>
      <td>1.807263e-02</td>
      <td>1.880007e-04</td>
      <td>1.581045e-06</td>
      <td>2.861076e-03</td>
      <td>1.820327e-04</td>
      <td>1.195052e-03</td>
      <td>...</td>
      <td>6.635312e-05</td>
      <td>4.834505e-05</td>
      <td>4.722993e-04</td>
      <td>1.908829e-02</td>
      <td>5.307228e-02</td>
      <td>3.280701e-08</td>
      <td>1.399956e-08</td>
      <td>1.044812e-04</td>
      <td>7.543200e-07</td>
      <td>8.032936e-07</td>
    </tr>
    <tr>
      <th>Ex_3_L4_5</th>
      <td>7.539445e-02</td>
      <td>6.819289e-05</td>
      <td>1.030208e-09</td>
      <td>1.737979e-01</td>
      <td>1.791637e-01</td>
      <td>2.081110e-01</td>
      <td>1.536111e-05</td>
      <td>4.785919e-02</td>
      <td>3.770103e-03</td>
      <td>6.018199e-01</td>
      <td>...</td>
      <td>8.773857e-03</td>
      <td>2.299776e-03</td>
      <td>1.031976e-01</td>
      <td>3.292956e-01</td>
      <td>4.178228e-01</td>
      <td>1.540545e-08</td>
      <td>8.910790e-09</td>
      <td>8.421935e-03</td>
      <td>2.524276e-05</td>
      <td>4.358436e-07</td>
    </tr>
    <tr>
      <th>Ex_6_L4_6</th>
      <td>1.575898e-06</td>
      <td>2.745482e-05</td>
      <td>6.694957e-09</td>
      <td>1.030967e-06</td>
      <td>1.205636e-03</td>
      <td>2.139607e-04</td>
      <td>1.182888e-03</td>
      <td>9.814599e-06</td>
      <td>3.724547e-08</td>
      <td>1.472958e-04</td>
      <td>...</td>
      <td>2.117671e-03</td>
      <td>1.107603e-02</td>
      <td>3.724185e-03</td>
      <td>4.251718e-03</td>
      <td>2.000385e-03</td>
      <td>6.538797e-08</td>
      <td>7.465867e-08</td>
      <td>1.571381e-02</td>
      <td>1.732987e-03</td>
      <td>1.025283e-06</td>
    </tr>
    <tr>
      <th>Inhib_5</th>
      <td>3.887602e-05</td>
      <td>3.755447e-03</td>
      <td>1.494762e-09</td>
      <td>4.206381e-06</td>
      <td>3.983635e-06</td>
      <td>3.805378e-08</td>
      <td>3.293550e-08</td>
      <td>1.616721e-03</td>
      <td>1.622213e-08</td>
      <td>1.483451e-08</td>
      <td>...</td>
      <td>1.865695e-07</td>
      <td>5.408108e-07</td>
      <td>6.714814e-04</td>
      <td>5.229215e-08</td>
      <td>7.182640e-08</td>
      <td>2.559299e-08</td>
      <td>1.129170e-08</td>
      <td>1.972515e-06</td>
      <td>4.577277e-09</td>
      <td>6.487975e-04</td>
    </tr>
    <tr>
      <th>Inhib_4_SST</th>
      <td>6.003727e-06</td>
      <td>1.779631e-03</td>
      <td>2.024276e-10</td>
      <td>2.864599e-06</td>
      <td>1.316401e-04</td>
      <td>4.011348e-07</td>
      <td>2.751985e-08</td>
      <td>2.927481e-05</td>
      <td>1.876359e-07</td>
      <td>4.262075e-07</td>
      <td>...</td>
      <td>4.628414e-07</td>
      <td>9.928970e-07</td>
      <td>8.035329e-06</td>
      <td>2.911775e-05</td>
      <td>4.350154e-05</td>
      <td>7.866907e-10</td>
      <td>8.385642e-10</td>
      <td>2.388119e-06</td>
      <td>5.029619e-09</td>
      <td>4.906387e-05</td>
    </tr>
    <tr>
      <th>OPCs_2</th>
      <td>9.411585e-12</td>
      <td>2.803001e-08</td>
      <td>2.958015e-05</td>
      <td>1.459952e-12</td>
      <td>3.119064e-07</td>
      <td>4.982614e-07</td>
      <td>6.164388e-05</td>
      <td>1.319579e-10</td>
      <td>2.043856e-12</td>
      <td>3.260431e-09</td>
      <td>...</td>
      <td>3.264469e-05</td>
      <td>2.948959e-04</td>
      <td>1.025248e-10</td>
      <td>2.536996e-09</td>
      <td>7.815594e-09</td>
      <td>1.441980e-03</td>
      <td>2.117293e-04</td>
      <td>4.960734e-04</td>
      <td>8.936862e-06</td>
      <td>6.285780e-07</td>
    </tr>
    <tr>
      <th>Oligos_1</th>
      <td>1.103193e-09</td>
      <td>1.334777e-08</td>
      <td>6.613036e-01</td>
      <td>1.950127e-10</td>
      <td>1.607728e-05</td>
      <td>5.314733e-06</td>
      <td>1.269162e-01</td>
      <td>3.042426e-09</td>
      <td>1.468542e-11</td>
      <td>3.137766e-07</td>
      <td>...</td>
      <td>1.934497e-04</td>
      <td>1.861525e-03</td>
      <td>3.766803e-11</td>
      <td>2.829274e-09</td>
      <td>1.265834e-08</td>
      <td>3.549326e-01</td>
      <td>5.432719e-01</td>
      <td>9.517118e-04</td>
      <td>4.933485e-02</td>
      <td>2.862671e-10</td>
    </tr>
    <tr>
      <th>Inhib_2_VIP</th>
      <td>1.142907e-04</td>
      <td>2.737476e-03</td>
      <td>6.132087e-08</td>
      <td>4.443296e-05</td>
      <td>2.542165e-06</td>
      <td>7.336904e-08</td>
      <td>1.811521e-08</td>
      <td>8.416994e-04</td>
      <td>1.239800e-06</td>
      <td>1.294661e-08</td>
      <td>...</td>
      <td>4.978689e-07</td>
      <td>1.027854e-06</td>
      <td>6.562318e-04</td>
      <td>2.064023e-06</td>
      <td>2.481426e-06</td>
      <td>7.098891e-07</td>
      <td>1.599951e-07</td>
      <td>3.750144e-06</td>
      <td>1.140748e-09</td>
      <td>1.657319e-02</td>
    </tr>
    <tr>
      <th>OPCs_1</th>
      <td>1.060887e-06</td>
      <td>5.427985e-06</td>
      <td>4.150887e-06</td>
      <td>2.897693e-07</td>
      <td>1.265667e-05</td>
      <td>1.501308e-05</td>
      <td>2.545834e-05</td>
      <td>1.609986e-06</td>
      <td>1.650403e-06</td>
      <td>5.230331e-07</td>
      <td>...</td>
      <td>3.125764e-04</td>
      <td>7.122649e-04</td>
      <td>9.678852e-08</td>
      <td>1.952605e-06</td>
      <td>7.306139e-06</td>
      <td>1.551582e-04</td>
      <td>1.766666e-05</td>
      <td>1.211827e-03</td>
      <td>7.375433e-06</td>
      <td>1.482470e-03</td>
    </tr>
    <tr>
      <th>Ex_4_L_6</th>
      <td>9.386843e-07</td>
      <td>8.146332e-07</td>
      <td>1.562857e-08</td>
      <td>1.397563e-06</td>
      <td>4.251486e-01</td>
      <td>3.273296e-01</td>
      <td>2.726563e-05</td>
      <td>4.943303e-07</td>
      <td>2.445179e-06</td>
      <td>3.013038e-01</td>
      <td>...</td>
      <td>4.658365e-02</td>
      <td>1.329014e-02</td>
      <td>1.465842e-07</td>
      <td>1.349780e-02</td>
      <td>5.322984e-02</td>
      <td>1.111096e-07</td>
      <td>7.676510e-08</td>
      <td>3.381010e-02</td>
      <td>3.375684e-05</td>
      <td>7.671878e-08</td>
    </tr>
    <tr>
      <th>Inhib_3_SST</th>
      <td>4.578171e-06</td>
      <td>3.401444e-02</td>
      <td>2.195845e-08</td>
      <td>2.991445e-06</td>
      <td>1.034968e-05</td>
      <td>5.285029e-06</td>
      <td>7.356957e-06</td>
      <td>2.125660e-05</td>
      <td>1.916202e-06</td>
      <td>1.477993e-06</td>
      <td>...</td>
      <td>3.658260e-05</td>
      <td>1.695212e-04</td>
      <td>1.678331e-05</td>
      <td>3.886961e-06</td>
      <td>5.464119e-06</td>
      <td>8.078320e-08</td>
      <td>9.694001e-08</td>
      <td>1.783061e-04</td>
      <td>3.585289e-06</td>
      <td>7.286732e-02</td>
    </tr>
    <tr>
      <th>Micro/Macro</th>
      <td>2.150780e-05</td>
      <td>3.946796e-05</td>
      <td>9.279989e-05</td>
      <td>3.148427e-06</td>
      <td>8.179383e-05</td>
      <td>3.732434e-06</td>
      <td>9.665425e-05</td>
      <td>1.013447e-04</td>
      <td>1.821458e-09</td>
      <td>4.200255e-06</td>
      <td>...</td>
      <td>2.788698e-05</td>
      <td>3.078261e-05</td>
      <td>8.546996e-07</td>
      <td>1.906584e-09</td>
      <td>6.599026e-09</td>
      <td>2.650844e-05</td>
      <td>7.879714e-05</td>
      <td>7.170058e-05</td>
      <td>1.959703e-05</td>
      <td>4.396588e-09</td>
    </tr>
    <tr>
      <th>Mix_1</th>
      <td>2.023518e-07</td>
      <td>4.801752e-08</td>
      <td>1.651792e-07</td>
      <td>4.279693e-08</td>
      <td>6.500719e-02</td>
      <td>9.163583e-05</td>
      <td>3.579773e-05</td>
      <td>8.461726e-07</td>
      <td>6.530758e-10</td>
      <td>1.560206e-04</td>
      <td>...</td>
      <td>4.352113e-05</td>
      <td>9.424517e-05</td>
      <td>2.463661e-08</td>
      <td>1.416680e-06</td>
      <td>1.149178e-05</td>
      <td>1.209067e-06</td>
      <td>7.319786e-07</td>
      <td>1.364390e-04</td>
      <td>2.047870e-05</td>
      <td>4.668186e-10</td>
    </tr>
    <tr>
      <th>Oligos_2</th>
      <td>1.319824e-08</td>
      <td>1.834519e-07</td>
      <td>5.481540e-02</td>
      <td>2.027216e-09</td>
      <td>1.892705e-05</td>
      <td>2.267293e-06</td>
      <td>7.492923e-04</td>
      <td>5.514750e-08</td>
      <td>1.110849e-10</td>
      <td>1.142498e-07</td>
      <td>...</td>
      <td>4.347417e-05</td>
      <td>2.343807e-04</td>
      <td>1.066354e-09</td>
      <td>1.877399e-08</td>
      <td>8.906188e-08</td>
      <td>1.582011e-01</td>
      <td>7.045989e-02</td>
      <td>2.358791e-04</td>
      <td>1.008167e-04</td>
      <td>6.806261e-08</td>
    </tr>
    <tr>
      <th>Inhib_1</th>
      <td>9.456518e-06</td>
      <td>5.382027e-06</td>
      <td>6.668554e-08</td>
      <td>4.203802e-06</td>
      <td>2.071941e-05</td>
      <td>2.205932e-08</td>
      <td>1.627803e-08</td>
      <td>1.251685e-05</td>
      <td>8.950122e-09</td>
      <td>5.379627e-08</td>
      <td>...</td>
      <td>7.012682e-08</td>
      <td>1.763423e-07</td>
      <td>2.499046e-06</td>
      <td>5.011632e-04</td>
      <td>7.377816e-04</td>
      <td>5.148976e-08</td>
      <td>5.739727e-08</td>
      <td>3.959746e-07</td>
      <td>9.493011e-10</td>
      <td>3.387155e-07</td>
    </tr>
    <tr>
      <th>Mix_5</th>
      <td>3.775398e-02</td>
      <td>2.983437e-04</td>
      <td>7.938143e-06</td>
      <td>2.217800e-02</td>
      <td>1.719568e-03</td>
      <td>9.524930e-05</td>
      <td>4.407734e-05</td>
      <td>3.544265e-02</td>
      <td>2.640762e-04</td>
      <td>7.582183e-05</td>
      <td>...</td>
      <td>2.161817e-04</td>
      <td>4.671341e-04</td>
      <td>2.099425e-03</td>
      <td>1.682449e-02</td>
      <td>3.010873e-02</td>
      <td>5.564242e-05</td>
      <td>2.004363e-05</td>
      <td>1.096651e-03</td>
      <td>1.725855e-05</td>
      <td>5.727104e-04</td>
    </tr>
    <tr>
      <th>Oligos_3</th>
      <td>1.339343e-06</td>
      <td>2.749919e-06</td>
      <td>6.864335e-03</td>
      <td>3.779722e-07</td>
      <td>1.681903e-04</td>
      <td>3.020849e-05</td>
      <td>8.184273e-03</td>
      <td>1.727511e-06</td>
      <td>3.279376e-07</td>
      <td>2.599987e-06</td>
      <td>...</td>
      <td>5.616409e-04</td>
      <td>1.803871e-03</td>
      <td>2.128524e-07</td>
      <td>1.337305e-05</td>
      <td>3.229907e-05</td>
      <td>2.281751e-01</td>
      <td>2.331905e-02</td>
      <td>2.335216e-03</td>
      <td>2.593313e-03</td>
      <td>5.908923e-06</td>
    </tr>
    <tr>
      <th>Ex_7_L4_6</th>
      <td>1.265088e-09</td>
      <td>1.448083e-08</td>
      <td>1.310187e-08</td>
      <td>1.278454e-09</td>
      <td>2.435715e-03</td>
      <td>2.486624e-04</td>
      <td>4.005212e-04</td>
      <td>1.146575e-09</td>
      <td>2.247563e-09</td>
      <td>2.020953e-05</td>
      <td>...</td>
      <td>7.581071e-04</td>
      <td>7.062975e-03</td>
      <td>1.216199e-09</td>
      <td>1.270670e-02</td>
      <td>5.825293e-02</td>
      <td>5.130069e-07</td>
      <td>2.386969e-07</td>
      <td>7.764046e-03</td>
      <td>3.676155e-04</td>
      <td>2.093845e-08</td>
    </tr>
    <tr>
      <th>Ex_1_L5_6</th>
      <td>6.708034e-14</td>
      <td>7.432730e-12</td>
      <td>1.169388e-04</td>
      <td>1.438007e-14</td>
      <td>4.978217e-02</td>
      <td>4.008515e-05</td>
      <td>8.784932e-05</td>
      <td>2.891945e-13</td>
      <td>6.747332e-16</td>
      <td>1.772819e-06</td>
      <td>...</td>
      <td>1.195745e-04</td>
      <td>8.653622e-04</td>
      <td>1.086763e-12</td>
      <td>2.970352e-03</td>
      <td>2.290949e-02</td>
      <td>8.726579e-03</td>
      <td>6.195587e-04</td>
      <td>8.044475e-03</td>
      <td>1.219586e-05</td>
      <td>2.063101e-13</td>
    </tr>
    <tr>
      <th>Mix_4</th>
      <td>2.181848e-05</td>
      <td>1.083354e-06</td>
      <td>1.585618e-06</td>
      <td>7.154743e-06</td>
      <td>1.327700e-03</td>
      <td>1.434868e-05</td>
      <td>8.078919e-06</td>
      <td>1.508239e-05</td>
      <td>6.845541e-08</td>
      <td>5.916182e-05</td>
      <td>...</td>
      <td>4.211684e-05</td>
      <td>8.542206e-05</td>
      <td>1.025069e-07</td>
      <td>1.186687e-05</td>
      <td>8.130258e-05</td>
      <td>7.573494e-06</td>
      <td>3.135555e-06</td>
      <td>3.640089e-04</td>
      <td>1.928896e-06</td>
      <td>9.510257e-08</td>
    </tr>
    <tr>
      <th>Inhib_6_SST</th>
      <td>1.554437e-03</td>
      <td>2.165699e-07</td>
      <td>5.253943e-08</td>
      <td>1.110127e-03</td>
      <td>1.013113e-05</td>
      <td>9.396200e-09</td>
      <td>1.047670e-08</td>
      <td>6.310049e-04</td>
      <td>2.091033e-05</td>
      <td>8.094664e-09</td>
      <td>...</td>
      <td>2.852434e-08</td>
      <td>1.117575e-07</td>
      <td>1.179586e-04</td>
      <td>1.979603e-02</td>
      <td>2.063605e-02</td>
      <td>1.167061e-06</td>
      <td>1.440366e-07</td>
      <td>2.645498e-07</td>
      <td>5.642926e-10</td>
      <td>2.383495e-05</td>
    </tr>
    <tr>
      <th>Ex_2_L5</th>
      <td>1.250792e-01</td>
      <td>1.487714e-01</td>
      <td>2.009409e-09</td>
      <td>1.187723e-01</td>
      <td>2.554095e-02</td>
      <td>7.842440e-05</td>
      <td>2.048043e-06</td>
      <td>1.362125e-01</td>
      <td>1.671452e-01</td>
      <td>2.363377e-04</td>
      <td>...</td>
      <td>1.753663e-05</td>
      <td>1.818949e-05</td>
      <td>1.038991e-03</td>
      <td>1.365497e-04</td>
      <td>6.925314e-04</td>
      <td>2.095807e-08</td>
      <td>1.111339e-08</td>
      <td>1.985297e-05</td>
      <td>2.102526e-06</td>
      <td>7.547128e-02</td>
    </tr>
    <tr>
      <th>Inhib_8_PVALB</th>
      <td>3.137120e-04</td>
      <td>5.000665e-01</td>
      <td>1.009287e-08</td>
      <td>1.461515e-04</td>
      <td>6.011871e-04</td>
      <td>9.916092e-06</td>
      <td>2.214579e-05</td>
      <td>3.225113e-03</td>
      <td>3.473086e-05</td>
      <td>4.203666e-06</td>
      <td>...</td>
      <td>1.729646e-05</td>
      <td>4.894003e-05</td>
      <td>9.109486e-04</td>
      <td>6.600137e-06</td>
      <td>1.380142e-05</td>
      <td>6.112612e-08</td>
      <td>6.199973e-08</td>
      <td>4.663732e-05</td>
      <td>1.755405e-05</td>
      <td>1.606022e-01</td>
    </tr>
    <tr>
      <th>Ex_10_L2_4</th>
      <td>7.426242e-01</td>
      <td>2.854269e-01</td>
      <td>2.601273e-07</td>
      <td>6.744429e-01</td>
      <td>3.196594e-02</td>
      <td>1.722727e-01</td>
      <td>1.039244e-03</td>
      <td>6.791140e-01</td>
      <td>8.284303e-01</td>
      <td>2.536228e-02</td>
      <td>...</td>
      <td>4.609047e-01</td>
      <td>1.767566e-01</td>
      <td>5.989573e-02</td>
      <td>4.285125e-04</td>
      <td>1.017905e-03</td>
      <td>2.726523e-06</td>
      <td>2.370724e-06</td>
      <td>1.177525e-01</td>
      <td>2.840737e-03</td>
      <td>6.517859e-01</td>
    </tr>
    <tr>
      <th>Astros_2</th>
      <td>4.407651e-05</td>
      <td>8.192485e-05</td>
      <td>2.790105e-04</td>
      <td>1.191013e-05</td>
      <td>2.897381e-05</td>
      <td>8.134085e-07</td>
      <td>5.111986e-06</td>
      <td>6.196243e-05</td>
      <td>2.585014e-06</td>
      <td>2.009086e-07</td>
      <td>...</td>
      <td>8.109741e-06</td>
      <td>1.246517e-05</td>
      <td>2.156572e-07</td>
      <td>5.336699e-07</td>
      <td>2.824991e-06</td>
      <td>1.574462e-02</td>
      <td>6.534528e-04</td>
      <td>4.654202e-05</td>
      <td>3.549511e-07</td>
      <td>2.068751e-03</td>
    </tr>
    <tr>
      <th>Endo</th>
      <td>1.846407e-07</td>
      <td>1.172838e-04</td>
      <td>7.158233e-02</td>
      <td>4.867097e-08</td>
      <td>3.630085e-03</td>
      <td>2.035400e-02</td>
      <td>5.225666e-01</td>
      <td>7.006853e-07</td>
      <td>3.583206e-09</td>
      <td>1.423539e-03</td>
      <td>...</td>
      <td>1.277818e-01</td>
      <td>2.940560e-01</td>
      <td>6.758000e-09</td>
      <td>7.188731e-08</td>
      <td>3.865507e-07</td>
      <td>1.336352e-02</td>
      <td>1.039849e-01</td>
      <td>2.371631e-01</td>
      <td>4.666061e-01</td>
      <td>2.154636e-07</td>
    </tr>
    <tr>
      <th>Astros_1</th>
      <td>1.174598e-10</td>
      <td>5.284930e-08</td>
      <td>2.032328e-01</td>
      <td>1.903077e-11</td>
      <td>1.755139e-06</td>
      <td>1.546321e-06</td>
      <td>1.910865e-03</td>
      <td>2.740917e-10</td>
      <td>6.233921e-12</td>
      <td>5.801396e-08</td>
      <td>...</td>
      <td>4.928529e-05</td>
      <td>2.360478e-04</td>
      <td>1.280484e-13</td>
      <td>2.185848e-12</td>
      <td>2.892360e-11</td>
      <td>1.807888e-01</td>
      <td>2.496789e-01</td>
      <td>1.675681e-04</td>
      <td>1.973617e-04</td>
      <td>9.244927e-09</td>
    </tr>
    <tr>
      <th>Inhib_7_PVALB</th>
      <td>3.107781e-03</td>
      <td>1.277954e-04</td>
      <td>3.368560e-10</td>
      <td>9.313971e-04</td>
      <td>2.075744e-02</td>
      <td>1.928053e-05</td>
      <td>1.606558e-07</td>
      <td>4.054386e-02</td>
      <td>3.437884e-07</td>
      <td>3.727704e-05</td>
      <td>...</td>
      <td>4.754766e-06</td>
      <td>5.486481e-06</td>
      <td>6.700169e-01</td>
      <td>4.568524e-01</td>
      <td>2.505307e-01</td>
      <td>3.965241e-09</td>
      <td>2.020975e-09</td>
      <td>2.161855e-05</td>
      <td>5.491373e-08</td>
      <td>6.068345e-07</td>
    </tr>
    <tr>
      <th>Mix_2</th>
      <td>1.004144e-05</td>
      <td>1.653449e-06</td>
      <td>6.025379e-07</td>
      <td>1.488964e-06</td>
      <td>1.154766e-01</td>
      <td>1.376075e-04</td>
      <td>1.566847e-05</td>
      <td>4.695126e-05</td>
      <td>3.664334e-09</td>
      <td>3.158174e-04</td>
      <td>...</td>
      <td>8.254055e-05</td>
      <td>1.792856e-04</td>
      <td>1.027101e-05</td>
      <td>9.353810e-05</td>
      <td>4.768217e-04</td>
      <td>3.585676e-06</td>
      <td>1.649501e-06</td>
      <td>4.078890e-04</td>
      <td>6.169567e-06</td>
      <td>1.721464e-08</td>
    </tr>
    <tr>
      <th>Mix_3</th>
      <td>8.550617e-03</td>
      <td>2.264480e-02</td>
      <td>1.084012e-08</td>
      <td>3.214811e-03</td>
      <td>6.843353e-03</td>
      <td>1.213823e-05</td>
      <td>3.341140e-06</td>
      <td>5.134280e-02</td>
      <td>7.748995e-05</td>
      <td>1.632890e-05</td>
      <td>...</td>
      <td>1.170016e-05</td>
      <td>2.899700e-05</td>
      <td>1.571581e-01</td>
      <td>1.226534e-01</td>
      <td>8.671648e-02</td>
      <td>1.105583e-07</td>
      <td>4.949691e-08</td>
      <td>3.657339e-05</td>
      <td>1.942507e-06</td>
      <td>1.783023e-02</td>
    </tr>
    <tr>
      <th>Ex_8_L5_6</th>
      <td>2.243421e-09</td>
      <td>6.304414e-08</td>
      <td>4.355186e-04</td>
      <td>1.454770e-09</td>
      <td>4.846143e-02</td>
      <td>2.702842e-01</td>
      <td>3.363685e-01</td>
      <td>3.047409e-09</td>
      <td>9.672512e-10</td>
      <td>6.697954e-02</td>
      <td>...</td>
      <td>3.508029e-01</td>
      <td>4.874664e-01</td>
      <td>7.399700e-11</td>
      <td>1.185523e-06</td>
      <td>1.446570e-05</td>
      <td>5.523010e-03</td>
      <td>2.653915e-03</td>
      <td>5.624420e-01</td>
      <td>4.760079e-01</td>
      <td>3.900745e-09</td>
    </tr>
    <tr>
      <th>Ex_9_L5_6</th>
      <td>1.461552e-05</td>
      <td>1.779963e-05</td>
      <td>9.042719e-10</td>
      <td>2.614222e-05</td>
      <td>2.345920e-03</td>
      <td>4.252439e-04</td>
      <td>1.545127e-06</td>
      <td>5.732153e-06</td>
      <td>6.430119e-05</td>
      <td>8.355253e-04</td>
      <td>...</td>
      <td>2.017272e-04</td>
      <td>1.503065e-04</td>
      <td>8.468319e-07</td>
      <td>8.313114e-04</td>
      <td>1.579871e-03</td>
      <td>1.049121e-08</td>
      <td>6.970614e-09</td>
      <td>2.146831e-04</td>
      <td>9.236263e-07</td>
      <td>1.210168e-05</td>
    </tr>
    <tr>
      <th>Astros_3</th>
      <td>9.727123e-11</td>
      <td>1.935676e-07</td>
      <td>1.232625e-03</td>
      <td>2.886797e-11</td>
      <td>6.258192e-06</td>
      <td>8.700958e-06</td>
      <td>2.204419e-04</td>
      <td>1.306050e-10</td>
      <td>2.727591e-10</td>
      <td>1.324405e-07</td>
      <td>...</td>
      <td>1.870019e-04</td>
      <td>6.419263e-04</td>
      <td>1.202216e-13</td>
      <td>1.377305e-10</td>
      <td>1.800484e-09</td>
      <td>3.284720e-02</td>
      <td>5.021272e-03</td>
      <td>7.355794e-04</td>
      <td>3.811958e-05</td>
      <td>1.928544e-06</td>
    </tr>
  </tbody>
</table>
<p>33 rows × 3635 columns</p>
</div>




```python
# write out the deconvolution result
for i in range(len(adata_l)-1):
    adata_i = adata_l[i].copy()
    dev_i = adata_i.uns['deconvolution']
    dev_i.to_csv('DLPFC_semiSMILE_dev_'+ section_ids[i]+'.csv', sep='\t')
```


```python

```


```python

```
