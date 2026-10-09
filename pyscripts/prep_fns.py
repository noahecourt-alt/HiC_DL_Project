import numpy as np
import torch
from sklearn.decomposition import PCA


def build_input_edges(hic_map): #follows the edge features described by Schultz et al
    hm = torch.tensor(hic_map, dtype=torch.float32)

    log_hm = torch.log1p(hm)
    log_hm = log_hm.unsqueeze(0) #adds the channel dimension. 1 channel.

    mean_list = []
    for i in range(hic_map.shape[0]):
        mean_list.append(np.mean(np.diag(hic_map, k=i)))
    
    mean_diag = np.zeros_like(hic_map, dtype=np.float32)
    for i in range(mean_diag.shape[0]):
        for j in range(mean_diag.shape[1]):
            mean_diag[i][j] = mean_list[np.abs(i-j)]
    md = mean_diag.copy()
    mean_diag = torch.tensor(mean_diag, dtype=torch.float32).unsqueeze(0)

    hm = hm.unsqueeze(0) #raw contact map

    input_tensor = torch.cat((hm, log_hm, mean_diag), dim=0) #concatinate the tensors in the channel dimension. 

    return input_tensor, md

def build_input_nodes(hic_map, mean_diag: np.ndarray, normalize: bool, eig: int): #follows the node features described by Schultz et al
    #genomic distance normalised map
    assert hic_map.shape == mean_diag.shape, "hic map and the gen. dist. norm. matrix need to be of the same size"
    norm_map = np.divide(hic_map, mean_diag, out=np.zeros_like(hic_map, dtype=np.float32), where=mean_diag!=0)
    norm_map = np.nan_to_num(norm_map, nan=0.0, posinf=0.0, neginf=0.0)

    pca = PCA(n_components=eig)
    pca.fit(norm_map)
    m = len(norm_map)
    topk_pcs = np.zeros((m, eig))
    for j in range(eig):
        pc = pca.components_[j]

        if normalize:
            min = np.min(pc)
            max = np.max(pc)
            if max > abs(min):
                val = max
            else:
                val = abs(min)

            # multiply by scale such that val x scale = 1
            scale = 1/val
            pc *= scale

        topk_pcs[:,j] = pc

    norm_map = torch.tensor(norm_map, dtype=torch.float32).unsqueeze(0)
    for x in range(eig):
        eig_chan = np.zeros_like(hic_map, dtype=np.float32)
        for i in range(m):
            for j in range(m):
                eig_chan[i][j] = topk_pcs[i][x] * topk_pcs[j][x]
        eig_chan = torch.tensor(eig_chan, dtype=torch.float32).unsqueeze(0)
        norm_map = torch.cat((norm_map, eig_chan), dim=0)

    return norm_map

def mean_fill(hic_map):
    for i in range(hic_map.shape[0]):
        diagonal = np.diag(hic_map, k=i)
        valid = diagonal[np.isfinite(diagonal) & (diagonal != 0)]

        if valid.size > 0:
            scale = valid.mean()
            hic = np.divide(hic_map, scale, dtype=np.float32)
            np.fill_diagonal(hic, 1.0)
            return hic


    raise ValueError("Hi-C map has no finite, non-zero values for normalization")

#wrapper for the channel feature functions.

def input_wrapper(hic_map, eig: int, normalize=False):
    hic = mean_fill(hic_map)
    edge_features, mean_diag = build_input_edges(hic)
    return torch.cat((edge_features, build_input_nodes(hic, mean_diag, normalize, eig)), dim=0)
