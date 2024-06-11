import datasets
import anndata as ad
import plotting
import os
import torch
import denoising
import layer_operations
import filtering
import gene_features

data = datasets.get_dataset("villacampa_lung_organoid", visualize=False)
param_dict = data.param_dict
dataset_path = "/home/dvegaa/spared/spared/processed_data/villacampa_data/villacampa_lung_organoid/2024-05-30-13-08-21"
adata = ad.read_h5ad(os.path.join(dataset_path, f'adata_raw.h5ad'))

#Get global exp fraction
adata_glob_exp = gene_features.get_glob_exp_frac(adata)
#X to array
adata.layers['counts'] = adata.X.toarray()
#TPM normalization
adata = layer_operations.tpm_normalization(param_dict["organism"], adata, from_layer='counts', to_layer='tpm')
#Transform the data with log1p (base 2.0)
adata = layer_operations.log1p_transformation(adata, from_layer='tpm', to_layer='log1p')

#def median_cleaner(collection: ad.AnnData, from_layer: str, to_layer: str, n_hops: int, hex_geometry: bool) -> ad.AnnData:
adata = denoising.median_cleaner(adata, from_layer='log1p', to_layer='d_log1p', n_hops=4, hex_geometry=param_dict["hex_geometry"])
#DONE

#Compute average moran for each gene in the layer d_log1p 
adata = gene_features.compute_moran(adata, hex_geometry=param_dict["hex_geometry"], from_layer='d_log1p')
#Filter genes by Moran's I
adata = filtering.filter_by_moran(adata, n_keep=param_dict['top_moran_genes'], from_layer='d_log1p')
#Apply combat
adata = layer_operations.combat_transformation(adata, batch_key=param_dict['combat_key'], from_layer='d_log1p', to_layer='c_d_log1p')
#Add a binary mask layer 
adata.layers['mask'] = adata.layers['tpm'] != 0

#def spackle_cleaner(adata: ad.AnnData, dataset: str, from_layer: str, to_layer: str, device) -> ad.AnnData:
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
adata = denoising.spackle_cleaner(adata=adata, dataset=data.dataset, from_layer="c_d_log1p", to_layer="c_t_log1p", device=device)
#DONE
breakpoint()