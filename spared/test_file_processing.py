import datasets
import processing
import os
import anndata as ad

breakpoint()
data = datasets.get_dataset("villacampa_lung_organoid", visualize=False)
dataset_path = "/home/dvegaa/spared/spared/processed_data/villacampa_data/villacampa_lung_organoid/2024-05-08-19-03-13"
collection_raw = ad.read_h5ad(os.path.join(dataset_path, f'adata_raw.h5ad'))

#def get_slide_from_collection(collection: ad.AnnData,  slide: str) -> ad.AnnData:
slide_id = collection_raw.obs["slide_id"].unique().tolist()[0]
adata_slide = processing.get_slide_from_collection(collection=collection_raw.copy(), slide=slide_id)
#DONE

#def get_exp_frac(adata: ad.AnnData) -> ad.AnnData:
collection_raw = processing.get_exp_frac(adata=collection_raw)
#DONE

#def get_glob_exp_frac(adata: ad.AnnData) -> ad.AnnData:
collection_raw = processing.get_glob_exp_frac(adata=collection_raw)
#DONE

#def filter_dataset(adata: ad.AnnData, param_dict: dict) -> ad.AnnData:
collection_raw = processing.filter_dataset(adata=collection_raw, param_dict=data.param_dict)
#DONE

#FIXME: when called on processed_data, the layer counts in created first and then the function is called
#This is because in raw adata there are no layers. So we imply that the user will add a layer before calling this function or what should we do?
#FIXME: from_layer values must be raw values (make sure this requirement is checked)
#def tpm_normalization(organism: str, adata: ad.AnnData, from_layer: str, to_layer: str) -> ad.AnnData:
collection_raw.layers['counts'] = collection_raw.X.toarray()
collection_raw = processing.tpm_normalization(organism=data.param_dict["organism"], adata=collection_raw, from_layer="counts", to_layer="tpm")
#DONE

#FIXME: from_layer to_layer parameters considerations
#def log1p_transformation(adata: ad.AnnData, from_layer: str, to_layer: str) -> ad.AnnData:
collection_raw = processing.log1p_transformation(adata=collection_raw, from_layer="tpm", to_layer="log1p")
#DONE

#def get_spatial_neighbors(adata: ad.AnnData, n_hops: int, hex_geometry: bool) -> dict:
spatial_nn_dict = processing.get_spatial_neighbors(adata=collection_raw, n_hops=6, hex_geometry=True)
#DONE

#FIXME: from_layer to_layer parameters considerations
#Leave the parameters but the function must verify that the form layer parameters check certain requirements
#def clean_noise(collection: ad.AnnData, from_layer: str, to_layer: str, n_hops: int, hex_geometry: bool) -> ad.AnnData:
collection_raw = processing.clean_noise(collection=collection_raw, from_layer='log1p', to_layer='d_log1p', n_hops=4, hex_geometry=True)
#DONE

#Leave the parameters but the function must verify that the form layer parameters check certain requirements
#FIXME: combat_transformation: layer it normalized and log (values follow a normal distribution)
#def combat_transformation(adata: ad.AnnData, batch_key: str, from_layer: str, to_layer: str) -> ad.AnnData:
collection_raw = processing.combat_transformation(adata=collection_raw, batch_key=data.param_dict['combat_key'], from_layer='d_log1p', to_layer='c_d_log1p')
collection_raw = processing.combat_transformation(adata=collection_raw, batch_key=data.param_dict['combat_key'], from_layer='log1p', to_layer='c_log1p')
#DONE

#def get_deltas(adata: ad.AnnData, from_layer: str, to_layer: str) -> ad.AnnData:
collection_raw = processing.get_deltas(adata=collection_raw, from_layer='log1p', to_layer='deltas')
collection_raw = processing.get_deltas(adata=collection_raw, from_layer='d_log1p', to_layer='d_deltas')
#DONE

#def compute_moran(adata: ad.AnnData, from_layer: str, hex_geometry: bool) -> ad.AnnData:
collection_raw = processing.compute_moran(adata=collection_raw, hex_geometry=True, from_layer='d_log1p')
#DONE

#def filter_by_moran(adata: ad.AnnData, n_keep: int, from_layer: str) -> ad.AnnData:
collection_raw = processing.filter_by_moran(adata=collection_raw, n_keep=data.param_dict['top_moran_genes'], from_layer='d_log1p')
#DONE

#FIXME: error if "mask" doesnt exist
#Assert that a mask must exist
#def add_noisy_layer(adata: ad.AnnData, prediction_layer: str) -> ad.AnnData:
collection_raw.layers['mask'] = collection_raw.layers['tpm'] != 0
collection_raw = processing.add_noisy_layer(adata=collection_raw, prediction_layer="c_d_log1p")
#DONE


