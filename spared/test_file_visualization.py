import datasets
import visualize
import os

breakpoint()
data = datasets.get_dataset("villacampa_lung_organoid", visualize=False)
folder_path="/home/dvegaa/spared/spared/processed_data/villacampa_data/villacampa_lung_organoid"
#Visualize all plots (DONE)
visualize.plot_tests(patch_scale=data.patch_scale, patch_size=data.patch_size, dataset=data.dataset, split_names=data.split_names, param_dict=data.param_dict, folder_path=folder_path, processed_adata=data.adata, raw_adata=data.raw_adata)

#Individual visualizations (DONE)
inv_folder_path="/home/dvegaa/spared/spared/processed_data/villacampa_data/villacampa_lung_organoid/inv_plots"
os.makedirs(inv_folder_path, exist_ok=True)

#def refine_plotting_slides_str(split_names: dict, collection: ad.AnnData, slide_list: str) -> str:
#str_plotting_slides = visualize.refine_plotting_slides_str(split_names: dict, collection: ad.AnnData, slide_list: str)

#def get_plotting_slides_adata(collection: ad.AnnData, slide_list: str) -> list:
#list_plotting_slides = visualize.get_plotting_slides_adata(collection: ad.AnnData, slide_list: str)

#def plot_all_slides(dataset: str, processed_adata: ad.AnnData, path: str) -> None:
visualize.plot_all_slides(dataset=data.dataset, processed_adata=data.adata, path=os.path.join(inv_folder_path, 'all_slides.png'))
#DONE

#def get_exp_frac(adata: ad.AnnData) -> ad.AnnData:
#adata_exp_frac = visualize.get_exp_frac(adata: ad.AnnData)

#def get_glob_exp_frac(adata: ad.AnnData) -> ad.AnnData:
#adata_glob_exp_frac = visualize.get_glob_exp_frac(adata: ad.AnnData)

#def plot_exp_frac(param_dict: dict, dataset: str, raw_adata: ad.AnnData, path: str) -> None:
visualize.plot_exp_frac(param_dict=data.param_dict, dataset=data.dataset, raw_adata=data.raw_adata, path=os.path.join(inv_folder_path, 'exp_frac.png'))
#DONE

#def plot_histograms(processed_adata: ad.AnnData, raw_adata: ad.AnnData, path: str) -> None:
visualize.plot_histograms(processed_adata=data.adata, raw_adata=data.raw_adata, path=os.path.join(inv_folder_path, 'filtering_histograms.png'))
#DONE

#def plot_random_patches(dataset: str, processed_adata: ad.AnnData, path: str, patch_scale: float = 1.0, patch_size: int = 224) -> None:
visualize.plot_random_patches(dataset=data.dataset, processed_adata=data.adata, path=os.path.join(inv_folder_path, 'random_patches.png'), patch_scale=data.patch_scale, patch_size=data.patch_size)
#DONE

#def visualize_moran_filtering(param_dict: dict, processed_adata: ad.AnnData, from_layer: str, path: str, top: bool = True) -> None:
os.makedirs(os.path.join(inv_folder_path, 'top_moran_genes'), exist_ok=True)
os.makedirs(os.path.join(inv_folder_path, 'bottom_moran_genes'), exist_ok=True)
layer = 'c_d_log1p'

visualize.visualize_moran_filtering(param_dict=data.param_dict, processed_adata=data.adata, from_layer=layer, path=os.path.join(inv_folder_path, 'top_moran_genes', f'{layer}.png'), top = True)
visualize.visualize_moran_filtering(param_dict=data.param_dict, processed_adata=data.adata, from_layer=layer, path = os.path.join(inv_folder_path, 'bottom_moran_genes', f'{layer}.png'), top = False)
#DONE

#def visualize_gene_expression(param_dict: dict, processed_adata: ad.AnnData, from_layer: str, path: str) -> None:
os.makedirs(os.path.join(inv_folder_path, 'expression_plots'), exist_ok=True)
layer = 'counts'
visualize.visualize_gene_expression(param_dict=data.param_dict, processed_adata=data.adata, from_layer=layer, path=os.path.join(inv_folder_path,'expression_plots', f'{layer}.png'))
#DONE

#def compute_dim_red(adata: ad.AnnData, from_layer: str) -> ad.AnnData:
#adata_dim_red = visualize.compute_dim_red(adata: ad.AnnData, from_layer: str)

#def plot_clusters(dataset: str, param_dict: dict, processed_adata: ad.AnnData, from_layer: str, path: str) -> None:
os.makedirs(os.path.join(inv_folder_path, 'cluster_plots'), exist_ok=True)
layer = 'c_d_log1p'
visualize.plot_clusters(dataset=data.dataset, param_dict=data.param_dict, processed_adata=data.adata, from_layer=layer, path=os.path.join(inv_folder_path, 'cluster_plots', f'{layer}.png'))
#DONE

#def plot_mean_std(dataset: str, processed_adata: ad.AnnData, raw_adata: ad.AnnData, path: str) -> None:
visualize.plot_mean_std(dataset=data.dataset, processed_adata=data.adata, raw_adata=data.raw_adata, path=os.path.join(inv_folder_path, 'mean_std_scatter.png'))
#DONE

#def plot_data_distribution_stats(dataset: str, processed_adata: ad.AnnData, path:str) -> None:
visualize.plot_data_distribution_stats(dataset=data.dataset, processed_adata=data.adata, path=os.path.join(inv_folder_path, 'splits_stats.png'))
#DONE

#def plot_mean_std_partitions(dataset: str, processed_adata: ad.AnnData, from_layer: str, path: str) -> None:
os.makedirs(os.path.join(inv_folder_path, 'mean_vs_std_partitions'), exist_ok=True)
layer = 'c_d_log1p'
visualize.plot_mean_std_partitions(dataset=data.dataset, processed_adata=data.adata, from_layer=layer, path=os.path.join(inv_folder_path, 'mean_vs_std_partitions', f'{layer}.png'))
#DONE

#VISUALIZACIONES DONE

