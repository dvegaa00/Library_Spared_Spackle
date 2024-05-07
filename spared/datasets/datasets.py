import glob
import scanpy as sc
import anndata as ad
import os
os.environ['USE_PYGEOS'] = '0' # To supress a warning from geopandas
import squidpy as sq
import pandas as pd
from tqdm import tqdm
import numpy as np
from anndata.experimental.pytorch import AnnLoader
from sklearn.preprocessing import StandardScaler
from PIL import Image
import matplotlib
import warnings
import wandb
import gzip
import shutil
import wget
import subprocess
from combat.pycombat import pycombat
import torchvision.models as tmodels
#from . import im_encoder
#from . import metrics
import torch.nn as nn
import torch
from torch.utils.data import DataLoader, Dataset
from matplotlib import pyplot as plt
import matplotlib as mpl
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap   
import matplotlib.colors as colors
from time import time
from datetime import datetime
import json
from torchvision.transforms import Normalize
from typing import Tuple
from torch_geometric.data import Data as geo_Data
from torch_geometric.loader import DataLoader as geo_DataLoader
from torch_geometric.utils import from_scipy_sparse_matrix
from torchvision import transforms
import torch
from positional_encodings.torch_encodings import PositionalEncoding2D
import argparse
import gzip
import plotly.express as px
import pathlib
from scipy.cluster import hierarchy
import random
import sys

#El path a spared es ahora diferente
SPARED_PATH = pathlib.Path(__file__).resolve().parent.parent

#Agregar el directorio padre al sys.path para los imports
sys.path.append(str(SPARED_PATH))
#import im_encoder and metrics files
from embeddings import im_encoder
from metrics import metrics

# Import visualization and processing function
from visualize import visualize
from processing import processing

# Import all reader classes
from readers.AbaloReader import AbaloReader
from readers.BatiukReader import BatiukReader
from readers.EricksonReader import EricksonReader
from readers.FanReader import FanReader
from readers.MirzazadehReader import MirzazadehReader
from readers.ParigiReader import ParigiReader
from readers.VicariReader import VicariReader
from readers.VillacampaReader import VillacampaReader
from readers.VisiumReader import VisiumReader
from readers.HudsonReader import HudsonReader
#Remover el directorio padre al sys.path 
sys.path.append(str(SPARED_PATH))

# Remove the max limit of pixels in a figure
Image.MAX_IMAGE_PIXELS = None

# Get the path of the spared database
#SPARED_PATH = pathlib.Path(__file__).parent

# Set warnings to ignore
warnings.filterwarnings("ignore", message="No data for colormapping provided via 'c'. Parameters 'cmap' will be ignored")
warnings.filterwarnings("ignore", message="Variable names are not unique. To make them unique, call `.var_names_make_unique`.")
warnings.filterwarnings("ignore", message="The 'nopython' keyword argument was not supplied to the 'numba.jit' decorator. The implicit default value for this argument is currently False, but it will be changed to True in Numba 0.59.0. See https://numba.readthedocs.io/en/stable/reference/deprecation.html#deprecation-of-object-mode-fall-back-behaviour-when-using-jit for details.")
warnings.filterwarnings("ignore", message="Changing the sparsity structure of a csr_matrix is expensive. lil_matrix is more efficient.")

# FIXME: Fix this warning FutureWarning: is_categorical_dtype is deprecated and will be removed in a future version. Use isinstance(dtype, CategoricalDtype) instead
# This is a problem between anndata and pandas
warnings.filterwarnings("ignore", category=FutureWarning, module='anndata')
warnings.filterwarnings("ignore", category=FutureWarning, module='squidpy')
warnings.filterwarnings("ignore", category=FutureWarning, module='scanpy')


# TODO: Think of implementing optional random subsampling of the dataset
class SpatialDataset():
    def __init__(self,
        dataset: str = 'V1_Breast_Cancer_Block_A',
        param_dict: dict = {
            'cell_min_counts':      1000,
            'cell_max_counts':      100000,
            'gene_min_counts':      1e3,
            'gene_max_counts':      1e6,
            'min_exp_frac':         0.8,
            'min_glob_exp_frac':    0.8,
            'real_data_percentage': 0.7,
            'top_moran_genes':      256,
            'wildcard_genes':       'None',    
            'combat_key':           'slide_id',
            'random_samples':       -1,        
            'plotting_slides':      'None',      
            'plotting_genes':       'None',       
                            }, 
        patch_scale: float = 1.0,
        patch_size: int = 224,
        force_compute: bool = False
        ):
        """
        This is a spatial data class that contains all the information about the dataset. It will call a reader class depending on the type
        of dataset (by now only visium and STNet are supported). The reader class will download the data and read it into an AnnData collection
        object. Then the dataset class will filter, process and plot quality control graphs for the dataset. The processed dataset will be stored
        for rapid access in the future.

        Args:
            dataset (str, optional): An string encoding the dataset type. Defaults to 'V1_Breast_Cancer_Block_A'.
            param_dict (dict, optional): Dictionary that contains filtering and processing parameters.
                                        Detailed information about each key can be found in the parser definition over utils.py. 
                                        Defaults to {
                                                'cell_min_counts':      1000,
                                                'cell_max_counts':      100000,
                                                'gene_min_counts':      1e3,
                                                'gene_max_counts':      1e6, 
                                                'min_exp_frac':         0.8,
                                                'min_glob_exp_frac':    0.8,
                                                'real_data_percentage': 0.7,
                                                'top_moran_genes':      256,
                                                'wildcard_genes':       'None',
                                                'combat_key':           'slide_id',
                                                'random_samples':       -1,
                                                'plotting_slides':      'None',
                                                'plotting_genes':       'None',
                                                }.
            patch_scale (float, optional): The scale of the patches to take into account. If bigger than 1, then the patches will be bigger than the original image. Defaults to 1.0.
            patch_size (int, optional): The pixel size of the patches. Defaults to 224.
            force_compute (bool, optional): Whether to force the processing computation or not. Defaults to False.
        """

        # We define the variables for the SpatialDataset class
        self.dataset = dataset
        self.param_dict = param_dict
        self.patch_scale = patch_scale
        self.patch_size = patch_size
        self.force_compute = force_compute
        self.hex_geometry = False if self.dataset == 'stnet_dataset' else True # FIXME: Be careful with this attribute if we want to include more technologies

        # We initialize the reader class (Both visium or stnet readers can be returned here)
        self.reader_class = self.initialize_reader()
        # We get the dict of split names
        self.split_names = self.reader_class.split_names
        # We get the dataset download path
        self.download_path = self.reader_class.download_path
        # Get the dataset path
        self.dataset_path = self.reader_class.dataset_path
        # We load or compute the processed adata with patches.
        self.adata = self.load_or_compute_adata()

    def initialize_reader(self):
        """
        This function uses the parameters of the class to initialize the appropiate reader class
        (Visium or STNet) and returns the reader class.
        """

        if 'vicari' in self.dataset:
            reader_class = VicariReader(
                dataset=self.dataset,
                param_dict=self.param_dict,
                patch_scale=self.patch_scale,
                patch_size=self.patch_size,
                force_compute=self.force_compute
            )
        elif 'villacampa' in self.dataset:
            reader_class = VillacampaReader(
                dataset=self.dataset,
                param_dict=self.param_dict,
                patch_scale=self.patch_scale,
                patch_size=self.patch_size,
                force_compute=self.force_compute
            )
        elif 'mirzazadeh' in self.dataset:
            reader_class = MirzazadehReader(
                dataset=self.dataset,
                param_dict=self.param_dict,
                patch_scale=self.patch_scale,
                patch_size=self.patch_size,
                force_compute=self.force_compute
            )
        elif 'abalo' in self.dataset:
            reader_class = AbaloReader(
                dataset=self.dataset,
                param_dict=self.param_dict,
                patch_scale=self.patch_scale,
                patch_size=self.patch_size,
                force_compute=self.force_compute
            )
        elif 'erickson' in self.dataset:
            reader_class = EricksonReader(
                dataset=self.dataset,
                param_dict=self.param_dict,
                patch_scale=self.patch_scale,
                patch_size=self.patch_size,
                force_compute=self.force_compute
            )
        elif 'batiuk' in self.dataset:
            reader_class = BatiukReader(
                dataset=self.dataset,
                param_dict=self.param_dict,
                patch_scale=self.patch_scale,
                patch_size=self.patch_size,
                force_compute=self.force_compute
            )
        elif 'parigi' in self.dataset:
            reader_class = ParigiReader(
                dataset=self.dataset,
                param_dict=self.param_dict,
                patch_scale=self.patch_scale,
                patch_size=self.patch_size,
                force_compute=self.force_compute
            )
        elif 'fan' in self.dataset:
            reader_class = FanReader(
                dataset=self.dataset,
                param_dict=self.param_dict,
                patch_scale=self.patch_scale,
                patch_size=self.patch_size,
                force_compute=self.force_compute
            )
        elif 'hudson' in self.dataset:
            reader_class = HudsonReader(
                dataset=self.dataset,
                param_dict=self.param_dict,
                patch_scale=self.patch_scale,
                patch_size=self.patch_size,
                force_compute=self.force_compute
            )    
        else:
            reader_class = VisiumReader(
                dataset=self.dataset,
                param_dict=self.param_dict,
                patch_scale=self.patch_scale,
                patch_size=self.patch_size,
                force_compute=self.force_compute
            )
        
        return reader_class
    
    # TODO: Test the complete processing pipeline against R implementation (GeoTcgaData package for TPM) (should we keep this todo?)
    # TODO: Update the docstring of this function (regarding process_dataset)
        
    def load_or_compute_adata(self) -> ad.AnnData:
        """
        This function does the main data pipeline. It will first check if the processed data exists in the dataset_path. If it does not exist,
        then it will compute it and save it. If it does exist, then it will load it and return it. If it is in the compute mode, then it will
        also save quality control plots.

        Returns:
            ad.AnnData: The processed AnnData object ready to be used for training.
        """
        # If processed data does not exist, then compute and save it 
        if (not os.path.exists(os.path.join(self.dataset_path, f'adata.h5ad'))) or self.force_compute:
            
            print('Computing main adata file from downloaded raw data...')
            collection_raw = self.reader_class.get_adata_collection()
            collection_filtered = processing.filter_dataset(adata=collection_raw, param_dict=self.param_dict)
            collection_processed = processing.process_dataset(
                dataset=self.dataset, adata=collection_filtered, param_dict=self.param_dict, hex_geometry=self.hex_geometry)

            # Save the processed data
            os.makedirs(self.dataset_path, exist_ok=True)
            collection_raw.write(os.path.join(self.dataset_path, f'adata_raw.h5ad'))
            collection_processed.write(os.path.join(self.dataset_path, f'adata.h5ad'))

            # QC plotting
            #TODO: ¿Borrar la parte de visualización?
            #visualize.plot_tests(self.patch_scale, self.patch_size, self.dataset, self.split_names, self.param_dict, self.dataset_path, collection_processed, collection_raw)
            # Copy figures folder into public database
            os.makedirs(os.path.join(SPARED_PATH, 'PublicDatabase', self.dataset), exist_ok=True)
            #if os.path.exists(os.path.join(SPARED_PATH, 'PublicDatabase', self.dataset, 'qc_plots')):
            #    shutil.rmtree(os.path.join(SPARED_PATH, 'PublicDatabase', self.dataset, 'qc_plots'))
            #shutil.copytree(os.path.join(self.dataset_path, 'qc_plots'), os.path.join(SPARED_PATH, 'PublicDatabase', self.dataset, 'qc_plots'), dirs_exist_ok=True)
            
            # Create README for dataset
            if not os.path.exists(os.path.join(SPARED_PATH, 'PublicDatabase', self.dataset, 'README.md')):
                shutil.copy(os.path.join(SPARED_PATH, 'PublicDatabase', 'README_template.md'), os.path.join(SPARED_PATH, 'PublicDatabase', self.dataset, 'README.md'))

        else:            
            # Load processed adata
            print(f'Loading main adata file from disk ({os.path.join(self.dataset_path, f"adata.h5ad")})...')
            # If the file already exists, load it
            collection_processed = ad.read_h5ad(os.path.join(self.dataset_path, f'adata.h5ad'))
            

            print('The loaded adata object looks like this:')
            print(collection_processed)

            # QC plotting if force plotting
            force_plotting = False
            if force_plotting:
                collection_raw = ad.read_h5ad(os.path.join(self.dataset_path, f'adata_raw.h5ad'))
                #TODO: ¿Borrar la parte de visualización?
                #visualize.plot_tests(self.patch_scale, self.patch_size, self.dataset, self.split_names, self.param_dict, self.dataset_path, collection_processed, collection_raw)
                # Copy figures folder into public database
                os.makedirs(os.path.join(SPARED_PATH, 'PublicDatabase', self.dataset), exist_ok=True)
                #if os.path.exists(os.path.join(SPARED_PATH, 'PublicDatabase', self.dataset, 'qc_plots')):
                #    shutil.rmtree(os.path.join(SPARED_PATH, 'PublicDatabase', self.dataset, 'qc_plots'))
                #shutil.copytree(os.path.join(self.dataset_path, 'qc_plots'), os.path.join(SPARED_PATH, 'PublicDatabase', self.dataset, 'qc_plots'), dirs_exist_ok=True)
                
                # Create README for dataset
                if not os.path.exists(os.path.join(SPARED_PATH, 'PublicDatabase', self.dataset, 'README.md')):
                    shutil.copy(os.path.join(SPARED_PATH, 'PublicDatabase', 'README_template.md'), os.path.join(SPARED_PATH, 'PublicDatabase', self.dataset, 'README.md'))

        return collection_processed
    
    # TODO: Chere where this is being used to update it into a processing function import
    # IMPORT FROM PROCESSING -> def compute_patches_embeddings_and_predictions(self, backbone: str ='densenet', model_path:str="best_stnet.pt", preds: bool=True) -> None:
    # def compute_patches_embeddings_and_predictions(adata: ad.AnnData, backbone: str ='densenet', model_path:str="best_stnet.pt", preds: bool=True, patch_size: int = 224, patch_scale: float=1.0)

    # TODO: Check where is this being used and if we should keep it
    def obtain_embeddings_resnet50(self):

        def extract(encoder, patches):
            return encoder(patches).view(-1,features)

        egn_transforms = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        resnet_encoder = tmodels.resnet50(True)
        features = resnet_encoder.fc.in_features
        modules = list(resnet_encoder.children())[:-1] # Encoder corresponds to ResNet50 without the fc layer
        resnet_encoder = torch.nn.Sequential(*modules)
        for p in resnet_encoder.parameters():
            p.requires_grad = False

        resnet_encoder = resnet_encoder.to(device)
        resnet_encoder.eval()

        # Get the patches
        flat_patches = self.adata.obsm[f'patches_scale_{self.patch_scale}']

        # Reshape all the patches to the original shape
        all_patches = flat_patches.reshape((-1, self.patch_size, self.patch_size, 3))
        torch_patches = torch.from_numpy(all_patches).permute(0, 3, 1, 2).float()    # Turn all patches to torch
        rescaled_patches = egn_transforms(torch_patches / 255)                       # Rescale patches to [0, 1]

        img_embedding = [extract(resnet_encoder, single_patch.unsqueeze(dim=0).to(device)) for single_patch in tqdm(rescaled_patches)]
        img_embedding = torch.cat(img_embedding).contiguous()             
        
        self.adata.obsm['resnet50_embeddings'] = img_embedding.cpu().numpy()
    
    # TODO: Check where is this being used and if we should keep it
    def get_nn_images(self) -> None:

        def get_nn_dist_and_ids_images(query_adata: ad.AnnData, ref_adata: ad.AnnData) -> Tuple[pd.DataFrame, pd.DataFrame]:
            
            # Get embeddings from query and ref as torch tensors
            query_embeddings = torch.Tensor(query_adata.obsm['resnet50_embeddings'])
            ref_embeddings = torch.Tensor(ref_adata.obsm['resnet50_embeddings'])

            # Compute euclidean distances from query to ref
            query_ref_distances = torch.cdist(query_embeddings, ref_embeddings, p=2)

            # Get the sorted distances and indexes
            sorted_distances, sorted_indexes = torch.sort(query_ref_distances, dim=1)

            # Trim the sorted distances and indexes to 100 nearest neighbors and convert to numpy
            sorted_distances = sorted_distances[:, :100].numpy()
            sorted_indexes = sorted_indexes[:, :100].numpy()

            # Get index vector in numpy (just to avoid warnings)
            index_vector = ref_adata.obs.index.values

            # Get the ids of the 100 nearest neighbors
            sorted_ids = index_vector[sorted_indexes]
            
            # Make a dataframe with the distances and ids with the query index as index
            sorted_distances_df = pd.DataFrame(sorted_distances, index=query_adata.obs.index.values)
            sorted_ids_df = pd.DataFrame(sorted_ids, index=query_adata.obs.index.values)

            return sorted_distances_df, sorted_ids_df

        print('Getting image nearest neighbors...')
        start = time()

        # Define train subset (this is where val and test will look for nearest neighbors)
        train_subset = self.adata[self.adata.obs['split'] == 'train']
        val_subset = self.adata[self.adata.obs['split'] == 'val']
        test_subset = self.adata[self.adata.obs['split'] == 'test']
        
        # Use the get_nn_dist_and_ids function to get the distances and ids of the nearest neighbors
        val_train_distances_df, val_train_ids_df = get_nn_dist_and_ids_images(val_subset, train_subset)
        test_train_distances_df, test_train_ids_df = get_nn_dist_and_ids_images(test_subset, train_subset)

        # Now get the patients of the train set
        train_patients = train_subset.obs['patient'].unique()
        
        # Define list of dataframes to store the distances and ids from the train set to the train set
        train_train_distances_dfs = []
        train_train_ids_dfs = []

        # Cycle through train patients
        for patient in train_patients:

            # Get the train patient data
            patient_data = train_subset[train_subset.obs['patient'] == patient]
            # Get the train patient data that is not for the current patient
            other_patient_data = train_subset[train_subset.obs['patient'] != patient]

            # Apply the get_nn_dist_and_ids function to get the distances and ids of the nearest neighbors
            curr_patient_distances_df, curr_patient_ids_df = get_nn_dist_and_ids_images(patient_data, other_patient_data)
            
            # Append the dataframes to the list
            train_train_distances_dfs.append(curr_patient_distances_df)
            train_train_ids_dfs.append(curr_patient_ids_df)

        # Concatenate the dataframes
        train_train_distances_df = pd.concat(train_train_distances_dfs)
        train_train_ids_df = pd.concat(train_train_ids_dfs)

        # Concatenate train, val and test distances and ids
        all_distances_df = pd.concat([train_train_distances_df, val_train_distances_df, test_train_distances_df])
        all_ids_df = pd.concat([train_train_ids_df, val_train_ids_df, test_train_ids_df])

        # Reindex the dataframes
        all_distances_df = all_distances_df.reindex(self.adata.obs.index.values)
        all_ids_df = all_ids_df.reindex(self.adata.obs.index.values)
        
        # Add the dataframes to the obsm
        self.adata.obsm['image_nn_distances'] = all_distances_df
        self.adata.obsm['image_nn_ids'] = all_ids_df

        end = time()
        print(f'Finished getting image nearest neighbors in {end - start:.2f} seconds')

    # IMPORT FROM PROCESSING -> def get_graph_dataloaders(self, layer: str = 'c_d_log1p', n_hops: int = 2, backbone: str ='densenet', model_path: str = "best_stnet.pt", batch_size: int = 128, shuffle: bool = True) -> Tuple[geo_DataLoader, geo_DataLoader, geo_DataLoader]:
    # def get_graph_dataloaders(adata: ad.AnnData, dataset_path: str='', layer: str = 'c_d_log1p', n_hops: int = 2, backbone: str ='densenet', model_path: str = "best_stnet.pt", batch_size: int = 128, 
                          # shuffle: bool = True, hex_geometry: bool=True, patch_size: int=224, patch_scale: float=1.0)

    # TODO: Check where is this being used and if we should keep it
    def log_pred_image(self, n_genes: int = 2, slides: dict = {}):
        
        ### 1. Get the selected genes

        # Get prediction and groundtruth layers from layers keys in the anndata
        pred_layer = [l for l in self.adata.layers.keys() if 'predictions' in l]
        pred_layer = pred_layer[0] if pred_layer else None
        gt_layer = [l.split(',')[1] for l in self.adata.layers.keys() if 'predictions' in l]
        gt_layer = gt_layer[0] if gt_layer else None
        # Handle delta prediction in normal scale
        if 'deltas' in gt_layer:
            gt_layer = gt_layer.replace('deltas', 'log1p')
        # Be sure the prediction layer and gt layer is present in dataset
        assert not (pred_layer is None), 'predictions layer not present in the adata'
        assert not (gt_layer is None), 'groundtruth layer not present in the adata'

        # Get partition names of current dataset
        partitions = list(self.adata.obs.split.unique())
        # Get dict of adatas separated by splits
        adatas_dict = {p: self.adata[self.adata.obs.split == p, :] for p in partitions}

        # Compute and ad detailed metrics for each split
        for p, curr_adata in adatas_dict.items():

            # Get detailed metrics from partition
            detailed_metrics = metrics.get_metrics(
                gt_mat = curr_adata.to_df(layer=gt_layer).values,
                pred_mat = curr_adata.to_df(layer=pred_layer).values,
                mask = curr_adata.to_df(layer='mask').values,
                detailed=True
            )

            # Add detalied metrics to global adata
            self.adata.var[f'pcc_{p}'] = detailed_metrics['detailed_PCC-Gene']
            self.adata.var[f'r2_{p}'] = detailed_metrics['detailed_R2-Gene']
            self.adata.var[f'mse_{p}'] = detailed_metrics['detailed_mse_gene']
            self.adata.var[f'mae_{p}'] = detailed_metrics['detailed_mae_gene']
            
            # Define plotly plots
            pcc_fig = px.histogram(self.adata.var, x=f'pcc_{p}', marginal='rug', hover_data=self.adata.var.columns)
            r2_fig = px.histogram(self.adata.var, x=f'r2_{p}', marginal='rug', hover_data=self.adata.var.columns)
            mse_fig = px.histogram(self.adata.var, x=f'mse_{p}', marginal='rug', hover_data=self.adata.var.columns)
            mae_fig = px.histogram(self.adata.var, x=f'mae_{p}', marginal='rug', hover_data=self.adata.var.columns)

            # Log plotly plot to wandb
            wandb.log({f'pcc_gene_{p}': wandb.Plotly(pcc_fig)})
            wandb.log({f'r2_gene_{p}': wandb.Plotly(r2_fig)})
            wandb.log({f'mse_gene_{p}': wandb.Plotly(mse_fig)})
            wandb.log({f'mae_gene_{p}': wandb.Plotly(mae_fig)})            
        
        # Get ordering split
        order_split = 'test' if 'test' in partitions else 'val'
        # Get selected genes based on the best pcc
        n_top = self.adata.var.nlargest(n_genes, columns=f'pcc_{order_split}').index.to_list()
        n_botom = self.adata.var.nsmallest(n_genes, columns=f'pcc_{order_split}').index.to_list()
        selected_genes = n_top + n_botom

        ### 2. Get the selected slides. NOTE: Only first slide is always selected in case slides is not specified by parameter.
        if slides == {}:
            slides = {p: list(adatas_dict[p].obs.slide_id.unique())[0] for p in partitions}
        
        def log_one_gene(self, gene, slides, gt_layer, pred_layer):
            
            # Get dict of individual slides adatas
            slides_adatas_dict = {p: processing.get_slide_from_collection(self.adata, slides[p]) for p in slides.keys()}

            # Get min and max of the selected gene in the slides
            gene_min_pred = min([dat[:, gene].layers[pred_layer].min() for dat in slides_adatas_dict.values()])
            gene_max_pred = max([dat[:, gene].layers[pred_layer].max() for dat in slides_adatas_dict.values()])
            
            gene_min_gt = min([dat[:, gene].layers[gt_layer].min() for dat in slides_adatas_dict.values()])
            gene_max_gt = max([dat[:, gene].layers[gt_layer].max() for dat in slides_adatas_dict.values()])
            
            gene_min = min([gene_min_pred, gene_min_gt])
            gene_max = max([gene_max_pred, gene_max_gt])

            # Define color normalization
            norm = matplotlib.colors.Normalize(vmin=gene_min, vmax=gene_max)

            # Declare figure
            fig, ax = plt.subplots(nrows=len(slides), ncols=4, layout='constrained')
            fig.set_size_inches(13, 3 * len(slides))

            # Define order of rows in dict
            order_dict = {'train': 0, 'val': 1, 'test': 2}

            # Iterate over partitions
            for p in slides.keys():
                
                # Get current row
                row = order_dict[p]
                
                curr_img = slides_adatas_dict[p].uns['spatial'][slides[p]]['images']['lowres']
                ax[row,0].imshow(curr_img)
                ax[row,0].set_ylabel(f'{p}:\n{slides[p]}\nPCC-Gene={round(self.adata.var.loc[gene, f"pcc_{p}"],3)}', fontsize='large')
                ax[row,0].set_xticks([])
                ax[row,0].set_yticks([])

                # Plot gt and pred of gene in the specified slides
                sq.pl.spatial_scatter(slides_adatas_dict[p], color=[gene], layer=gt_layer, ax=ax[row,1], cmap='jet', norm=norm, colorbar=False, title='')
                sq.pl.spatial_scatter(slides_adatas_dict[p], color=[gene], layer=pred_layer, ax=ax[row,2], cmap='jet', norm=norm, colorbar=False, title='')
                sq.pl.spatial_scatter(slides_adatas_dict[p], color=[gene], layer=pred_layer, ax=ax[row,3], cmap='jet', colorbar=True, title='')
                
                # Set y labels
                ax[row,1].set_ylabel('')
                ax[row,2].set_ylabel('', fontsize='large')
                ax[row,3].set_ylabel('')


            # Format figure
            for axis in ax.flatten():
                axis.set_xlabel('')
                # Turn off all spines
                axis.spines['top'].set_visible(False)
                axis.spines['right'].set_visible(False)
                axis.spines['bottom'].set_visible(False)
                axis.spines['left'].set_visible(False)

            # Refine figure appereance
            ax[0, 0].set_title(f'Original Image', fontsize='large')
            ax[0, 1].set_title(f'Groundtruth\nFixed Scale', fontsize='large')
            ax[0, 2].set_title(f'Prediction\nFixed Scale', fontsize='large')
            ax[0, 3].set_title(f'Prediction\nVariable Scale', fontsize='large')

            # Add fixed colorbar
            fig.colorbar(matplotlib.cm.ScalarMappable(norm=norm, cmap='jet'), ax=ax[:3, 2], location='right', fraction=0.05, aspect=25*len(slides)+5)

            # Get ordering split
            order_split = 'test' if 'test' in slides.keys() else 'val'
            fig.suptitle(f'{gene}: PCC_{order_split}={round(self.adata.var.loc[gene, f"pcc_{order_split}"],3)}', fontsize=20)
            # Log plot 
            wandb.log({gene: fig})
            plt.close()

        # FIXME: Wandb logging is not working
        def log_pred_correlations(self, pred_layer, top_k=50):

            """This function logs the correlation matrices of the top k genes in the ground truth and predictions layers.

            Args:
                pred_layer (str): The name of the predictions layer.
                top_k (int, optional): The number of top genes to log. Defaults to 50.
            """
            
            # Get prediction and groundtruth layers from layers keys in the anndata
            exp_pred = self.adata.layers[pred_layer]
            exp_gt = self.adata.layers[gt_layer]

            #take mean of expression
            mean = np.mean(exp_gt, axis=1)
            #find the indices of the top_k highest values
            ind = np.argpartition(mean, -top_k)[-top_k:]

            # Compute the correlation matrixes subsetting the top k genes
            corr_matrix_gt = np.corrcoef(exp_gt[ind, :])
            corr_matrix_pred = np.corrcoef(exp_pred[ind, :])

            # Hierarchical clustering
            dendrogram = hierarchy.dendrogram(hierarchy.linkage(corr_matrix_gt, method='ward'), no_plot=True)
            # Get the cluster indexes
            cluster_idx = dendrogram['leaves']

            # Reaorder the correlation matrixes based on the clustering results
            corr_matrix_pred = corr_matrix_pred[cluster_idx, :]
            corr_matrix_pred = corr_matrix_pred[:, cluster_idx]
            corr_matrix_gt = corr_matrix_gt[cluster_idx, :]
            corr_matrix_gt = corr_matrix_gt[:, cluster_idx]

            # Create subplots
            fig, axes = plt.subplots(1, 2, layout='constrained')
            fig.set_size_inches(10, 5)

            # Calculate min and max for color normalization
            vmin = min(corr_matrix_gt.min(), corr_matrix_pred.min())
            vmax = max(corr_matrix_gt.max(), corr_matrix_pred.max())
            
            # Plot heatmap for predictions
            sns.heatmap(corr_matrix_pred, ax=axes[0], xticklabels=False, cmap='viridis', yticklabels=False, cbar=False, vmin=vmin, vmax=vmax)
            axes[0].set_title("Predictions")

            # Plot heatmap for ground truth
            sns.heatmap(corr_matrix_gt, ax=axes[1], xticklabels=False, cmap='viridis', yticklabels=False, cbar=True, vmin=vmin, vmax=vmax)
            axes[1].set_title("Ground Truth")

            # Log plot
            wandb.log({'correlations': wandb.Image(fig)})
            plt.close()

        #TODO: Mean and variance plot
        def log_mean_variance(self, pred_layer):

            # Get prediction and groundtruth layers from layers keys in the anndata
            exp_pred = self.adata.layers[pred_layer]
            exp_gt = self.adata.layers[gt_layer]

            #take the mean expression for each gene 
            mean_gt = np.mean(exp_gt, axis=0)
            mean_pred = np.mean(exp_pred, axis=0)
            #take the variance of expression for each gene 
            var_gt = np.var(exp_gt, axis=0)
            var_pred = np.var(exp_pred, axis=0)

            #sort genes by mean expression
            sorted_indexes = np.argsort(mean_gt)
            mean_gt = mean_gt[sorted_indexes]
            mean_pred = mean_pred[sorted_indexes]

            #sort genes by variance
            sorted_indexes = np.argsort(var_gt)
            var_gt = var_gt[sorted_indexes]
            var_pred = var_pred[sorted_indexes]

            #Make plots

            # Prepare data
            genes = range(len(mean_pred))  
            data_mean = pd.DataFrame({
                'Genes': genes,
                'Mean Expression Prediction': mean_pred,
                'Mean Expression Ground Truth': mean_gt
            })
            data_variance = pd.DataFrame({
                'Genes': genes,
                'Variance Prediction': var_pred,
                'Variance Ground Truth': var_gt
            })

            # Figure configurations
            #sns.set_style("darkgrid")
            sns.set_palette("tab10")
            
            # Crear la figura y los ejes
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

            # Mean subplot
            sns.scatterplot(ax=ax1, data=data_mean.melt(id_vars='Genes', var_name='Condition', value_name='Mean Expression'),
                            x='Genes', y='Mean Expression', hue='Condition', palette=['orange', 'blue'], s=10, alpha=0.5)
            ax1.set_xlabel('Genes')
            ax1.set_ylabel('Mean Expression')

            # Variance subplot
            sns.scatterplot(ax=ax2, data=data_variance.melt(id_vars='Genes', var_name='Condition', value_name='Variance'),
                            x='Genes', y='Variance', hue='Condition', palette=['orange', 'blue'], s=10, alpha=0.5)
            ax2.set_xlabel('Genes')
            ax2.set_ylabel('Variance')

            # Crear una leyenda personalizada con las etiquetas deseadas y colocarla en el gráfico
            handles, labels = ax1.get_legend_handles_labels()
            fig.legend(handles, ['Prediction', 'Ground Truth'], loc='upper center', ncol=2, frameon=False)

            # Quitar spines
            sns.despine()

            # Hide individual legends
            ax1.get_legend().remove()
            ax2.get_legend().remove()

            # Log plot
            wandb.log({'mean_variance': wandb.Image(fig)})
            plt.close()
        
        for gene in selected_genes:
            log_one_gene(self, gene, slides, gt_layer, pred_layer)
        
        log_pred_correlations(self, pred_layer)
        log_mean_variance(self, pred_layer)


class HisToGeneDataset(Dataset):
    def __init__(self, adata, set_str):
        self.set = set_str
        if self.set == None:
            self.adata = adata
        else:
            self.adata = adata[adata.obs.split == self.set]
        self.idx_2_slide = {idx: slide for idx, slide in enumerate(self.adata.obs.slide_id.unique())}
        
        #Perform transformations
        self.transforms = Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        tissue_tiles = self.adata.obsm['patches_scale_1.0']
        # Pass to torch tensor
        tissue_tiles = torch.from_numpy(tissue_tiles)
        w = round(np.sqrt(tissue_tiles.shape[1]/3))
        tissue_tiles = tissue_tiles.reshape((tissue_tiles.shape[0], w, w, -1))
        # Permute dimensions to be in correct order for normalization
        tissue_tiles = tissue_tiles.permute(0,3,1,2).contiguous()
        # Make transformations in tissue tiles
        tissue_tiles = tissue_tiles/255.
        # Transform tiles
        tissue_tiles = self.transforms(tissue_tiles)
        # Flatten tiles
        # self.adata.obsm['patches_scale_1.0_transformed'] = tissue_tiles.view(tissue_tiles.shape[0], -1)
        self.adata.obsm['patches_scale_1.0_transformed_numpy'] = tissue_tiles.view(tissue_tiles.shape[0], -1).numpy()

        # Define mask layer
        self.adata.layers['mask'] = self.adata.layers['tpm'] != 0

    def __len__(self):
        return len(self.idx_2_slide)

    def __getitem__(self, idx):
        
        # Get the slide from the index
        slide = self.idx_2_slide[idx]
        # Get the adata of the slide
        adata_slide = self.adata[self.adata.obs.slide_id == slide]

        # Get the patches
        patch = torch.from_numpy(adata_slide.obsm['patches_scale_1.0_transformed_numpy'])
        # Get the coordinates
        coor = torch.from_numpy(adata_slide.obs[['array_row', 'array_col']].values)
        # Get the expression
        exp = torch.from_numpy(adata_slide.X.toarray())
        # Get the mask
        mask = torch.from_numpy(adata_slide.layers['mask'])
        
        return patch, coor, exp, mask


def get_dataset(dataset_name: str) -> SpatialDataset:
    """
    This function receives the name of a dataset and retrieves a SpatialDataset object according to the arguments.
    Args:
        dataset_name (str): The name of the dataset.
    Returns:
        dataset: The specified dataset in a SpatialDataset object.
    """
    
    # Get the name of the config based on the dataset
    config = os.path.join(SPARED_PATH, 'configs', f'{dataset_name}.json')
    # Load the config
    with open(config, 'r') as f:
        config_dict = json.load(f)

    # Assign auxiliary variables for dataset
    patch_scale = config_dict['patch_scale']
    patch_size = config_dict['patch_size']
    force_compute = config_dict['force_compute']

    # Refine config dict into a param dict
    [config_dict.pop(k) for k in ['patch_scale', 'patch_size', 'force_compute', 'dataset', 'n_hops', 'prediction_layer']]

    # Declare the spatial dataset
    dataset = SpatialDataset(
        dataset=dataset_name, 
        param_dict=config_dict, 
        patch_scale=patch_scale, 
        patch_size=patch_size, 
        force_compute=force_compute
    )

    return dataset
   

# Test code only for debugging
if __name__ == "__main__":
    
    # Auxiliary function to use booleans in parser
    str2bool = lambda x: (str(x).lower() == 'true')

    # Define a simple parser and add an argument for the config file
    parser = argparse.ArgumentParser(description='Test code for datasets.')
    parser.add_argument('--config',             type=str,       default=os.path.join(SPARED_PATH, 'configs', '10xgenomic_human_breast_cancer.json'), help='Path to the config file.')
    parser.add_argument('--prepare_datasets',   type=str2bool,  default=False, help='If True then it processes all datasets.')
    parser.add_argument('--graphs_ie_paths',    type=str,       default='None', help='Path to the folders with optimal image encoder models to get graphs from. E.g. os.path.join("optimal_models", "spared_vit_backbone_c_d_deltas"')
    args = parser.parse_args()
    
    # Define dataset list
    dataset_list = ['10xgenomic_human_brain',                      '10xgenomic_human_breast_cancer',
                    '10xgenomic_mouse_brain_coronal',              '10xgenomic_mouse_brain_sagittal_anterior',
                    '10xgenomic_mouse_brain_sagittal_posterior',   'abalo_human_squamous_cell_carcinoma',
                    'erickson_human_prostate_cancer_p1',           'erickson_human_prostate_cancer_p2',
                    'fan_mouse_brain_coronal',                     'fan_mouse_olfatory_bulb',
                    'mirzazadeh_human_colon_p1',                   'mirzazadeh_human_colon_p2',
                    'mirzazadeh_human_pediatric_brain_tumor_p1',   'mirzazadeh_human_pediatric_brain_tumor_p2',
                    'mirzazadeh_human_prostate_cancer',            'mirzazadeh_human_small_intestine',
                    'mirzazadeh_mouse_bone',                       'mirzazadeh_mouse_brain_p1',
                    'mirzazadeh_mouse_brain_p2',                   'mirzazadeh_mouse_brain',
                    'parigi_mouse_intestine',                      'vicari_human_striatium',
                    'vicari_mouse_brain',                          'villacampa_kidney_organoid',
                    'villacampa_lung_organoid',                    'villacampa_mouse_brain']
    
    # If prepare datasets then run the dataset pipeline for all available datasets
    if args.prepare_datasets == True:
        
        # Define complete config files list
        config_list = [os.path.join(SPARED_PATH, 'configs', f'{dset}.json') for dset in dataset_list]

        # Iterate over config files
        for curr_config_path in config_list:

           # Load the config file
            with open(curr_config_path, 'r') as f:
                config = json.load(f)

            # Define param dict
            param_dict = {
                'cell_min_counts':       config['cell_min_counts'],
                'cell_max_counts':       config['cell_max_counts'],
                'gene_min_counts':       config['gene_min_counts'],
                'gene_max_counts':       config['gene_max_counts'],
                'min_exp_frac':          config['min_exp_frac'],
                'min_glob_exp_frac':     config['min_glob_exp_frac'],
                'real_data_percentage':  config['real_data_percentage'],
                'top_moran_genes':       config['top_moran_genes'],
                'wildcard_genes':        config['wildcard_genes'],
                'combat_key':            config['combat_key'],
                'random_samples':        config['random_samples'],
                'plotting_slides':       config['plotting_slides'],
                'plotting_genes':        config['plotting_genes'],
            }

            # Process the dataset and store it as adata
            test_dataset = SpatialDataset(
                dataset =       config['dataset'], 
                param_dict =    param_dict, 
                patch_scale =   config['patch_scale'], 
                patch_size =    config['patch_size'], 
                force_compute = config['force_compute']
                ) 
    
    elif args.graphs_ie_paths != 'None':

        # iterate over datasets
        for dset in dataset_list:
            # Get the dataset
            dataset = get_dataset(dset)
            # Get model path
            model_path = glob.glob(os.path.join(args.graphs_ie_paths, f'{dset}', '**', '*.ckpt'), recursive=True)[0]
            # Get layer
            layer_dict = {
                'spared_vit_backbone_c_d_deltas': 'c_d_deltas',
                'spared_vit_backbone_c_t_deltas': 'c_t_deltas',
                'spared_vit_backbone_noisy_d':    'noisy_d',
            }
            layer = layer_dict[os.path.basename(os.path.normpath(args.graphs_ie_paths))]
            # Get the graphs
            train_dl, val_dl, test_dl = processing.get_graph_dataloaders(
                adata=dataset.adata, dataset_path=dataset.dataset_path, layer=layer, n_hops=3, backbone='ViT', model_path=model_path, batch_size=256, shuffle=False,
                hex_geometry=dataset.hex_geometry, patch_size=dataset.patch_size, patch_scale=dataset.patch_scale)
            
    # If prepare datasets and get graphs are false then only process the single dataset specified by the config arg
    else: 
        # Load the config file
        with open(args.config, 'r') as f:
            config = json.load(f)

        # Define param dict
        param_dict = {
            'cell_min_counts':       config['cell_min_counts'],
            'cell_max_counts':       config['cell_max_counts'],
            'gene_min_counts':       config['gene_min_counts'],
            'gene_max_counts':       config['gene_max_counts'],
            'min_exp_frac':          config['min_exp_frac'],
            'min_glob_exp_frac':     config['min_glob_exp_frac'],
            'real_data_percentage':  config['real_data_percentage'],
            'top_moran_genes':       config['top_moran_genes'],
            'wildcard_genes':        config['wildcard_genes'],
            'combat_key':            config['combat_key'],
            'random_samples':        config['random_samples'],
            'plotting_slides':       config['plotting_slides'],
            'plotting_genes':        config['plotting_genes'],
        }

        # Process the dataset and store it as adata
        test_dataset = SpatialDataset(
            dataset =       config['dataset'], 
            param_dict =    param_dict, 
            patch_scale =   config['patch_scale'], 
            patch_size =    config['patch_size'], 
            force_compute = config['force_compute']
        )
        

    