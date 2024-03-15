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
from . import im_encoder
from . import metrics
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

# Import all reader classes
from .readers.AbaloReader import AbaloReader
from .readers.BatiukReader import BatiukReader
from .readers.EricksonReader import EricksonReader
from .readers.FanReader import FanReader
from .readers.MirzazadehReader import MirzazadehReader
from .readers.ParigiReader import ParigiReader
from .readers.VicariReader import VicariReader
from .readers.VillacampaReader import VillacampaReader
from .readers.VisiumReader import VisiumReader
from .readers.HudsonReader import HudsonReader

# Remove the max limit of pixels in a figure
Image.MAX_IMAGE_PIXELS = None

# Get the path of the spared database
SPARED_PATH = pathlib.Path(__file__).parent

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

    def get_slide_from_collection(self, collection: ad.AnnData,  slide: str) -> ad.AnnData:
        """
        This function receives a slide name and returns an adata object of the specified slide based on the collection of slides
        in collection.

        Args: 
            collection (ad.AnnData): AnnData object with all the slides.
            slide (str): Name of the slide to get from the collection. Must be in the column 'slide_id' of the obs dataframe of the collection.

        Returns:
            ad.AnnData: An anndata object with the specified slide.
        """

        # Get the slide from the collection
        slide_adata = collection[collection.obs['slide_id'] == slide].copy()
        # Modify the uns dictionary to include only the information of the slide
        slide_adata.uns['spatial'] = {slide: collection.uns['spatial'][slide]}

        # Return the slide
        return slide_adata

    def filter_dataset(self, adata: ad.AnnData) -> ad.AnnData:
        """
        This function takes a completely unfiltered and unprocessed (in raw counts) slide collection and filters it
        (both samples and genes) according to self.param_dict. A summary list of the steps is the following:

            1. Filter out observations with total_counts outside the range [cell_min_counts, cell_max_counts].
               This filters out low quality observations not suitable for analysis.
            2. Compute the exp_frac for each gene. This means that for each slide in the collection we compute
                the fraction of the observations that express each gene and then took the minimum across all the slides.
            3. Compute the glob_exp_frac for each gene. This is similar to the exp_frac but instead of computing for each
               slide and taking the minimum we compute it for the whole collection. Slides don't matter here.
            4. Filter out genes. This depends on the wildcard_genes parameter and the options are the following:
                
                a. 'None':
                    - Filter out genes that are not expressed in at least min_exp_frac of spots in each slide.
                    - Filter out genes that are not expressed in at least min_glob_exp_frac of cells in the whole collection.
                    - Filter out genes with counts outside the range [gene_min_counts, gene_max_counts]
                
                b. else:
                    - Read .txt file with wildcard_genes and leave only the genes that are in this file

            5. If there are cells with zero counts in all genes then remove them
            6. Compute quality control metrics

        Args:
            adata (ad.AnnData): An unfiltered and unprocessed (in raw counts) slide collection. Has the patches in obsm.

        Returns:
            ad.AnnData: The filtered adata collection. Patches have not been reshaped here.
        """

        ### Define auxiliary functions

        def get_exp_frac(adata: ad.AnnData) -> ad.AnnData:
            """
            This function computes the expression fraction for each gene in the dataset. Internally it gets the
            expression fraction for each slide and then takes the minimum across all the slides.
            """
            # Get the unique slide ids
            slide_ids = adata.obs['slide_id'].unique()

            # Define zeros matrix of shape (n_genes, n_slides)
            exp_frac = np.zeros((adata.n_vars, len(slide_ids)))

            # Iterate over the slide ids
            for i, slide_id in enumerate(slide_ids):
                # Get current slide adata
                slide_adata = adata[adata.obs['slide_id'] == slide_id, :]
                # Get current slide expression fraction
                curr_exp_frac = np.squeeze(np.asarray((slide_adata.X > 0).sum(axis=0) / slide_adata.n_obs))
                # Add current slide expression fraction to the matrix
                exp_frac[:, i] = curr_exp_frac
            
            # Compute the minimum expression fraction for each gene across all the slides
            min_exp_frac = np.min(exp_frac, axis=1)

            # Add the minimum expression fraction to the var dataframe of the slide collection
            adata.var['exp_frac'] = min_exp_frac

            # Return the adata
            return adata

        def get_glob_exp_frac(adata: ad.AnnData) -> ad.AnnData:
            """
            This function computes the global expression fraction for each gene in the dataset.

            Args:
                adata (ad.AnnData): An unfiltered and unprocessed (in raw counts) slide collection.

            Returns:
                ad.AnnData: The same slide collection with the glob_exp_frac added to the var dataframe.
            """
            # Get global expression fraction
            glob_exp_frac = np.squeeze(np.asarray((adata.X > 0).sum(axis=0) / adata.n_obs))

            # Add the global expression fraction to the var dataframe of the slide collection
            adata.var['glob_exp_frac'] = glob_exp_frac

            # Return the adata
            return adata


        # Start tracking time
        print('Starting data filtering...')
        start = time()

        # Get initial gene and observation numbers
        n_genes_init = adata.n_vars
        n_obs_init = adata.n_obs

        ### Filter out samples:

        # Find indexes of cells with total_counts outside the range [cell_min_counts, cell_max_counts]
        sample_counts = np.squeeze(np.asarray(adata.X.sum(axis=1)))
        bool_valid_samples = (sample_counts > self.param_dict['cell_min_counts']) & (sample_counts < self.param_dict['cell_max_counts'])
        valid_samples = adata.obs_names[bool_valid_samples]

        # Subset the adata to keep only the valid samples
        adata = adata[valid_samples, :].copy()

        ### Filter out genes:

        # Compute the min expression fraction for each gene across all the slides
        adata = get_exp_frac(adata)
        # Compute the global expression fraction for each gene
        adata = get_glob_exp_frac(adata)
        
        # If no wildcard genes are specified then filter genes based in min_exp_frac and total counts
        if self.param_dict['wildcard_genes'] == 'None':
            
            gene_counts = np.squeeze(np.asarray(adata.X.sum(axis=0)))
                        
            # Find indexes of genes with total_counts inside the range [gene_min_counts, gene_max_counts]
            bool_valid_gene_counts = (gene_counts > self.param_dict['gene_min_counts']) & (gene_counts < self.param_dict['gene_max_counts'])
            # Get the valid genes
            valid_genes = adata.var_names[bool_valid_gene_counts]
            
            # Subset the adata to keep only the valid genes
            adata = adata[:, valid_genes].copy()     
        
            # Filter by expression fractions - order by descending expression fraction
            df_exp = adata.var.copy().sort_values('exp_frac', ascending=False)
            # Calculate the mean glob_exp_frac of top expression fraction genes
            df_exp['Row'] = range(1, len(df_exp) + 1)
            df_exp['vol_real_data'] = df_exp['glob_exp_frac'].cumsum() / (df_exp['Row'])      
            df_exp = df_exp.drop(['Row'], axis=1)
            # Get the valid genes
            num_genes = np.where(df_exp['vol_real_data'] >= self.param_dict['real_data_percentage'])[0][-1]
            valid_genes = df_exp.iloc[:num_genes + 1]['gene_ids']
            # Subset the adata to keep only the valid genes
            adata = adata[:, valid_genes].copy()
        
        # If there are wildcard genes then read them and subset the dataset to just use them
        else:
            # Read valid wildcard genes
            genes = pd.read_csv(self.param_dict['wildcard_genes'], sep=" ", header=None, index_col=False)
            # Turn wildcard genes to pandas Index object
            valid_genes = pd.Index(genes.iloc[:, 0], name='')
            # Subset processed adata with wildcard genes
            adata = adata[:, valid_genes].copy()
        
        ### Remove cells with zero counts in all genes:

        # If there are cells with zero counts in all genes then remove them
        null_cells = adata.X.sum(axis=1) == 0
        if null_cells.sum() > 0:
            adata = adata[~null_cells].copy()
            print(f"Removed {null_cells.sum()} cells with zero counts in all selected genes")
        
        ### Compute quality control metrics:

        # As we have removed the majority of the genes, we recompute the quality metrics
        sc.pp.calculate_qc_metrics(adata, inplace=True, log1p=False, percent_top=None)

        # Print the number of genes and cells that survived the filtering
        print(f'Data filtering took {time() - start:.2f} seconds')
        print(f"Number of genes that passed the filtering:        {adata.n_vars} out of {n_genes_init} ({100*adata.n_vars/n_genes_init:.2f}%)")
        print(f"Number of observations that passed the filtering: {adata.n_obs} out of {n_obs_init} ({100*adata.n_obs/n_obs_init:.2f}%)")

        return adata

    # TODO: Test the complete processing pipeline against R implementation (GeoTcgaData package for TPM)
    # TODO: Update the docstring of this function
    def process_dataset(self, adata: ad.AnnData) -> ad.AnnData:
        """
        This function performs the complete processing pipeline for a dataset. It only computes over the expression values of the dataset
        (adata.X). The processing pipeline is the following:
 
            1. Normalize the data with tpm normalization (tpm layer)
            2. Transform the data with log1p (log1p layer)
            3. Denoise the data with the adaptive median filter (d_log1p layer)
            4. Compute moran I for each gene in each slide and average moranI across slides (add results to .var['d_log1p_moran'])
            5. Filter dataset to keep the top self.param_dict['top_moran_genes'] genes with highest moran I.
            6. Perform ComBat batch correction if specified by the 'combat_key' parameter (c_d_log1p layer)
            7. Compute the deltas from the mean for each gene (computed from log1p layer and c_d_log1p layer if batch correction was performed)
            8. Add a binary mask layer specifying valid observations for metric computation.


        Args:
            adata (ad.AnnData): The AnnData object to process. Must be already filtered.

        Returns:
            ad.Anndata: The processed AnnData object with all the layers and results added.
        """

        ### Define processing functions:

        def tpm_normalization(adata: ad.AnnData, from_layer: str, to_layer: str) -> ad.AnnData:
            """
            This function apply tpm normalization to an AnnData object. It also removes genes that are not fount in the gtf annotation file.
            The counts of the anndata are taken from the layer 'from_layer' and the results are stored in the layer 'to_layer'.
            Args:
                adata (ad.Anndata): The Anndata object to normalize.
                from_layer (str): The layer to take the counts from.
                to_layer (str): The layer to store the results of the normalization.
            Returns:
                ad.Anndata: The normalized Anndata object with TPM values in the .layers[to_layer] attribute.
            """
            
            # Get the number of genes before filtering
            initial_genes = adata.shape[1]

            # Automatically download the gtf annotation file if it is not already downloaded
            if not os.path.exists(os.path.join(SPARED_PATH, 'data', 'annotations', 'gencode.v43.basic.annotation.gtf.gz')):
                print('Automatically downloading gtf annotation file...')
                os.makedirs(os.path.join(SPARED_PATH, 'data', 'annotations'), exist_ok=True)
                wget.download(
                    'https://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_human/release_43/gencode.v43.basic.annotation.gtf.gz',
                    out = os.path.join(SPARED_PATH, 'data', 'annotations', 'gencode.v43.basic.annotation.gtf.gz'))

            # Define gtf path
            gtf_path = os.path.join(SPARED_PATH, 'data', 'annotations', 'gencode.v43.basic.annotation.gtf')

            # Unzip the data in annotations folder if it is not already unzipped
            if not os.path.exists(gtf_path):
                with gzip.open(os.path.join(SPARED_PATH, 'data', 'annotations', 'gencode.v43.basic.annotation.gtf.gz'), 'rb') as f_in:
                    with open(gtf_path, 'wb') as f_out:
                        shutil.copyfileobj(f_in, f_out)

            # Define gtf mouse path
            gtf_path_mouse = os.path.join(SPARED_PATH, 'data', 'annotations', 'gencode.vM33.basic.annotation.gtf')

            # Unzip the data in annotations folder if it is not already unzipped
            if not os.path.exists(gtf_path_mouse):            
                with gzip.open(os.path.join(SPARED_PATH, 'data', 'annotations', 'gencode.vM33.basic.annotation.gtf.gz'), 'rb') as f_in:
                    with open(gtf_path_mouse, 'wb') as f_out:
                        shutil.copyfileobj(f_in, f_out)

            # Obtain a txt with gene lengths
            gene_length_path = os.path.join(SPARED_PATH, 'data', 'annotations', 'gene_length.txt')
            if not os.path.exists(gene_length_path):
                command = f'python {os.path.join(SPARED_PATH, "gtftools.py")} -l {gene_length_path} {gtf_path}'
                command_list = command.split(' ')
                subprocess.call(command_list)   

            gene_length_path_mouse = os.path.join(SPARED_PATH, 'data', 'annotations', 'gene_length_mouse.txt')
            if not os.path.exists(gene_length_path_mouse):
                command = f'python {os.path.join(SPARED_PATH, "gtftools.py")} -l {gene_length_path_mouse} {gtf_path_mouse}'
                command_list = command.split(' ')
                subprocess.call(command_list) 

            # Upload the gene lengths
            if "mouse" in self.dataset.lower():
                glength_df = pd.read_csv(gene_length_path_mouse, delimiter='\t', usecols=['gene', 'merged'])
            else:
                glength_df = pd.read_csv(gene_length_path, delimiter='\t', usecols=['gene', 'merged'])

            # For the gene column, remove the version number
            glength_df['gene'] = glength_df['gene'].str.split('.').str[0]

            # Drop gene duplicates. NOTE: This only eliminates 40/60k genes so it is not a big deal
            glength_df = glength_df.drop_duplicates(subset=['gene'])

            # Find the genes that are in the gtf annotation file
            common_genes=list(set(adata.var_names)&set(glength_df["gene"]))

            # Subset both adata and glength_df to keep only the common genes
            adata = adata[:, common_genes].copy()
            glength_df = glength_df[glength_df["gene"].isin(common_genes)].copy()

            # Reindex the glength_df to genes
            glength_df = glength_df.set_index('gene')
            # Reindex glength_df to adata.var_names
            glength_df = glength_df.reindex(adata.var_names)
            # Assert indexes of adata.var and glength_df are the same
            assert (adata.var.index == glength_df.index).all()

            # Add gene lengths to adata.var
            adata.var['gene_length'] = glength_df['merged'].values

            # Divide each column of the counts matrix by the gene length. Save the result in layer "to_layer"
            adata.layers[to_layer] = adata.layers[from_layer] / adata.var['gene_length'].values.reshape(1, -1)
            # Make that each row sums to 1e6
            adata.layers[to_layer] = np.nan_to_num(adata.layers[to_layer] / (np.sum(adata.layers[to_layer], axis=1).reshape(-1, 1)/1e6))
            # Pass layer to np.array
            adata.layers[to_layer] = np.array(adata.layers[to_layer])

            # Print the number of genes that were not found in the gtf annotation file
            failed_genes = initial_genes - adata.n_vars
            print(f'Number of genes not found in GTF file by TPM normalization: {initial_genes - adata.n_vars} out of {initial_genes} ({100*failed_genes/initial_genes:.2f}%) ({adata.n_vars} remaining)')

            # Return the transformed AnnData object
            return adata

        def log1p_transformation(adata: ad.AnnData, from_layer: str, to_layer: str) -> ad.AnnData:
            """
            Simple wrapper around sc.pp.log1p to transform data from adata.layers[from_layer] with log1p (base 2)
            and save it in adata.layers[to_layer].

            Args:
                adata (ad.AnnData): The AnnData object to transform.
                from_layer (str): The layer to take the data from.
                to_layer (str): The layer to store the results of the transformation.

            Returns:
                ad.AnnData: The transformed AnnData object with log1p transformed data in adata.layers[to_layer].
            """

            # Transform the data with log1p
            transformed_adata = sc.pp.log1p(adata, base= 2.0, layer=from_layer, copy=True)

            # Add the log1p transformed data to adata.layers[to_layer]
            adata.layers[to_layer] = transformed_adata.layers[from_layer]

            # Return the transformed AnnData object
            return adata

        def clean_noise(collection: ad.AnnData, from_layer: str, to_layer: str, n_hops: int, hex_geometry: bool) -> ad.AnnData:
            """
            This wrapper function that computes the adaptive median filter for all the slides in the collection and then concatenates the results
            into another collection. Details of the adaptive median filter can be found in the adaptive_median_filter_peper function.

            Args:
                collection (ad.AnnData): The AnnData collection to process. Contains all the slides.
                from_layer (str): The layer to compute the adaptive median filter from. Where to clean the noise from.
                to_layer (str): The layer to store the results of the adaptive median filter. Where to store the cleaned data.
                n_hops (int): The maximum number of concentric rings in the graph to take into account to compute the median. Analogous to the max window size.
                hex_geometry (bool): Whether the graph is hexagonal or not. If True, then the graph is hexagonal. If False, then the graph is a grid. Only
                                     used to compute the spatial neighbors and only true for visium datasets.

            Returns:
                ad.AnnData: The processed AnnData collection with the results of the adaptive median filter stored in the layer 'to_layer'.
            """
            
            ### Define function to get spatial neighbors in an AnnData object
            def get_spatial_neighbors(adata: ad.AnnData, n_hops: int, hex_geometry: bool) -> dict:
                """
                This function computes a neighbors dictionary for an AnnData object. The neighbors are computed according to topological distances over
                a graph defined by the hex_geometry connectivity. The neighbors dictionary is a dictionary where the keys are the indexes of the observations
                and the values are lists of the indexes of the neighbors of each observation. The neighbors include the observation itself and are found
                inside a n_hops neighborhood of the observation.

                Args:
                    adata (ad.AnnData): The AnnData object to process. Importantly it is only from a single slide. Can not be a collection of slides.
                    n_hops (int): The size of the neighborhood to take into account to compute the neighbors.
                    hex_geometry (bool): Whether the graph is hexagonal or not. If True, then the graph is hexagonal. If False, then the graph is a grid. Only
                                         used to compute the spatial neighbors and only true for visium datasets.

                Returns:
                    dict: The neighbors dictionary. The keys are the indexes of the observations and the values are lists of the indexes of the neighbors of each observation.
                """
                # Compute spatial_neighbors
                if hex_geometry:
                    sq.gr.spatial_neighbors(adata, coord_type='generic', n_neighs=6) # Hexagonal visium case
                else:
                    sq.gr.spatial_neighbors(adata, coord_type='grid', n_neighs=8) # Grid STNet dataset case

                # Get the adjacency matrix
                adj_matrix = adata.obsp['spatial_connectivities']

                # Define power matrix
                power_matrix = adj_matrix.copy()
                # Define the output matrix
                output_matrix = adj_matrix.copy()

                # Iterate through the hops
                for i in range(n_hops-1):
                    # Compute the next hop
                    power_matrix = power_matrix * adj_matrix
                    # Add the next hop to the output matrix
                    output_matrix = output_matrix + power_matrix

                # Zero out the diagonal
                output_matrix.setdiag(0)
                # Threshold the matrix to 0 and 1
                output_matrix = output_matrix.astype(bool).astype(int)

                # Define neighbors dict
                neighbors_dict_index = {}

                # Iterate through the rows of the output matrix
                for i in range(output_matrix.shape[0]):
                    # Get the non-zero elements of the row
                    non_zero_elements = output_matrix[i].nonzero()[1]
                    # Add the neighbors to the neighbors dicts. NOTE: the first index is the query obs
                    neighbors_dict_index[i] = [i] + list(non_zero_elements)
                
                # Return the neighbors dict
                return neighbors_dict_index

            ### Define cleaning function for single slide:
            def adaptive_median_filter_pepper(adata: ad.AnnData, from_layer: str, to_layer: str, n_hops: int, hex_geometry: bool) -> ad.AnnData:
                """
                This function computes the adaptive median filter for pairs (obs, gene) with a zero value (peper noise) in the layer 'from_layer' and
                stores the result in the layer 'to_layer'. The max window size is a neighborhood of n_hops defined by the conectivity hex_geometry
                inputed by parameter. This means the number of concentric rings in a graph to take into account to compute the median.

                Args:
                    adata (ad.AnnData): The AnnData object to process. Importantly it is only from a single slide. Can not be a collection of slides.
                    from_layer (str): The layer to compute the adaptive median filter from.
                    to_layer (str): The layer to store the results of the adaptive median filter.
                    n_hops (int): The maximum number of concentric rings in the graph to take into account to compute the median. Analogous to the max window size.
                    hex_geometry (bool): Whether the graph is hexagonal or not. If True, then the graph is hexagonal. If False, then the graph is a grid. Only
                                        used to compute the spatial neighbors and only true for visium datasets.

                Returns:
                    ad.AnnData: The AnnData object with the results of the adaptive median filter stored in the layer 'to_layer'.
                """
                # Define original expression matrix
                original_exp = adata.layers[from_layer]    

                medians = np.zeros((adata.n_obs, n_hops, adata.n_vars))

                # Iterate over the hops
                for i in range(1, n_hops+1):
                    
                    # Get dictionary of neighbors for a given number of hops
                    curr_neighbors_dict = get_spatial_neighbors(adata, i, hex_geometry)

                    # Iterate over observations
                    for j in range(adata.n_obs):
                        # Get the list of indexes of the neighbors of the j'th observation
                        neighbors_idx = curr_neighbors_dict[j]
                        # Get the expression matrix of the neighbors
                        neighbor_exp = original_exp[neighbors_idx, :]
                        # Get the median of the expression matrix
                        median = np.median(neighbor_exp, axis=0)

                        # Store the median in the medians matrix
                        medians[j, i-1, :] = median
                
                # Also robustly compute the median of the non-zero values for each gene
                general_medians = np.apply_along_axis(lambda v: np.median(v[np.nonzero(v)]), 0, original_exp)
                general_medians[np.isnan(general_medians)] = 0.0 # Correct for possible nans

                # Define corrected expression matrix
                corrected_exp = np.zeros_like(original_exp)

                ### Now that all the possible medians are computed. We code for each observation:
                
                # Note: i indexes over observations, j indexes over genes
                for i in range(adata.n_obs):
                    for j in range(adata.n_vars):
                        
                        # Get real expression value
                        z_xy = original_exp[i, j]

                        # Only apply adaptive median filter if real expression is zero
                        if z_xy != 0:
                            corrected_exp[i,j] = z_xy
                            continue
                        
                        else:

                            # Definie initial stage and window size
                            current_stage = 'A'
                            k = 0

                            while True:

                                # Stage A:
                                if current_stage == 'A':
                                    
                                    # Get median value
                                    z_med = medians[i, k, j]

                                    # If median is not zero then go to stage B
                                    if z_med != 0:
                                        current_stage = 'B'
                                        continue
                                    # If median is zero, then increase window and repeat stage A
                                    else:
                                        k += 1
                                        if k < n_hops:
                                            current_stage = 'A'
                                            continue
                                        # If we have the biggest window size, then return the median
                                        else:
                                            # NOTE: Big modification to the median filter here. Be careful
                                            corrected_exp[i,j] = general_medians[j]
                                            break


                                # Stage B:
                                elif current_stage == 'B':
                                    
                                    # Get window median
                                    z_med = medians[i, k, j]

                                    # If real expression is not peper then return it
                                    if z_xy != 0:
                                        corrected_exp[i,j] = z_xy
                                        break
                                    # If real expression is peper, then return the median
                                    else:
                                        corrected_exp[i,j] = z_med
                                        break

                # Add corrected expression to adata
                adata.layers[to_layer] = corrected_exp

                return adata

            # Print message
            print('Applying adaptive median filter to collection...')

            # Get the unique slides
            slides = np.unique(collection.obs['slide_id'])

            # Define the corrected adata list
            corrected_adata_list = []

            # Iterate over the slides
            for slide in tqdm(slides):
                # Get the adata of the slide
                adata = collection[collection.obs['slide_id'] == slide].copy()
                # Apply adaptive median filter
                adata = adaptive_median_filter_pepper(adata, from_layer, to_layer, n_hops, hex_geometry)
                # Append to the corrected adata list
                corrected_adata_list.append(adata)
            
            # Concatenate the corrected adata list
            corrected_collection = ad.concat(corrected_adata_list, join='inner', merge='same')
            # Restore the uns attribute
            corrected_collection.uns = collection.uns

            return corrected_collection

        def combat_transformation(adata: ad.AnnData, batch_key: str, from_layer: str, to_layer: str) -> ad.AnnData:
            """
            Batch correction using pycombat. The batches are defined by the batch_key column in adata.obs. The input data for
            the batch correction is adata.layers[from_layer] and the output is stored in adata.layers[to_layer].

            Args:
                adata (ad.AnnData): The AnnData object to transform. Must have log1p transformed data in adata.layers[from_layer].
                batch_key (str): The column in adata.obs that defines the batches.
                from_layer (str): The layer to take the data from.
                to_layer (str): The layer to store the results of the transformation.

            Returns:
                ad.AnnData: The transformed AnnData object with batch corrected data in adata.layers[to_layer].
            """
            # Get expression matrix dataframe
            df = adata.to_df(layer = from_layer).T
            batch_list = adata.obs[batch_key].values.tolist()

            # Apply pycombat batch correction
            corrected_df = pycombat(df, batch_list, par_prior=True)

            # Assign batch corrected expression to .layers[to_layer] attribute
            adata.layers[to_layer] = corrected_df.T

            return adata
        
        def get_deltas(adata: ad.AnnData, from_layer: str, to_layer: str) -> ad.AnnData:
            """
            Compute the deviations from the mean expression of each gene in adata.layers[from_layer] and save them
            in adata.layers[to_layer]. Also add the mean expression of each gene to adata.var[f'{from_layer}_avg_exp'].

            Args:
                adata (ad.AnnData): The AnnData object to update. Must have expression values in adata.layers[from_layer].
                from_layer (str): The layer to take the data from.
                to_layer (str): The layer to store the results of the transformation.

            Returns:
                ad.AnnData: The updated AnnData object with the deltas and mean expression.
            """

            # Get the expression matrix of both train and global data
            glob_expression = adata.to_df(layer=from_layer)
            train_expression = adata[adata.obs['split'] == 'train'].to_df(layer=from_layer)

            # Define scaler
            scaler = StandardScaler(with_mean=True, with_std=False)

            # Fit the scaler to the train data
            scaler = scaler.fit(train_expression)
            
            # Get the centered expression matrix of the global data
            centered_expression = scaler.transform(glob_expression)

            # Add the deltas to adata.layers[to_layer]	
            adata.layers[to_layer] = centered_expression

            # Add the mean expression to adata.var[f'{from_layer}_avg_exp']
            adata.var[f'{from_layer}_avg_exp'] = scaler.mean_

            # Return the updated AnnData object
            return adata

        def compute_moran(adata: ad.AnnData, hex_geometry: bool, from_layer: str) -> ad.AnnData:
            """
            This function cycles over each slide in the adata object and computes the Moran's I for each gene.
            After that, it averages the Moran's I for each gene across all slides and saves it in adata.var[f'{from_layer}_moran'].
            The input data for the Moran's I computation is adata.layers[from_layer].

            Args:
                adata (ad.AnnData): The AnnData object to update. Must have expression values in adata.layers[from_layer].
                from_layer (str): The key in adata.layers with the values used to compute Moran's I.
                hex_geometry (bool): Whether the data is hexagonal or not. This is used to compute the spatial neighbors before computing Moran's I.

            Returns:
                ad.AnnData: The updated AnnData object with the average Moran's I for each gene in adata.var[f'{from_layer}_moran'].
            """
            print(f'Computing Moran\'s I for each gene over each slide using data of layer {from_layer}...')

            # Get the unique slide_ids
            slide_ids = adata.obs['slide_id'].unique()

            # Create a dataframe to store the Moran's I for each slide
            moran_df = pd.DataFrame(index = adata.var.index, columns=slide_ids)

            # Cycle over each slide
            for slide in slide_ids:
                # Get the annData for the current slide
                slide_adata = self.get_slide_from_collection(adata, slide)
                # Compute spatial_neighbors
                if hex_geometry:
                    # Hexagonal visium case
                    sq.gr.spatial_neighbors(slide_adata, coord_type='generic', n_neighs=6)
                else:
                    # Grid STNet dataset case
                    sq.gr.spatial_neighbors(slide_adata, coord_type='grid', n_neighs=8)
                # Compute Moran's I
                sq.gr.spatial_autocorr(
                    slide_adata,
                    mode="moran",
                    layer=from_layer,
                    genes=slide_adata.var_names,
                    n_perms=1000,
                    n_jobs=-1,
                    seed=42
                )

                # Get moran I
                moranI = slide_adata.uns['moranI']['I']
                # Reindex moranI to match the order of the genes in the adata object
                moranI = moranI.reindex(adata.var.index)

                # Add the Moran's I to the dataframe
                moran_df[slide] = moranI

            # Compute the average Moran's I for each gene
            adata.var[f'{from_layer}_moran'] = moran_df.mean(axis=1)

            # Return the updated AnnData object
            return adata

        def filter_by_moran(adata: ad.AnnData, n_keep: int, from_layer: str) -> ad.AnnData:
            """
            This function filters the genes in adata.var by the Moran's I. It keeps the n_keep genes with the highest Moran's I.
            The Moran's I values will be selected from adata.var[f'{from_layer}_moran'].

            Args:
                adata (ad.AnnData): The AnnData object to update. Must have adata.var[f'{from_layer}_moran'].
                n_keep (int): The number of genes to keep.
                from_layer (str): Layer for which the Moran's I was computed the key in adata.var is f'{from_layer}_moran'.

            Returns:
                ad.AnnData: The updated AnnData object with the filtered genes.
            """

            # Assert that the number of genes is at least n_keep
            assert adata.n_vars >= n_keep, f'The number of genes in the AnnData object is {adata.n_vars}, which is less than n_keep ({n_keep}).'

            # Select amount of top genes depending on the available amount if n_keep is not specified
            if n_keep <= 0:
                n_keep = round(adata.n_vars * 0.25, 0)
                if np.abs(n_keep - 128) > np.abs(n_keep - 32):
                    n_keep = 32
                else:
                    n_keep = 128

            print(f"Filtering genes by Moran's I. Keeping top {n_keep} genes.")
            
            # Sort the genes by Moran's I
            sorted_genes = adata.var.sort_values(by=f'{from_layer}_moran', ascending=False).index

            # Get genes to keep list
            genes_to_keep = list(sorted_genes[:n_keep])

            # Filter the genes andata object
            adata = adata[:, genes_to_keep]

            # Return the updated AnnData object
            return adata


        ### Now compute all the processing steps
        # NOTE: The d prefix stands for denoised
        # NOTE: The c prefix stands for combat

        # Start the timer and print the start message
        start = time()
        print('Starting data processing...')

        # First add raw counts to adata.layers['counts']
        adata.layers['counts'] = adata.X.toarray()
        
        # Make TPM normalization
        adata = tpm_normalization(adata, from_layer='counts', to_layer='tpm')

        # Transform the data with log1p (base 2.0)
        adata = log1p_transformation(adata, from_layer='tpm', to_layer='log1p')

        # Denoise the data with pepper noise
        adata = clean_noise(adata, from_layer='log1p', to_layer='d_log1p', n_hops=4, hex_geometry=self.hex_geometry)

        # Compute average moran for each gene in the layer d_log1p 
        adata = compute_moran(adata, hex_geometry=self.hex_geometry, from_layer='d_log1p')

        # Filter genes by Moran's I
        adata = filter_by_moran(adata, n_keep=self.param_dict['top_moran_genes'], from_layer='d_log1p')

        # If combat key is specified, apply batch correction
        if self.param_dict['combat_key'] != 'None':
            adata = combat_transformation(adata, batch_key=self.param_dict['combat_key'], from_layer='log1p', to_layer='c_log1p')
            adata = combat_transformation(adata, batch_key=self.param_dict['combat_key'], from_layer='d_log1p', to_layer='c_d_log1p')

        # Compute deltas and mean expression for all log1p, d_log1p, c_log1p and c_d_log1p
        adata = get_deltas(adata, from_layer='log1p', to_layer='deltas')
        adata = get_deltas(adata, from_layer='d_log1p', to_layer='d_deltas')
        adata = get_deltas(adata, from_layer='c_log1p', to_layer='c_deltas')
        adata = get_deltas(adata, from_layer='c_d_log1p', to_layer='c_d_deltas')

        # Add a binary mask layer specifying valid observations for metric computation
        adata.layers['mask'] = adata.layers['tpm'] != 0
        # Print with the percentage of the dataset that was replaced
        print('Percentage of imputed observations with median filter: {:5.3f}%'.format(100 * (~adata.layers["mask"]).sum() / (adata.n_vars*adata.n_obs)))

        # Print the number of cells and genes in the dataset
        print(f'Processing of the data took {time() - start:.2f} seconds')
        print(f'The processed dataset looks like this:')
        print(adata)
        
        return adata

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
            collection_filtered = self.filter_dataset(collection_raw)
            collection_processed = self.process_dataset(collection_filtered)

            # Save the processed data
            os.makedirs(self.dataset_path, exist_ok=True)
            collection_raw.write(os.path.join(self.dataset_path, f'adata_raw.h5ad'))
            collection_processed.write(os.path.join(self.dataset_path, f'adata.h5ad'))

            # QC plotting
            self.plot_tests(collection_processed, collection_raw)
            # Copy figures folder into public database
            os.makedirs(os.path.join(SPARED_PATH, 'PublicDatabase', self.dataset), exist_ok=True)
            if os.path.exists(os.path.join(SPARED_PATH, 'PublicDatabase', self.dataset, 'qc_plots')):
                shutil.rmtree(os.path.join(SPARED_PATH, 'PublicDatabase', self.dataset, 'qc_plots'))
            shutil.copytree(os.path.join(self.dataset_path, 'qc_plots'), os.path.join(SPARED_PATH, 'PublicDatabase', self.dataset, 'qc_plots'), dirs_exist_ok=True)
            
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
                self.plot_tests(collection_processed, collection_raw)
                # Copy figures folder into public database
                os.makedirs(os.path.join(SPARED_PATH, 'PublicDatabase', self.dataset), exist_ok=True)
                if os.path.exists(os.path.join(SPARED_PATH, 'PublicDatabase', self.dataset, 'qc_plots')):
                    shutil.rmtree(os.path.join(SPARED_PATH, 'PublicDatabase', self.dataset, 'qc_plots'))
                shutil.copytree(os.path.join(self.dataset_path, 'qc_plots'), os.path.join(SPARED_PATH, 'PublicDatabase', self.dataset, 'qc_plots'), dirs_exist_ok=True)
                
                # Create README for dataset
                if not os.path.exists(os.path.join(SPARED_PATH, 'PublicDatabase', self.dataset, 'README.md')):
                    shutil.copy(os.path.join(SPARED_PATH, 'PublicDatabase', 'README_template.md'), os.path.join(SPARED_PATH, 'PublicDatabase', self.dataset, 'README.md'))

        return collection_processed
    
    # TODO: Update documentation
    def plot_tests(self, processed_adata: ad.AnnData, raw_adata: ad.AnnData)->None:
        """
        This function calls all the plotting functions in the class to create 6 quality control plots to check if the processing step of
        the dataset is performed correctly. The results are saved in dataset_logs folder and indexed by date and time. A dictionary
        in json format with all the dataset parameters is saved in the same log folder for reproducibility. Finally, a txt with the names of the
        genes used in processed adata is also saved in the folder.
        """

        ### Define function to get an adata list of plotting slides
        
        def refine_plotting_slides_str(self, collection: ad.AnnData, slide_list: str) -> str:
            """
            This function refines the plotting slides string to assure all slides are on the dataset. It works in the following way:
            1. If all slides are in the dataset it does nothing and returns the same slide_list parameter.
            2. If any slide is missing in the dataset of slide_list=='None' then it does one of 2 things:
                a. If the dataset has 4 or less slides all slides are set as plotting slides
                b. If the dataset has more than 4 slides it iterates over splits (train/val/test) and choses a single slide at a time without replacement. 

            Args:
                self(SpatialDataset): Dataset object.
                collection (ad.AnnData): Processed and filtered data ready to use by the model.
                slide_list (str): String with a list of slides separated by commas.

            Returns:
                str: Refined version of the slide_list string.
            """
            # Get bool value indicating if all slides in the string are on the dataset
            plot_slides_in_dataset = all([sl in collection.obs.slide_id.unique() for sl in slide_list.split(',')])

            # Decide if a refinement must be done
            if (not plot_slides_in_dataset) or (slide_list=='None'):
                
                # Check if there are less than 4 slides. If so, all are plotting slides
                if len(collection.obs.slide_id.unique()) <= 4:
                    slide_list = ','.join(collection.obs.slide_id.unique())
                    print(f'Plotting slides were None or missing in the dataset. And there are 4 or less slides. Setting all slides as plotting slides: {slide_list}')
                
                # If more than 4 slides, iterate over splits and chose randomly without replacement
                else:
                    plotting_slide_list = []
                    # Get list of unique splits
                    split_list = collection.obs.split.unique()
                    # Get a copy of the dictionary of splits to slides
                    split2slide_list = self.split_names.copy()
                    
                    # Set counter to 0
                    count = 0

                    # Iterate until we get 4 slides
                    while len(plotting_slide_list) < 4:
                        
                        # Get current split ant current slide list
                        curr_split = split_list[count % len(split_list)]
                        curr_slide_list = split2slide_list[curr_split]

                        # If the current slide list has slides choose one and delete it from the list
                        if len(curr_slide_list) > 0:
                            curr_slide = np.random.choice(curr_slide_list, 1)[0]
                            plotting_slide_list.append(curr_slide)
                            split2slide_list[curr_split].remove(curr_slide)
                        
                        # Update counter
                        count+=1
                    
                    # Update dataset parameter
                    slide_list = ','.join(plotting_slide_list)
                    print(f'Plotting slides were None or missing in the dataset. And there are more than 4 slides. Setting slides internally from all splits: {slide_list}')
            
            return slide_list

        def get_plotting_slides_adata(self, collection: ad.AnnData, slide_list: str) -> list:
            """
            This function receives a string with a list of slides separated by commas and returns a list of anndata objects with
            the specified slides taken from the collection parameter. 

            Args:
                self(SpatialDataset): Dataset object.
                collection (ad.AnnData): Processed and filtered data ready to use by the model.
                slide_list (str): String with a list of slides separated by commas.

            Returns:
                list: List of anndata objects with the specified slides.
            """

            # Get the slides from the collection
            s_adata_list = [self.get_slide_from_collection(collection,  slide) for slide in slide_list.split(',')]

            # Return the slides
            return s_adata_list
    
        ### Define all plotting functions

        def plot_all_slides(self, processed_adata: ad.AnnData, path: str) -> None:
            """
            This function takes a slide collection and plot all the whole slide images in a square aspect ratio.

            Args:
                self(SpatialDataset): Dataset object.
                processed_adata (ad.AnnData): Processed and filtered data ready to use by the model.
                path (str): Path to save the plot.
            """

            # Get unique slide ids
            unique_ids = sorted(processed_adata.obs['slide_id'].unique())

            # Get number of rows and columns for the number of slides
            n = int(np.ceil(np.sqrt(len(unique_ids))))
            m = int(np.ceil(float(len(unique_ids))/float(n)))

            # Define figure
            fig, ax = plt.subplots(nrows=m, ncols=n)
            fig.set_size_inches(7, (7*m/n)+1)
            
            # Flatten axes list
            ax = ax.flatten()

            for i, curr_id in enumerate(unique_ids):
                curr_img = processed_adata.uns['spatial'][curr_id]['images']['lowres']
                ax[i].imshow(curr_img)
                ax[i].set_title(curr_id, fontsize='large')

            # Remove axis for all the figure
            [axis.axis('off') for axis in ax]
            fig.suptitle(f'All Histology Images from {self.dataset}', fontsize='x-large')
            fig.tight_layout()

            fig.savefig(path, dpi=300)
            plt.close()

        def plot_exp_frac(self, raw_adata: ad.AnnData, path: str) -> None:
            """
            This function plots a heatmap of the expression fraction and global expression fraction for the complete collection of slides.

            Args:
                raw_adata (ad.AnnData): An unfiltered and unprocessed (in raw counts) slide collection.
                path (str): Path to save the plot.
            """

            def get_exp_frac(adata: ad.AnnData) -> ad.AnnData:
                """
                This function computes the expression fraction for each gene in the dataset. Internally it gets the
                expression fraction for each slide and then takes the minimum across all the slides.
                """
                # Get the unique slide ids
                slide_ids = adata.obs['slide_id'].unique()

                # Define zeros matrix of shape (n_genes, n_slides)
                exp_frac = np.zeros((adata.n_vars, len(slide_ids)))

                # Iterate over the slide ids
                for i, slide_id in enumerate(slide_ids):
                    # Get current slide adata
                    slide_adata = adata[adata.obs['slide_id'] == slide_id, :]
                    # Get current slide expression fraction
                    curr_exp_frac = np.squeeze(np.asarray((slide_adata.X > 0).sum(axis=0) / slide_adata.n_obs))
                    # Add current slide expression fraction to the matrix
                    exp_frac[:, i] = curr_exp_frac
                
                # Compute the minimum expression fraction for each gene across all the slides
                min_exp_frac = np.min(exp_frac, axis=1)

                # Add the minimum expression fraction to the var dataframe of the slide collection
                adata.var['exp_frac'] = min_exp_frac

                # Return the adata
                return adata
            
            def get_glob_exp_frac(adata: ad.AnnData) -> ad.AnnData:
                """
                This function computes the global expression fraction for each gene in the dataset.

                Args:
                    adata (ad.AnnData): An unfiltered and unprocessed (in raw counts) slide collection.

                Returns:
                    ad.AnnData: The same slide collection with the glob_exp_frac added to the var dataframe.
                """
                # Get global expression fraction
                glob_exp_frac = np.squeeze(np.asarray((adata.X > 0).sum(axis=0) / adata.n_obs))

                # Add the global expression fraction to the var dataframe of the slide collection
                adata.var['glob_exp_frac'] = glob_exp_frac

                # Return the adata
                return adata
            
            # Find indexes of cells with total_counts outside the range [cell_min_counts, cell_max_counts]
            sample_counts = np.squeeze(np.asarray(raw_adata.X.sum(axis=1)))
            bool_valid_samples = (sample_counts > self.param_dict['cell_min_counts']) & (sample_counts < self.param_dict['cell_max_counts'])
            valid_samples = raw_adata.obs_names[bool_valid_samples]

            # Subset the raw_adata to keep only the valid samples
            raw_adata = raw_adata[valid_samples, :].copy()

            # Compute the min expression fraction for each gene across all the slides
            raw_adata = get_exp_frac(raw_adata)
            # Compute the global expression fraction for each gene
            raw_adata = get_glob_exp_frac(raw_adata)
            
            # Histogram matrix
            hist_mat, edge_exp_frac, edge_glob_exp_frac = np.histogram2d(raw_adata.var['exp_frac'], raw_adata.var['glob_exp_frac'], range=[[0,1],[0,1]], bins=20)
            # Define dataframe
            index_str = [f'{int(100*per)}%' for per in edge_exp_frac[1:]]
            col_str = [f'{int(100*per)}%' for per in edge_glob_exp_frac[1:]]

            hist_df = pd.DataFrame(hist_mat.astype(int), index=index_str, columns=col_str)
            # Plot params
            scale = 3
            fig_size = (50, 40)
            tit_size = 80
            lab_size = 40
            
            # Define colormap
            d_colors = ["white", "darkcyan"]
            cmap1 = LinearSegmentedColormap.from_list("mycmap", d_colors)

            # Plot global expression fraction dataframe
            plt.figure(figsize=fig_size)
            sns.set_theme(font_scale=scale)
            ax = sns.heatmap(hist_df, annot=True, linewidths=.5, fmt='g', cmap=cmap1, linecolor='k', norm=colors.LogNorm(vmin=0.9, vmax=10000))
            
            # Define figure styling
            plt.suptitle(f'Expression Fraction {self.dataset}', fontsize=tit_size)
            plt.yticks(rotation=0)
            plt.xticks(rotation=90)
            ax.tick_params(labelsize=lab_size)
            plt.xlabel("Global Expression Fraction", fontsize=tit_size)
            plt.ylabel("Expression Fraction", fontsize=tit_size)

            # Define color bar configs
            cbar = ax.collections[0].colorbar
            cbar.ax.tick_params(labelsize=lab_size)
            cbar.ax.set_ylabel('Number of Genes', fontsize=tit_size)
            plt.tight_layout()
            
            # Save figure
            plt.savefig(path, dpi=300)
            plt.close()
            # Return font scale to normal and matplotlib defaults to not mess the other figures
            sns.set_theme(font_scale=1.0)
            mpl.rcParams.update(mpl.rcParamsDefault)

        def plot_histograms(self, processed_adata: ad.AnnData, raw_adata: ad.AnnData, path: str) -> None:
            """
            This function plots a figure that analyses the effect of the filtering over the data.
            The first row corresponds to the raw data (which has patches and excludes constant genes) and the second row
            plots the filtered and processed data. Histograms of total:
                
                1. Counts per cell
                2. Cells with expression
                3. Total counts per gene
                4. Moran I statistics (only in processed data)
            
            are generated. The plot is saved in the specified path.
            Cell filtering histograms are in red, gene filtering histograms are in blue and autocorrelation filtering histograms are in green.

            Args:
                self (SpatialDataset): Dataset object.
                processed_adata (ad.AnnData): Processed and filtered data ready to use by the model.
                raw_adata (ad.AnnData): Loaded data from .h5ad file that is not filtered but has patch information.
                path (str): Path to save histogram plot.
            """

            ### Define function to get the expression fraction
            def get_exp_frac(adata: ad.AnnData) -> ad.AnnData:
                """
                This function computes the expression fraction for each gene in the dataset. Internally it gets the
                expression fraction for each slide and then takes the minimum across all the slides.
                """
                # Get the unique slide ids
                slide_ids = adata.obs['slide_id'].unique()

                # Define zeros matrix of shape (n_genes, n_slides)
                exp_frac = np.zeros((adata.n_vars, len(slide_ids)))

                # Iterate over the slide ids
                for i, slide_id in enumerate(slide_ids):
                    # Get current slide adata
                    slide_adata = adata[adata.obs['slide_id'] == slide_id, :]
                    # Get current slide expression fraction
                    curr_exp_frac = np.squeeze(np.asarray((slide_adata.X > 0).sum(axis=0) / slide_adata.n_obs))
                    # Add current slide expression fraction to the matrix
                    exp_frac[:, i] = curr_exp_frac
                
                # Compute the minimum expression fraction for each gene across all the slides
                min_exp_frac = np.min(exp_frac, axis=1)

                # Add the minimum expression fraction to the var dataframe of the slide collection
                adata.var['exp_frac'] = min_exp_frac

                # Return the adata
                return adata

            # Compute qc metrics for raw and processed data in order to have total counts updated
            sc.pp.calculate_qc_metrics(raw_adata, inplace=True, log1p=False, percent_top=None)
            sc.pp.calculate_qc_metrics(processed_adata, inplace=True, log1p=False, percent_top=None, layer='counts')
            
            # Compute the expression fraction of the raw_adata
            raw_adata = get_exp_frac(raw_adata)

            # Create figures
            fig, ax = plt.subplots(nrows=2, ncols=5)
            fig.set_size_inches(18.75, 5)

            bin_num = 50

            # Plot histogram of the number of counts that each cell has
            raw_adata.obs['total_counts'].hist(ax=ax[0,0], bins=bin_num, grid=False, color='k')
            processed_adata.obs['total_counts'].hist(ax=ax[1,0], bins=bin_num, grid=False, color='darkred')

            # Plot histogram of the expression fraction of each gene
            raw_adata.var['exp_frac'].plot(kind='hist', ax=ax[0,1], bins=bin_num, grid=False, color='k', logy=True)
            processed_adata.var['exp_frac'].plot(kind = 'hist', ax=ax[1,1], bins=bin_num, grid=False, color='darkcyan', logy=True)

            # Plot histogram of the number of cells that express a given gene
            raw_adata.var['n_cells_by_counts'].plot(kind='hist', ax=ax[0,2], bins=bin_num, grid=False, color='k', logy=True)
            processed_adata.var['n_cells_by_counts'].plot(kind = 'hist', ax=ax[1,2], bins=bin_num, grid=False, color='darkcyan', logy=True)
            
            # Plot histogram of the number of total counts per gene
            raw_adata.var['total_counts'].plot(kind='hist', ax=ax[0,3], bins=bin_num, grid=False, color='k', logy=True)
            processed_adata.var['total_counts'].plot(kind = 'hist', ax=ax[1,3], bins=bin_num, grid=False, color='darkcyan', logy=True)
            
            # Plot histogram of the MoranI statistic per gene
            # raw_adata.var['moranI'].plot(kind='hist', ax=ax[0,4], bins=bin_num, grid=False, color='k', logy=True)
            processed_adata.var['d_log1p_moran'].plot(kind = 'hist', ax=ax[1,4], bins=bin_num, grid=False, color='darkgreen', logy=True)

            # Lists to format axes
            tit_list = ['Raw: Total counts',        'Raw: Expression fraction',         'Raw: Cells with expression',       'Raw: Total gene counts',       'Raw: MoranI statistic',
                        'Processed: Total counts',  'Processed: Expression fraction',   'Processed: Cells with expression', 'Processed: Total gene counts', 'Processed: MoranI statistic']
            x_lab_list = ['Total counts', 'Expression fraction', 'Cells with expression', 'Total counts', 'MoranI statistic']*2
            y_lab_list = ['# of cells', '# of genes', '# of genes', '# of genes', '# of genes']*2

            # Format axes
            for i, axis in enumerate(ax.flatten()):
                # Not show moran in raw data because it has no sense to compute it
                if i == 4:
                    # Delete frame 
                    axis.axis('off')
                    continue
                axis.set_title(tit_list[i])
                axis.set_xlabel(x_lab_list[i])
                axis.set_ylabel(y_lab_list[i])
                axis.spines[['right', 'top']].set_visible(False)

            # Shared x axes between plots
            ax[1,0].sharex(ax[0,0])
            ax[1,1].sharex(ax[0,1])
            ax[1,2].sharex(ax[0,2])
            ax[1,3].sharex(ax[0,3])
            ax[1,4].sharex(ax[0,4])

            # Shared y axes between
            ax[1,0].sharey(ax[0,0])
            
            fig.tight_layout()
            fig.savefig(path, dpi=300)
            plt.close()

        def plot_random_patches(self, processed_adata: ad.AnnData, path: str) -> None:
            """
            This function gets 16 flat random patches (with the specified dims) from the processed adata objects. It
            reshapes them to a bidimensional form and shows them. The plot is saved to the specified path.

            Args:
                self (SpatialDataset): Dataset object.
                processed_adata (ad.AnnData): Processed and filtered data ready to use by the model.
                path (str): Path to save the image.
            """
            # Get the flat patches from the dataset
            flat_patches = processed_adata.obsm[f'patches_scale_{self.patch_scale}']
            # Reshape the patches for them to have image form
            patches = flat_patches.reshape((-1, self.patch_size, self.patch_size, 3))
            # Choose 16 random patches
            chosen = np.random.randint(low=0, high=patches.shape[0], size=16)
            # Get plotting patches
            plotting_patches = patches[chosen, :, :, :]

            # Declare image
            im, ax = plt.subplots(nrows=4, ncols=4)

            # Cycle over each random patch
            for i, ax in enumerate(ax.reshape(-1)):
                ax.imshow(plotting_patches[i, :, :, :])
                ax.axis('off')
            
            # Set figure formatting
            im.suptitle(f'Random patches from {self.dataset}')
            plt.tight_layout()
            im.savefig(path, dpi=300)
            plt.close()

        def visualize_moran_filtering(self, processed_adata: ad.AnnData, from_layer: str, path: str, top: bool = True) -> None:
            """
            This function visualizes the spatial expression of the 4 most and least auto-correlated genes in processed_adata.
            The title of each subplot shows the value of the moran I statistic for a given gene. The plot is saved to the specified
            path. This plot uses the slide list in string format in self.param_dict['plotting_slides']to plot these specific observations.
            If no list is provided (self.param_dict['plotting_slides']=='None'), 4 random slides are chosen. 

            Args:
                self (SpatialDataset): Dataset object.
                processed_adata (ad.AnnData): Processed and filtered data ready to use by the model.
                from_layer (str): Layer of the adata object to use for plotting.
                path (str): Path to save the generated image.
                top (bool, optional): If True, the top 4 most auto-correlated genes are visualized. If False, the top 4 least
                                    auto-correlated genes are visualized. Defaults to True.
            """
            
            plotting_key = from_layer

            # Get the slides to visualize in adata format
            s_adata_list = get_plotting_slides_adata(self, processed_adata, self.param_dict['plotting_slides'])

            # Get te top 4 most or least auto-correlated genes in processed data depending on the value of top
            # NOTE: The selection of genes is done in the complete collection of slides, not in the specified slides
            moran_key = 'd_log1p_moran'
            if top:
                selected_table = processed_adata.var.nlargest(4, columns=moran_key)
            else:
                selected_table = processed_adata.var.nsmallest(4, columns=moran_key)

            # Declare figure
            fig, ax = plt.subplots(nrows=4, ncols=len(s_adata_list))
            fig.set_size_inches(4 * len(s_adata_list) , 13)

            # Cycle over slides
            for i in range(len(selected_table)):

                # Get min and max of the selected gene in the slides
                gene_min = min([dat[:, selected_table.index[i]].layers[plotting_key].min() for dat in s_adata_list])
                gene_max = max([dat[:, selected_table.index[i]].layers[plotting_key].max() for dat in s_adata_list])

                # Define color normalization
                norm = matplotlib.colors.Normalize(vmin=gene_min, vmax=gene_max)

                for j in range(len(s_adata_list)):
                    
                    # Define bool to only plot the colorbar in the last column
                    cbar = True if j==(len(s_adata_list)-1) else False

                    # Plot selected genes in the specified slides
                    sq.pl.spatial_scatter(s_adata_list[j], color=[selected_table.index[i]], layer= plotting_key, ax=ax[i,j], cmap='jet', norm=norm, colorbar=cbar)
                    
                    # Set slide name
                    if i==0:
                        ax[i,j].set_title(f'{self.param_dict["plotting_slides"].split(",")[j]}', fontsize=15)
                    else:
                        ax[i,j].set_title('')
                    
                    # Set gene name and moran I value
                    if j==0:
                        ax[i,j].set_ylabel(f'{selected_table.index[i]}: $I = {selected_table[moran_key].iloc[i].round(3)}$', fontsize=13)
                    else:
                        ax[i,j].set_ylabel('')

            # Format figure
            for axis in ax.flatten():
                axis.set_xlabel('')
                # Turn off all spines
                axis.spines['top'].set_visible(False)
                axis.spines['right'].set_visible(False)
                axis.spines['bottom'].set_visible(False)
                axis.spines['left'].set_visible(False)

            # Define title
            tit_str = 'most (top)' if top else 'least (bottom)'

            fig.suptitle(f'Top 4 {tit_str} auto-correlated genes in processed data', fontsize=20)
            fig.tight_layout()
            # Save plot 
            fig.savefig(path, dpi=300)
            plt.close()

        def visualize_gene_expression(self, processed_adata: ad.AnnData, from_layer: str, path: str) -> None:
            """
            This function selects the genes specified in self.param_dict['plotting_genes'] and self.param_dict['plotting_slides']
            to plot gene expression for the specified genes in the specified slides. If either of them is 'None', then the method
            chooses randomly (4 genes or 4 slides in the stnet_dataset or 2 slides in visium datasets). The data is plotted from
            the .layers[from_layer] expression matrix

            Args:
                self (SpatialDataset): Dataset object.
                processed_adata (ad.AnnData): The processed adata with the filtered patient collection
                from_layer (str): The key to the layer of the data to plot.
                path (str): Path to save the image.
            """

            # Get the slides to visualize in adata format
            s_adata_list = get_plotting_slides_adata(self, processed_adata, self.param_dict['plotting_slides'])

            # Define gene list
            gene_list = self.param_dict['plotting_genes'].split(',')

            # Try to get the specified genes otherwise choose randomly
            try:
                gene_table = processed_adata[:, gene_list].var
            except:
                print('Could not find all the specified plotting genes, choosing randomly')
                gene_list = np.random.choice(processed_adata.var_names, size=4, replace=False)
                gene_table = processed_adata[:, gene_list].var
            
            # Declare figure
            fig, ax = plt.subplots(nrows=4, ncols=len(s_adata_list))
            fig.set_size_inches(4 * len(s_adata_list) , 13)

            # Cycle over slides
            for i in range(len(gene_table)):

                # Get min and max of the selected gene in the slides
                gene_min = min([dat[:, gene_table.index[i]].layers[from_layer].min() for dat in s_adata_list])
                gene_max = max([dat[:, gene_table.index[i]].layers[from_layer].max() for dat in s_adata_list])

                # Define color normalization
                norm = matplotlib.colors.Normalize(vmin=gene_min, vmax=gene_max)

                for j in range(len(s_adata_list)):

                    # Define bool to only plot the colorbar in the last column
                    cbar = True if j==(len(s_adata_list)-1) else False
                    
                    # Plot selected genes in the specified slides
                    sq.pl.spatial_scatter(s_adata_list[j], layer=from_layer, color=[gene_table.index[i]], ax=ax[i,j], cmap='jet', norm=norm, colorbar=cbar)
                    
                    # Set slide name
                    if i==0:
                        ax[i,j].set_title(f'{self.param_dict["plotting_slides"].split(",")[j]}', fontsize=15)
                    else:
                        ax[i,j].set_title('')
                    
                    # Set gene name with moran I value 
                    if j==0:
                        moran_key = 'd_log1p_moran'
                        ax[i,j].set_ylabel(f'{gene_table.index[i]}: $I = {gene_table[moran_key].iloc[i].round(3)}$', fontsize=13)
                    else:
                        ax[i,j].set_ylabel('')
            
            # Format figure
            for axis in ax.flatten():
                axis.set_xlabel('')
                # Turn off all spines
                axis.spines['top'].set_visible(False)
                axis.spines['right'].set_visible(False)
                axis.spines['bottom'].set_visible(False)
                axis.spines['left'].set_visible(False)
            
            fig.suptitle('Gene expression in processed data', fontsize=20)
            fig.tight_layout()
            # Save plot
            fig.savefig(path, dpi=300)
            plt.close()

        def plot_clusters(self, processed_adata: ad.AnnData, from_layer: str, path: str) -> None:
            """
            This function generates a plot that visualizes Leiden clusters spatially in the slides in self.param_dict['plotting_slides'].
            The slides can be specified in self.param_dict['plotting_slides'] or chosen randomly.
            
            It plots:
                1. The spatial distribution of the Leiden clusters in the slides.
                2. UMAP embeddings of each slide colored by Leiden clusters.
                3. General UMAP embedding of the complete dataset colored by Leiden clusters and the batch correction key.
                4. PCA embeddings of the complete dataset colored by the batch correction key.

            Args:
                self (SpatialDataset): Dataset object.
                processed_adata (ad.AnnData): Processed and filtered data ready to use by the model.
                from_layer (str): The key in adata.layers where the expression matrix is stored.
                path (str): Path to save the image.
            """

            ### Define function to get dimensionality reductions depending on the layer
            def compute_dim_red(adata: ad.AnnData, from_layer: str) -> ad.AnnData:
                """
                Simple wrapper around sc.pp.pca, sc.pp.neighbors, sc.tl.umap and sc.tl.leiden to compute the embeddings and cluster the data.
                Everything will be computed using the expression matrix stored in adata.layers[from_layer]. 

                Args:
                    adata (ad.AnnData): The AnnData object to transform. Must have expression values in adata.layers[from_layer].
                    from_layer (str): The key in adata.layers where the expression matrix is stored.

                Returns:
                    ad.AnnData: The transformed AnnData object with the embeddings and clusters.
                """
                
                # Start the timer
                # start = time()
                # print(f'Computing embeddings and clusters using data of layer {from_layer}...')
                
                # Set the key layer as the main expression matrix
                adata_copy = adata.copy()
                adata_copy.X = adata_copy.layers[from_layer]
                

                # Compute the embeddings and clusters
                sc.pp.pca(adata_copy, random_state=42)
                sc.pp.neighbors(adata_copy, random_state=42)
                sc.tl.umap(adata_copy, random_state=42)
                sc.tl.leiden(adata_copy, key_added="cluster", random_state=42)
                
                # Restore the original expression matrix as counts layer
                adata_copy.X = adata_copy.layers['counts']

                # Print the time it took to compute the embeddings and clusters
                # print(f'Embeddings and clusters computed in {time() - start:.2f} seconds')

                # Return the adapted AnnData object
                return adata_copy

            # Update the adata object with the embeddings and clusters
            updated_adata = compute_dim_red(processed_adata, from_layer)

            # Get the slides to visualize in adata format
            s_adata_list = get_plotting_slides_adata(self, updated_adata, self.param_dict['plotting_slides'])

            # Define dictionary from cluster to color
            clusters = updated_adata.obs['cluster'].unique()
            # Sort clusters
            clusters = np.sort([int(cl) for cl in clusters])
            clusters = [str(cl) for cl in clusters]
            # Define color palette
            colors = sns.color_palette('hls', len(clusters))
            palette = dict(zip(clusters, colors))
            gray_palette = dict(zip(clusters, ['gray']*len(clusters)))

            # Declare figure
            fig = plt.figure(layout="constrained")
            gs0 = fig.add_gridspec(1, 2)
            gs00 = gs0[0].subgridspec(4, 2)
            gs01 = gs0[1].subgridspec(3, 1)

            fig.set_size_inches(15,14)

            # Cycle over slides
            for i in range(len(s_adata_list)):
                
                curr_clusters = s_adata_list[i].obs['cluster'].unique()
                # Sort clusters
                curr_clusters = np.sort([int(cl) for cl in curr_clusters])
                curr_clusters = [str(cl) for cl in curr_clusters]
                # # Define color palette
                spatial_colors = matplotlib.colors.ListedColormap([palette[x] for x in curr_clusters])
                
                # Get ax for spatial plot and UMAP plot
                spatial_ax = fig.add_subplot(gs00[i, 0])
                umap_ax = fig.add_subplot(gs00[i, 1])

                # Plot cluster colors in spatial space
                sq.pl.spatial_scatter(s_adata_list[i], color=['cluster'], ax=spatial_ax, palette=spatial_colors)

                spatial_ax.get_legend().remove()
                spatial_ax.set_title('Spatial', fontsize=18)
                spatial_ax.set_ylabel(f'{self.param_dict["plotting_slides"].split(",")[i]}', fontsize=12)
                spatial_ax.set_xlabel('')
                # Turn off all spines
                spatial_ax.spines['top'].set_visible(False)
                spatial_ax.spines['right'].set_visible(False)
                spatial_ax.spines['bottom'].set_visible(False)
                spatial_ax.spines['left'].set_visible(False)
                

                # Plot cluster colors in UMAP space for slide and all collection
                sc.pl.umap(updated_adata, layer=from_layer, color=['cluster'], ax=umap_ax, frameon=False, palette=gray_palette, s=10, cmap=None, alpha=0.2)
                umap_ax.get_legend().remove()
                sc.pl.umap(s_adata_list[i], layer=from_layer, color=['cluster'], ax=umap_ax, frameon=False, palette=palette, s=10, cmap=None)
                umap_ax.get_legend().remove()
                umap_ax.set_title('UMAP', fontsize=18)
                
            # Get axes for leiden clusters, patient and cancer types
            leiden_ax = fig.add_subplot(gs01[0])
            patient_ax = fig.add_subplot(gs01[1])
            pca_ax = fig.add_subplot(gs01[2])

            # Plot leiden clusters in UMAP space
            sc.pl.umap(updated_adata, color=['cluster'], ax=leiden_ax, frameon=False, palette=palette, s=10, cmap=None)
            leiden_ax.get_legend().set_title('Leiden Clusters')
            leiden_ax.get_legend().get_title().set_fontsize(15)
            leiden_ax.set_title('UMAP & Leiden Clusters', fontsize=18)

            # Plot batch_key in UMAP space
            sc.pl.umap(updated_adata, color=[self.param_dict['combat_key']], ax=patient_ax, frameon=False, palette='tab20', s=10, cmap=None)
            patient_ax.get_legend().set_title(self.param_dict['combat_key'])
            patient_ax.get_legend().get_title().set_fontsize(15)
            patient_ax.set_title(f"UMAP & {self.param_dict['combat_key']}", fontsize=18)

            # Plot cancer types in UMAP space
            sc.pl.pca(updated_adata, color=[self.param_dict['combat_key']], ax=pca_ax, frameon=False, palette='tab20', s=10, cmap=None)
            pca_ax.get_legend().set_title(self.param_dict['combat_key'])
            pca_ax.get_legend().get_title().set_fontsize(15)
            pca_ax.set_title(f'PCA & {self.param_dict["combat_key"]}', fontsize=18)
            
            # Format figure and save
            fig.suptitle(f'Cluster visualization for {self.dataset} in layer {from_layer}', fontsize=25)
            # fig.tight_layout()
            fig.savefig(path, dpi=300)
            plt.close(fig)

        def plot_mean_std(self, processed_adata: ad.AnnData, raw_adata: ad.AnnData, path: str) -> None:
            """
            This function plots a scatter of mean and standard deviation of genes present in raw_adata (black) and all the layers with non-zero
            mean in processed_adata. It is used to see the effect of filtering and processing in the genes. The plot is saved to the specified path.

            Args:
                self (SpatialDataset): Dataset object.
                processed_adata (ad.AnnData): Processed and filtered data ready to use by the model.
                raw_adata (ad.AnnData): Data loaded data from .h5ad file that is not filtered but has patch information.
                path (str): Path to save the image.
            """
            # Copy raw data to auxiliary data
            aux_raw_adata = raw_adata.copy()

            # Normalize and log transform aux_raw_adata
            sc.pp.normalize_total(aux_raw_adata, inplace=True)
            sc.pp.log1p(aux_raw_adata)

            # Get means and stds from raw data
            raw_mean = aux_raw_adata.to_df().mean(axis=0)
            raw_std = aux_raw_adata.to_df().std(axis=0)

            # Define list of layers to plot
            layers = ['log1p', 'd_log1p', 'c_log1p', 'c_d_log1p']

            plt.figure()
            plt.scatter(raw_mean, raw_std, s=1, c='k', label='Raw data')
            for layer in layers:
                # Get means and stds from processed data
                pro_mean = processed_adata.to_df(layer=layer).mean(axis=0)
                pro_std = processed_adata.to_df(layer=layer).std(axis=0)
                plt.scatter(pro_mean, pro_std, s=1, label=f'{layer} data')
            plt.xlabel('Mean $Log(x+1)$')
            plt.ylabel('Std $Log(x+1)$')
            plt.legend(loc='best')
            plt.title(f'Mean Std plot {self.dataset}')
            plt.gca().spines[['right', 'top']].set_visible(False)

            plt.tight_layout()
            plt.savefig(path, dpi=300)
            plt.close()

        def plot_data_distribution_stats(self, processed_adata: ad.AnnData, path:str):
            """
            This function plots a pie chart and bar plots of the distribution of spots and slides in the dataset split.

            Args:
                self (SpatialDataset): Dataset object.
                processed_adata (ad.AnnData): Processed and filtered data ready to use by the model.
                path (str): Path to save the image.
            """
            patients = processed_adata.obs['patient'].unique()
            slides = processed_adata.obs['slide_id'].unique()
            quant_spots = [processed_adata[processed_adata.obs['split'] == split].shape[0] for split in ['train', 'val', 'test']]
            metadata = f"Patients: {len(patients)}\nSlides: {len(slides)}\nSpots: {processed_adata.shape[0]}"
            labels_pie = ['Train', 'Valid', 'Test']
            if quant_spots[-1] == 0:
                quant_spots.pop()
                labels_pie.pop()

            # Create figures
            fig, ax = plt.subplots(nrows=1, ncols=3)
            fig.set_size_inches(18.75, 6.5)
            
            # Format axes
            ax[0].pie(quant_spots, labels = labels_pie,
                    wedgeprops = {"linewidth": 1, "edgecolor": "white"}, 
                    autopct = lambda x: '{:.1f}%\n{:.0f}'.format(x, x*processed_adata.shape[0]/100), 
                    colors = sns.color_palette('Set2'),
                    textprops = {"fontsize": 15})

            spot_counts = processed_adata.obs.groupby(['patient', 'split'], observed=False)['unique_id'].nunique().unstack(fill_value=0)
            spot_counts = spot_counts.reindex(columns=['train','val','test'])
            slide_counts = processed_adata.obs.groupby(['patient', 'split'], observed=False)['slide_id'].nunique().unstack(fill_value=0)
            slide_counts = slide_counts.reindex(columns=['train','val','test'])

            spot_counts.plot.bar(stacked=True, rot=0, color=sns.color_palette('Set2'), ax=ax[1], legend=False, fontsize=13)
            slide_counts.plot.bar(stacked=True, rot=0, color=sns.color_palette('Set2'), ax=ax[2], legend=False, fontsize=13)

            ax[1].spines[['right', 'top']].set_visible(False)
            ax[2].spines[['right', 'top']].set_visible(False)

            ax[0].set_title('Spot distribution per split', fontsize = 17)
            ax[1].set_title('Spots per patient', fontsize = 17)
            ax[2].set_title('Slides per patient', fontsize = 17)

            ax[1].set_ylabel('# spots', fontsize = 15)
            ax[2].set_ylabel('# slides', fontsize = 15)

            ax[1].set_xlabel('Patient', fontsize = 15)
            ax[2].set_xlabel('Patient', fontsize = 15)

            fig.suptitle(f"Data distribution for {self.dataset}", fontsize = 20)
            plt.figtext(0.02, 0.08, metadata, fontsize=15, wrap=True, bbox ={'facecolor':'whitesmoke', 'alpha':0.3, 'pad':5})
            fig.tight_layout()
            fig.savefig(path, dpi=300)
            plt.close()

        def plot_mean_std_partitions(self, processed_adata: ad.AnnData, from_layer: str, path: str) -> None:
            """
            This function plots a scatter of mean and standard deviation of genes present in processed_adata drawing with a different color different data
            splits (train/val/test). This is all done for the specified layer in the from_layer parameter. This function is used to see how tractable is
            the task. The plot is saved to the specified path.

            Args:
                self (SpatialDataset): Dataset object.
                processed_adata (ad.AnnData): Processed and filtered data ready to use by the model.
                from_layer (str): The key in adata.layers where the expression matrix is stored.
                path (str): Path to save the image.
            """
            
            # Copy processed adata to avoid problems
            aux_processed_adata = processed_adata.copy() 

            plt.figure()
            for curr_split in aux_processed_adata.obs['split'].unique():
                # Get means and stds from processed data
                curr_mean = aux_processed_adata[aux_processed_adata.obs.split==curr_split, :].to_df(layer=from_layer).mean(axis=0)
                curr_std = aux_processed_adata[aux_processed_adata.obs.split==curr_split, :].to_df(layer=from_layer).std(axis=0)
                plt.scatter(curr_mean, curr_std, s=1, label=f'{curr_split} data')
            plt.xlabel('Mean $Log(x+1)$')
            plt.ylabel('Std $Log(x+1)$')
            plt.legend(loc='best')
            plt.title(f'Mean Std plot {self.dataset}')
            plt.gca().spines[['right', 'top']].set_visible(False)

            plt.tight_layout()
            plt.savefig(path, dpi=300)
            plt.close()

        print('Started quality control plotting')
        start = time()

        # Define directory path to save data
        save_path = os.path.join(self.dataset_path, 'qc_plots')
        os.makedirs(save_path, exist_ok=True)
        # Define interest layers
        relevant_layers = ['log1p', 'd_log1p', 'c_d_log1p']
        complete_layers = ['counts', 'tpm', 'log1p', 'd_log1p', 'c_d_log1p']

        # Assure that the plotting genes are in the data and if not, set random plotting genes
        if not all([gene in processed_adata.var_names for gene in self.param_dict['plotting_genes'].split(',')]):
            self.param_dict['plotting_genes'] = ','.join(np.random.choice(processed_adata.var_names, 4))
            print(f'Plotting genes not in data. Setting random plotting genes: {self.param_dict["plotting_genes"]}')
        
        # Refine plotting slides string to assure they are in the dataset
        self.param_dict['plotting_slides'] = refine_plotting_slides_str(self, processed_adata, self.param_dict['plotting_slides'])


        # Plot partitions mean vs std scatter
        print('Started partitions mean vs std scatter plotting')
        os.makedirs(os.path.join(save_path, 'mean_vs_std_partitions'), exist_ok=True)
        for lay in tqdm(relevant_layers):
            plot_mean_std_partitions(self, processed_adata, from_layer=lay, path=os.path.join(save_path, 'mean_vs_std_partitions', f'{lay}.png'))

        # Plot all slides in collection
        print('Started all slides plotting')
        plot_all_slides(self, processed_adata, os.path.join(save_path, 'all_slides.png'))

        # Make plot of filtering histograms
        print('Started filtering histograms plotting')
        plot_histograms(self, processed_adata, raw_adata, os.path.join(save_path, 'filtering_histograms.png'))

        # Make plot of random patches
        print('Started random patches plotting')
        plot_random_patches(self, processed_adata, os.path.join(save_path, 'random_patches.png'))

        # Create save paths fot top and bottom moran genes
        os.makedirs(os.path.join(save_path, 'top_moran_genes'), exist_ok=True)
        os.makedirs(os.path.join(save_path, 'bottom_moran_genes'), exist_ok=True)
        print('Started moran filtering plotting')
        # Plot moran filtering
        for lay in tqdm(relevant_layers):
            # Make plot of 4 most moran genes and 4 less moran genes (in the chosen slides)
            visualize_moran_filtering(self, processed_adata, from_layer=lay, path = os.path.join(save_path, 'top_moran_genes', f'{lay}.png'), top = True)
            visualize_moran_filtering(self, processed_adata, from_layer=lay, path = os.path.join(save_path, 'bottom_moran_genes', f'{lay}.png'), top = False)
        

        # Create save paths for cluster plots
        os.makedirs(os.path.join(save_path, 'cluster_plots'), exist_ok=True)
        print('Started cluster plotting')
        # Plot cluster graphs
        for lay in tqdm(relevant_layers):
            plot_clusters(self, processed_adata, from_layer=lay, path=os.path.join(save_path, 'cluster_plots', f'{lay}.png'))
        
        # Define expression layers
        os.makedirs(os.path.join(save_path, 'expression_plots'), exist_ok=True)
        print('Started gene expression plotting')
        # Plot of gene expression in the chosen slides for the 4 chosen genes
        for lay in tqdm(complete_layers):
            visualize_gene_expression(self, processed_adata, from_layer=lay, path=os.path.join(save_path,'expression_plots', f'{lay}.png'))

        # Make plot of mean vs std per gene must be programmed manually.
        print('Started mean vs std plotting')
        plot_mean_std(self, processed_adata, raw_adata, os.path.join(save_path, 'mean_std_scatter.png'))

        # Make plot of data distribution statistics.
        print('Started data distribution statistics plotting')
        plot_data_distribution_stats(self, processed_adata, os.path.join(save_path, 'splits_stats.png'))

        # Plot expression fraction 2D histogram
        print('Started expression fraction plotting')
        plot_exp_frac(self, raw_adata, os.path.join(save_path, 'exp_frac.png'))
        
        # Print the time that took to plot quality control
        end = time()
        print(f'Quality control plotting took {round(end-start, 2)}s')
        print(f'Images saved in {save_path}')

    def compute_patches_embeddings_and_predictions(self, backbone: str ='densenet', model_path:str="best_stnet.pt", preds: bool=True) -> None:
            
            # Define a cuda device if available
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            model = im_encoder.ImageEncoder(backbone=backbone, use_pretrained=True, latent_dim=self.adata.n_vars)

            if model_path != "None":
                saved_model = torch.load(model_path)
                # Check if state_dict is inside a nested dictionary
                if 'state_dict' in saved_model.keys():
                    saved_model = saved_model['state_dict']

                model.load_state_dict(saved_model)
            
            if backbone == 'resnet':
                weights = tmodels.ResNet18_Weights.DEFAULT
                if not preds:
                    model.encoder.fc = nn.Identity()
            elif backbone == 'resnet50':
                weights = tmodels.ResNet50_Weights.DEFAULT
                if not preds:
                    model.encoder.fc = nn.Identity()
            elif backbone == 'ConvNeXt':
                weights = tmodels.ConvNeXt_Tiny_Weights.DEFAULT
                if not preds:
                    model.encoder.classifier[2] = nn.Identity()
            elif backbone == 'EfficientNetV2':
                weights = tmodels.EfficientNet_V2_S_Weights.DEFAULT 
                if not preds:
                    model.encoder.classifier[1] = nn.Identity()
            elif backbone == 'InceptionV3':
                weights = tmodels.Inception_V3_Weights.DEFAULT
                if not preds:
                    model.encoder.fc = nn.Identity()
            elif backbone == "MaxVit":
                weights = tmodels.MaxVit_T_Weights.DEFAULT
                if not preds:
                    model.encoder.classifier[5] = nn.Identity()
            elif backbone == "MobileNetV3":
                weights = tmodels.MobileNet_V3_Small_Weights.DEFAULT
                if not preds:
                    model.encoder.classifier[3] = nn.Identity()
            elif backbone == "ResNetXt":
                weights = tmodels.ResNeXt50_32X4D_Weights.DEFAULT
                if not preds:
                    model.encoder.fc = nn.Identity()
            elif backbone == "ShuffleNetV2":
                weights = tmodels.ShuffleNet_V2_X0_5_Weights.DEFAULT
                if not preds:
                    model.encoder.fc = nn.Identity()
            elif backbone == "ViT":
                weights = tmodels.ViT_B_16_Weights.DEFAULT
                if not preds:
                    model.encoder.heads.head = nn.Identity()
            elif backbone == "WideResnet":
                weights = tmodels.Wide_ResNet50_2_Weights.DEFAULT
                if not preds:
                    model.encoder.fc = nn.Identity()
            elif backbone == 'densenet':
                weights = tmodels.DenseNet121_Weights.DEFAULT
                if not preds:
                    model.encoder.classifier = nn.Identity() 
            elif backbone == 'swin':
                weights = tmodels.Swin_T_Weights.DEFAULT
                if not preds:
                    model.encoder.head = nn.Identity()
            else:
                raise ValueError(f'Backbone {backbone} not supported')

            model.to(device)
            model.eval()

            preprocess = weights.transforms(antialias=True)

            # Get the patches
            flat_patches = self.adata.obsm[f'patches_scale_{self.patch_scale}']

            # Reshape all the patches to the original shape
            all_patches = flat_patches.reshape((-1, self.patch_size, self.patch_size, 3))
            torch_patches = torch.from_numpy(all_patches).permute(0, 3, 1, 2).float()    # Turn all patches to torch
            rescaled_patches = torch_patches / 255                                       # Rescale patches to [0, 1]
            processed_patches = preprocess(rescaled_patches)                             # Preprocess patches
            
            # Create a dataloader
            dataloader = DataLoader(processed_patches, batch_size=256, shuffle=False, num_workers=4)

            # Declare lists to store the embeddings or predictions
            outputs = []

            with torch.no_grad():
                if preds:
                    desc = 'Getting predictions'
                else:
                    desc = 'Getting embeddings'
                for batch in tqdm(dataloader, desc=desc):
                    batch = batch.to(device)                    # Send batch to device                
                    batch_output = model(batch)                 # Get embeddings or predictions
                    outputs.append(batch_output)                # Append to list


            # Concatenate all embeddings or predictions
            outputs = torch.cat(outputs, dim=0)
        
            # Pass embeddings or predictions to cpu and add to self.data.obsm
            if preds:
                self.adata.obsm[f'predictions_{backbone}'] = outputs.cpu().numpy()
            else:
                self.adata.obsm[f'embeddings_{backbone}'] = outputs.cpu().numpy()

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

    def get_graphs(self, n_hops: int, layer: str) -> dict:
        """
        This function wraps the get_graphs_one_slide function to get the graphs for all the slides in the dataset. For details
        on the get_graphs_one_slide function see its docstring.

        Args:
            n_hops (int): The number of hops to compute each graph.
            layer (str): The layer of the graph to predict. Will be added as y to the graph.

        Returns:
            dict: A dictionary where the slide names are the keys and pytorch geometric graphs are values.
        """

        ### Define auxiliar functions ###

        def get_graphs_one_slide(self, adata: ad.AnnData, n_hops: int, layer: str, hex_geometry: bool) -> Tuple[dict,int]:
            """
            This function receives an AnnData object with a single slide and for each node computes the graph in an
            n_hops radius in a pytorch geometric format. It returns a dictionary where the patch names are the keys
            and a pytorch geometric graph for each one as values. NOTE: The first node of every graph is the center.

            Args:
                adata (ad.AnnData): The AnnData object with the slide data.
                n_hops (int): The number of hops to compute the graph.
                layer (str): The layer of the graph to predict. Will be added as y to the graph.
                hex_geometry (bool): Whether the slide has hexagonal geometry or not.

            Returns:
                Tuple(dict,int)
                dict: A dictionary where the patch names are the keys and pytorch geometric graph for each one as values.
                    NOTE: The first node of every graph is the center.
                int: Max absolute value of d pos in the slide                      
            """
            # Compute spatial_neighbors
            if hex_geometry:
                sq.gr.spatial_neighbors(adata, coord_type='generic', n_neighs=6) # Hexagonal visium case
            else:
                sq.gr.spatial_neighbors(adata, coord_type='grid', n_neighs=8) # Grid STNet dataset case

            # Get the adjacency matrix
            adj_matrix = adata.obsp['spatial_connectivities']

            # Define power matrix
            power_matrix = adj_matrix.copy()
            # Define the output matrix
            output_matrix = adj_matrix.copy()

            # Iterate through the hops
            for i in range(n_hops-1):
                # Compute the next hop
                power_matrix = power_matrix * adj_matrix
                # Add the next hop to the output matrix
                output_matrix = output_matrix + power_matrix

            # Zero out the diagonal
            output_matrix.setdiag(0)
            # Threshold the matrix to 0 and 1
            output_matrix = output_matrix.astype(bool).astype(int)

            # Define dict from index to obs name
            index_to_obs = {i: obs for i, obs in enumerate(adata.obs.index.values)}

            # Define neighbors dicts (one with names and one with indexes)
            neighbors_dict_index = {}
            neighbors_dict_names = {}
            matrices_dict = {}

            # Iterate through the rows of the output matrix
            for i in range(output_matrix.shape[0]):
                # Get the non-zero elements of the row
                non_zero_elements = output_matrix[i].nonzero()[1]
                # Get the names of the neighbors
                non_zero_names = [index_to_obs[index] for index in non_zero_elements]
                # Add the neighbors to the neighbors dicts. NOTE: the first index is the query obs
                neighbors_dict_index[i] = [i] + list(non_zero_elements)
                neighbors_dict_names[index_to_obs[i]] = np.array([index_to_obs[i]] + non_zero_names)
                
                # Subset the matrix to the non-zero elements and store it in the matrices dict
                matrices_dict[index_to_obs[i]] = output_matrix[neighbors_dict_index[i], :][:, neighbors_dict_index[i]]

            
            ### Get pytorch geometric graphs ###
            patch_names = adata.obs.index.values                                                                        # Get global patch names
            layers_dict = {key: torch.from_numpy(adata.layers[key]).type(torch.float32) for key in adata.layers.keys()} # Get global layers
            patches = torch.from_numpy(adata.obsm[f'patches_scale_{self.patch_scale}'])                                 # Get global patches
            pos = torch.from_numpy(adata.obs[['array_row', 'array_col']].values)                                        # Get global positions

            # Get embeddings and predictions keys
            emb_key_list = [k for k in adata.obsm.keys() if 'embeddings' in k]
            pred_key_list = [k for k in adata.obsm.keys() if 'predictions' in k]
            assert len(emb_key_list) == 1, 'There are more than 1 or no embedding keys in adata.obsm'
            assert len(pred_key_list) == 1, 'There are more than 1 or no prediction keys in adata.obsm'
            emb_key, pred_key = emb_key_list[0], pred_key_list[0]

            # If embeddings and predictions are present in obsm, get them
            embeddings = torch.from_numpy(adata.obsm[emb_key]).type(torch.float32)
            predictions = torch.from_numpy(adata.obsm[pred_key]).type(torch.float32)

            # If layer contains delta then add a used_mean attribute to the graph
            used_mean = torch.from_numpy(self.adata.var[f'{layer}_avg_exp'.replace('deltas', 'log1p')].values).type(torch.float32) if 'deltas' in layer else None

            # Define the empty graph dict
            graph_dict = {}
            max_abs_d_pos=-1

            # Cycle over each obs
            for i in tqdm(range(len(neighbors_dict_index)), leave=False, position=1):
                central_node_name = index_to_obs[i]                                                 # Get the name of the central node
                curr_nodes_idx = torch.tensor(neighbors_dict_index[i])                              # Get the indexes of the nodes in the graph
                curr_adj_matrix = matrices_dict[central_node_name]                                  # Get the adjacency matrix of the graph (precomputed)
                curr_edge_index, curr_edge_attribute = from_scipy_sparse_matrix(curr_adj_matrix)    # Get the edge index and edge attribute of the graph
                curr_layers = {key: layers_dict[key][curr_nodes_idx] for key in layers_dict.keys()} # Get the layers of the graph filtered by the nodes
                curr_pos = pos[curr_nodes_idx]                                                      # Get the positions of the nodes in the graph
                curr_d_pos = curr_pos - curr_pos[0]                                                 # Get the relative positions of the nodes in the graph

                # Define the graph
                graph_dict[central_node_name] = geo_Data(
                    # x=patches[curr_nodes_idx],
                    y=curr_layers[layer],
                    edge_index=curr_edge_index,
                    # edge_attr=curr_edge_attribute,
                    pos=curr_pos,
                    d_pos=curr_d_pos,
                    # patch_names=patch_names[neighbors_dict_index[i]],
                    embeddings=embeddings[curr_nodes_idx],
                    predictions=predictions[curr_nodes_idx] if predictions is not None else None,
                    used_mean=used_mean if used_mean is not None else None,
                    num_nodes=len(curr_nodes_idx),
                    mask=layers_dict['mask'][curr_nodes_idx]
                    # **curr_layers
                )

                max_curr_d_pos=curr_d_pos.abs().max()
                if max_curr_d_pos>max_abs_d_pos:
                    max_abs_d_pos=max_curr_d_pos

            #cast as int
            max_abs_d_pos=int(max_abs_d_pos)
            
            # Return the graph dict
            return graph_dict, max_abs_d_pos

        def get_sin_cos_positional_embeddings(self, graph_dict: dict, max_d_pos: int) -> dict:
            
            """This function adds the positional embeddings of each node to the graph dict.

            Args:
                graph_dict (dict): A dictionary where the patch names are the keys and pytorch geometric graph for each one as values
                max_d_pos (int): Max absolute value in the relative position matrix.

            Returns:
                dict: The input graph dict with the information of positional encodings for each graph.
            """
            graph_dict_keys = list(graph_dict.keys())
            embedding_dim = graph_dict[graph_dict_keys[0]].embeddings.shape[1]

            # Define the positional encoding model
            p_encoding_model= PositionalEncoding2D(embedding_dim)

            # Define the empty grid with size (batch_size, x, y, channels)
            grid_size = torch.zeros([1, 2*max_d_pos+1, 2*max_d_pos+1, embedding_dim])

            # Obtain the embeddings for each position
            positional_look_up_table = p_encoding_model(grid_size)        

            for key, value in graph_dict.items():
                d_pos = value.d_pos
                grid_pos = d_pos + max_d_pos
                graph_dict[key].positional_embeddings = positional_look_up_table[0,grid_pos[:,0],grid_pos[:,1],:]
            
            return graph_dict

        print('Computing graphs...')

        # Get unique slide ids
        unique_ids = self.adata.obs['slide_id'].unique()

        # Global dictionary to store the graphs (pytorch geometric graphs)
        graph_dict = {}
        max_global_d_pos=-1

        # Iterate through slides
        for slide in tqdm(unique_ids, leave=True, position=0):
            curr_adata = self.get_slide_from_collection(self.adata, slide)
            curr_graph_dict, max_curr_d_pos = get_graphs_one_slide(self, curr_adata, n_hops, layer, self.hex_geometry)
            
            # Join the current dictionary to the global dictionary
            graph_dict = {**graph_dict, **curr_graph_dict}

            if max_curr_d_pos>max_global_d_pos:
                max_global_d_pos=max_curr_d_pos
        
        graph_dict = get_sin_cos_positional_embeddings(self, graph_dict, max_global_d_pos)

        # Return the graph dict
        return graph_dict
    
    def get_pretrain_dataloaders(self, layer: str = 'c_d_log1p', batch_size: int = 128, shuffle: bool = True, use_cuda: bool = False) -> Tuple[AnnLoader, AnnLoader, AnnLoader]:
        """
        This function returns the dataloaders for the pre-training phase. This means training a purely vision-based model on only
        the patches to predict the gene expression of the patches.

        Args:
            layer (str, optional): The layer to use for the pre-training. The self.adata.X will be set to that of 'layer'. Defaults to 'deltas'.
            batch_size (int, optional): The batch size of the loaders. Defaults to 128.
            shuffle (bool, optional): Whether to shuffle the data in the loaders. Defaults to True.
            use_cuda (bool, optional): True for using cuda in the loader. Defaults to False.

        Returns:
            Tuple[AnnLoader, AnnLoader, AnnLoader]: The train, validation and test dataloaders. If there is no test set, the test dataloader is None.
        """
        # Get the sample indexes for the train, validation and test sets
        idx_train, idx_val, idx_test = self.adata.obs[self.adata.obs.split == 'train'].index, self.adata.obs[self.adata.obs.split == 'val'].index, self.adata.obs[self.adata.obs.split == 'test'].index

        # FIXME: Put this in process_dataset
        ##### Adition to handle noisy training (this is temporal and should be in process_dataset) #####

        # Handle noisy training
        if layer == 'noisy':
            # Copy the layer c_d_log1p to the layer noisy
            c_d_log1p = self.adata.layers['c_d_log1p'].copy()
            # Get zero mask
            zero_mask = ~self.adata.layers['mask']
            # Zero out the missing values
            c_d_log1p[zero_mask] = 0
            # Add the layer to the adata
            self.adata.layers['noisy'] = c_d_log1p

            # Give warning to say that the noisy layer is being used
            print('Using noisy layer for training. This will probably yield bad results.')

        # Handle noisy_delta case
        if layer == 'noisy_d':
            # Get vector with gene means
            gene_means = self.adata.var['c_d_log1p_avg_exp'].values
            # Expand gene means to the shape of the layer
            gene_means = np.repeat(gene_means.reshape(1, -1), self.adata.n_obs, axis=0)
            # Get valid mask
            valid_mask = self.adata.layers['mask']
            # Initialize noisy deltas
            noisy_deltas = -gene_means 
            # Assign delta values in positions where valid mask is true
            noisy_deltas[valid_mask] = self.adata.layers['c_d_deltas'][valid_mask]
            # Add the layer to the adata
            self.adata.layers['noisy_d'] = noisy_deltas

            # Give warning to say that the noisy layer is being used
            print('Using noisy_delta layer for training. This will probably yield bad results.')

        ##### End of adition #####

        # Set the X of the adata to the layer casted to float32
        self.adata.X = self.adata.layers[layer].astype(np.float32)

        imp_model_str = 'transformer model' if layer in ['c_t_log1p', 'c_t_deltas'] else 'median filter'

        # Print with the percentage of the dataset that was replaced
        print('Percentage of imputed observations with {}: {:5.3f}%'.format(imp_model_str, 100 * (~self.adata.layers["mask"]).sum() / (self.adata.n_vars*self.adata.n_obs)))

        # If the prediction layer is some form of deltas, add the used mean of the layer as a column in the var
        if 'deltas' in layer:
            # Add a var column of used means of the layer
            mean_key = f'{layer}_avg_exp'.replace('deltas', 'log1p')
            self.adata.var['used_mean'] = self.adata.var[mean_key]
        
        ### Adition to handle noisy training (this is temporal and should be in process_dataset) ###
        if layer == 'noisy_d':
            # Add a var column of used means of the layer
            mean_key = f'c_d_log1p_avg_exp'
            self.adata.var['used_mean'] = self.adata.var[mean_key]
        ### End of adition ###

        # Subset the global data handle also the possibility that there is no test set
        adata_train, adata_val = self.adata[idx_train, :], self.adata[idx_val, :]
        adata_test = self.adata[idx_test, :] if len(idx_test) > 0 else None

        # Declare dataloaders
        train_dataloader = AnnLoader(adata_train, batch_size=batch_size, shuffle=shuffle, use_cuda=use_cuda)
        val_dataloader = AnnLoader(adata_val, batch_size=batch_size, shuffle=shuffle, use_cuda=use_cuda)
        test_dataloader = AnnLoader(adata_test, batch_size=batch_size, shuffle=shuffle, use_cuda=use_cuda) if adata_test is not None else None

        return train_dataloader, val_dataloader, test_dataloader
    
    # TODO: Make this function more elegant
    def get_graph_dataloaders(self, layer: str = 'c_d_log1p', n_hops: int = 2, backbone: str ='densenet', model_path: str = "best_stnet.pt", batch_size: int = 128, shuffle: bool = True) -> Tuple[geo_DataLoader, geo_DataLoader, geo_DataLoader]:
        # Get dictionary of parameters to get the graphs
        curr_graph_params = {
            'n_hops': n_hops,
            'layer': layer,
            'backbone': backbone,
            'model_path': model_path
        }        

        # Create graph directory if it does not exist
        os.makedirs(os.path.join(self.dataset_path, 'graphs'), exist_ok=True)
        # Get the filenames of the parameters of all directories in the graph folder
        filenames = glob.glob(os.path.join(self.dataset_path, 'graphs', '**', 'graph_params.json' ), recursive=True)

        # Define boolean to check if the graphs are already saved
        found_graphs = False

        # Iterate over all the filenames and check if the parameters are the same
        for filename in filenames:
            with open(filename, 'r') as f:
                # Load the parameters of the dataset
                saved_params = json.load(f)
                # Check if the parameters are the same
                if saved_params == curr_graph_params:
                    print(f'Graph data already saved in {filename}')
                    found_graphs = True
                    # Track the time and load the graphs
                    start = time()
                    train_graphs = torch.load(os.path.join(os.path.dirname(filename), 'train_graphs.pt'))
                    val_graphs = torch.load(os.path.join(os.path.dirname(filename), 'val_graphs.pt'))
                    test_graphs = torch.load(os.path.join(os.path.dirname(filename), 'test_graphs.pt')) if os.path.exists(os.path.join(os.path.dirname(filename), 'test_graphs.pt')) else None
                    print(f'Loaded graphs in {time() - start:.2f} seconds.')
                    break

        # If the graphs are not found, compute them
        if not found_graphs:
            
            # Print that we are computing the graphs
            print('Graphs not found in file, computing graphs...')

            # FIXME: Put this in process_dataset
            ##### Adition to handle noisy training (this is temporal and should be in process_dataset) #####

            # Handle noisy training
            if layer == 'noisy':
                # Copy the layer c_d_log1p to the layer noisy
                c_d_log1p = self.adata.layers['c_d_log1p'].copy()
                # Get zero mask
                zero_mask = ~self.adata.layers['mask']
                # Zero out the missing values
                c_d_log1p[zero_mask] = 0
                # Add the layer to the adata
                self.adata.layers['noisy'] = c_d_log1p

                # Give warning to say that the noisy layer is being used
                print('Using noisy layer for training. This will probably yield bad results.')

            # Handle noisy_delta case
            if layer == 'noisy_d':
                # Get vector with gene means
                gene_means = self.adata.var['c_d_log1p_avg_exp'].values
                # Expand gene means to the shape of the layer
                gene_means = np.repeat(gene_means.reshape(1, -1), self.adata.n_obs, axis=0)
                # Get valid mask
                valid_mask = self.adata.layers['mask']
                # Initialize noisy deltas
                noisy_deltas = -gene_means 
                # Assign delta values in positions where valid mask is true
                noisy_deltas[valid_mask] = self.adata.layers['c_d_deltas'][valid_mask]
                # Add the layer to the adata
                self.adata.layers['noisy_d'] = noisy_deltas

                # Give warning to say that the noisy layer is being used
                print('Using noisy_delta layer for training. This will probably yield bad results.')

            ##### End of adition #####

            # We compute the embeddings and predictions for the patches
            self.compute_patches_embeddings_and_predictions(preds=True, backbone=backbone, model_path=model_path)
            self.compute_patches_embeddings_and_predictions(preds=False, backbone=backbone, model_path=model_path)
            
            # Get graph dicts
            general_graph_dict = self.get_graphs(n_hops=n_hops, layer=layer)

            # Get the train, validation and test indexes
            idx_train, idx_val, idx_test = self.adata.obs[self.adata.obs.split == 'train'].index, self.adata.obs[self.adata.obs.split == 'val'].index, self.adata.obs[self.adata.obs.split == 'test'].index

            # Get list of graphs
            train_graphs = [general_graph_dict[idx] for idx in idx_train]
            val_graphs = [general_graph_dict[idx] for idx in idx_val]
            test_graphs = [general_graph_dict[idx] for idx in idx_test] if len(idx_test) > 0 else None

            print('Saving graphs...')
            # Create graph directory if it does not exist with the current time
            graph_dir = os.path.join(self.dataset_path, 'graphs', datetime.now().strftime("%d-%m-%Y-%H-%M-%S"))
            os.makedirs(graph_dir, exist_ok=True)

            # Save the graph parameters
            with open(os.path.join(graph_dir, 'graph_params.json'), 'w') as f:
                # Write the json
                json.dump(curr_graph_params, f, indent=4)

            torch.save(train_graphs, os.path.join(graph_dir, 'train_graphs.pt'))
            torch.save(val_graphs, os.path.join(graph_dir, 'val_graphs.pt'))
            torch.save(test_graphs, os.path.join(graph_dir, 'test_graphs.pt')) if test_graphs is not None else None
        

        # Declare dataloaders
        train_dataloader = geo_DataLoader(train_graphs, batch_size=batch_size, shuffle=shuffle)
        val_dataloader = geo_DataLoader(val_graphs, batch_size=batch_size, shuffle=shuffle)
        test_dataloader = geo_DataLoader(test_graphs, batch_size=batch_size, shuffle=shuffle) if test_graphs is not None else None

        return train_dataloader, val_dataloader, test_dataloader
    
    # TODO: Add documentation
    def get_predictions(self, model, layer: str = 'c_d_log1p', batch_size: int = 128, use_cuda: bool = False)->None:
        """
        _summary_

        Args:
            model (_type_): _description_
            layer (str, optional): _description_. Defaults to 'c_d_log1p'.
            batch_size (int, optional): _description_. Defaults to 128.
            use_cuda (bool, optional): _description_. Defaults to False.
        """
        # Set the X of the adata to the layer casted to float32
        self.adata.X = self.adata.layers[layer].astype(np.float32)

        # Get complete dataloader
        dataloader = AnnLoader(self.adata, batch_size=batch_size, shuffle=False, use_cuda=use_cuda)

        # Define global variables
        glob_expression_pred = None
        glob_ids = None

        # Set model to eval mode
        model.eval()

        # Get complete predictions
        with torch.no_grad():
            for data in dataloader:
                # Get patches
                tissue_tiles = data.obsm[f'patches_scale_{self.patch_scale}']
                tissue_tiles = tissue_tiles.reshape((tissue_tiles.shape[0], self.patch_size, self.patch_size, -1))
                # Permute dimensions to be in correct order for normalization
                tissue_tiles = tissue_tiles.permute(0,3,1,2).contiguous()
                # Make transformations in tissue tiles
                tissue_tiles = tissue_tiles/255.
                tissue_tiles = model.test_transforms(tissue_tiles)

                # Get output of the model
                # If tissue tiles is tuple then we will compute outputs of the 8 symmetries and then average them for prediction
                if isinstance(tissue_tiles, tuple):
                    pred_list = [model(tissue_rot) for tissue_rot in tissue_tiles]
                    pred_stack = torch.stack(pred_list)
                    expression_pred = pred_stack.mean(dim=0)
                # If tissue tiles is not tuple then a single prediction is done with the original image
                else:
                    expression_pred = model(tissue_tiles)

                # Concat batch to get global predictions and IDs
                glob_expression_pred = expression_pred if glob_expression_pred is None else torch.cat((glob_expression_pred, expression_pred))
                glob_ids = data.obs['unique_id'].tolist() if glob_ids is None else glob_ids + data.obs['unique_id'].tolist()

            # Handle delta prediction
            if 'deltas' in layer:
                mean_key = f'{layer}_avg_exp'.replace('deltas', 'log1p')
                means = torch.tensor(self.adata.var[mean_key], device=glob_expression_pred.device)
                glob_expression_pred = glob_expression_pred+means
            
            
            # Put complete predictions in a single dataframe
            pred_matrix = glob_expression_pred.detach().cpu().numpy()
            pred_df = pd.DataFrame(pred_matrix, index=glob_ids, columns=self.adata.var_names)
            pred_df = pred_df.reindex(self.adata.obs.index)

            # Log predictions to wandb
            wandb_df = pred_df.reset_index(names='sample')
            wandb.log({'predictions': wandb.Table(dataframe=wandb_df)})
            
            # Add layer to adata
            self.adata.layers[f'predictions,{layer}'] = pred_df
    
    # TODO: Add documentation  
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
            slides_adatas_dict = {p: self.get_slide_from_collection(self.adata, slides[p]) for p in slides.keys()}

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
    
    def get_adata_for_problem(self, problem:str="imputation_normal", seed:int=42, train: float=0.6, val: float=0.4, test: float=None) -> ad.AnnData:
        """
        This function receives an anndata split into train, valid and test by slides and 
        returns the same anndata split into train, valid and test by random patches from all slides if
        split equals "imputation_random" and return the same anndata as the received one if split equals "imputation_normal". 

        Args: 
            adata (ad.AnnData): AnnData object splitted by slides.
            seed (string): seed for random splits
            train (float): percentage of distribution for train split
            val (float): percentage of distribution for valid split
            test (float): percentage of distribution for test split (if no test, should be set as None)

        Returns:
            ad.AnnData: AnnData splitted into train, valid and test either by slides or by random patches
        """
        
        # Split data randomly 
        n_patches = np.arange (self.adata.shape[0]).tolist()
        random.seed(seed)
        # Define new train adata
        train_patches = random.sample(n_patches, int(self.adata.shape[0]*train))
        n_patches = [n for n in n_patches if n not in train_patches]
        new_train_adata = self.adata[train_patches].copy()
        # Define new valid adata
        val_patches = random.sample(n_patches, int(self.adata.shape[0]*val))
        new_val_adata = self.adata[val_patches].copy()
        # Define new splits for train and valid
        new_train_adata.obs["split"] = "train"
        new_val_adata.obs["split"] = "val"
        
        if "test" in self.adata.obs["split"].unique().tolist() and test != None:
            # Define new test adata
            test_patches = [n for n in n_patches if n not in val_patches]
            new_test_adata = self.adata[test_patches].copy()
            # Define new split for test
            new_test_adata.obs["split"] = "test"
            # Adata corresponding only to the patches that are going to be used to evaluate (intersection patches)
            metric_patches_adata = new_test_adata[self.adata[test_patches].obs["split"]=="test"].copy()
            # Concatenate the new splits adata
            problem_adata = ad.concat([new_train_adata, new_val_adata, new_test_adata], merge="same", uns_merge="first")
        else:
            # Adata corresponding only to the patches that are going to be used to evaluate (intersection patches)
            metric_patches_adata = new_val_adata[self.adata[val_patches].obs["split"]=="val"].copy()
            # Concatenate the new splits adata
            problem_adata = ad.concat([new_train_adata, new_val_adata], merge="same", uns_merge="first")
        
        # Create AnnData depending on the problem input
        if problem == 'imputation_normal':
            # If problem input corresponds to imputation_normal the the return AnnData is the original one
            problem_adata = self.adata.copy()

        #Add intersection patches in adata.uns
        problem_adata.uns["intersection_patches"] = metric_patches_adata.obsm[f'patches_scale_{self.patch_scale}']
    
        return problem_adata

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
            train_dl, val_dl, test_dl = dataset.get_graph_dataloaders(layer=layer, n_hops=3, backbone='ViT', model_path=model_path, batch_size=256, shuffle=False)

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
        

    
