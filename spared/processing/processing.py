import anndata as ad
from anndata.experimental.pytorch import AnnLoader
import torch
#from . import im_encoder
import torchvision.models as tmodels
import torch.nn as nn
from positional_encodings.torch_encodings import PositionalEncoding2D
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
import glob
import json
from time import time
from datetime import datetime
from torch_geometric.data import Data as geo_Data
from torch_geometric.loader import DataLoader as geo_DataLoader
import numpy as np
import pandas as pd
import pathlib
import shutil
import wget
import gzip
import subprocess
from combat.pycombat import pycombat
import scanpy as sc
from sklearn.preprocessing import StandardScaler
import squidpy as sq
from torch_geometric.utils import from_scipy_sparse_matrix
from typing import Tuple
import sys
from typing import Tuple

#FIXME: Remove comments in spanish

from torchvision import transforms 
# Get the path of the spared database
# SPARED_PATH = pathlib.Path(__file__).parent

# El path a spared es ahora diferente
SPARED_PATH = pathlib.Path(__file__).resolve().parent.parent

# Agregar el directorio padre al sys.path para los imports
sys.path.append(str(SPARED_PATH))
# Import im_encoder.py file
from embeddings import im_encoder
# Remove the path from sys.path
sys.path.remove(str(SPARED_PATH))

### Data analysis and filtering functions: 

def get_slide_from_collection(collection: ad.AnnData,  slide: str) -> ad.AnnData:
    """ Retrieve a slide from a collection of slides.

    This function receives a slide name and returns an AnnData object of the specified slide based on the collection of slides
    in the collection parameter.

    Args: 
        collection (ad.AnnData): AnnData object with all the slides concatenated.
        slide (str): Name of the slide to get from the collection. Must be in the ``slide_id`` column of the ``collection.obs`` dataframe.

    Returns:
        ad.AnnData: An AnnData object with the specified slide.
    """

    # Get the slide from the collection
    slide_adata = collection[collection.obs['slide_id'] == slide].copy()
    # Modify the uns dictionary to include only the information of the slide
    slide_adata.uns['spatial'] = {slide: collection.uns['spatial'][slide]}

    # Return the slide
    return slide_adata

def get_exp_frac(adata: ad.AnnData) -> ad.AnnData:
    """ Compute the expression fraction for all genes.

    The expression fraction of a gene in a slide is defined as the proportion of spots where that gene is expressed. It is a number between ``0``
    and ``1`` where ``0`` means that the gene is not expressed in any spot and ``1`` means that the gene is expressed in all the spots.

    To compute an aggregation of expression fractions in a complete dataset, this function gets the
    expression fraction for each slide and then takes the minimum across all the slides. Hence the final number is a lower bound that ensures
    that the gene is expressed in at least that fraction of the spots in each of the slides.

    Args:
        adata (ad.AnnData): A slide collection where non-expressed genes have a value of ``0`` in the ``adata.X`` matrix.

    Returns:
        ad.AnnData: The updated slide collection with the added information into the ``adata.var['exp_frac']`` column.
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
    """ Compute the global expression fraction for all genes.
    
    This function computes the global expression fraction for each gene in a dataset.

    The global expression fraction of a gene in a dataset is defined as the proportion of spots where that gene is expressed. It is a number between ``0``
    and ``1`` where ``0`` means that the gene is not expressed in any spot and ``1`` means that the gene is expressed in all the spots. Its difference
    with the expression fraction is that the global expression fraction is computed for the whole dataset and not for each slide.

    Args:
        adata (ad.AnnData): A slide collection where a non-expressed genes have a value of ``0`` in the ``adata.X`` matrix.

    Returns:
        ad.AnnData: The updated slide collection with the information added into the  ``adata.var['glob_exp_frac']`` column.
    """
    # Get global expression fraction
    glob_exp_frac = np.squeeze(np.asarray((adata.X > 0).sum(axis=0) / adata.n_obs))

    # Add the global expression fraction to the var dataframe of the slide collection
    adata.var['glob_exp_frac'] = glob_exp_frac

    # Return the adata
    return adata

def filter_dataset(adata: ad.AnnData, param_dict: dict) -> ad.AnnData:
    """ Perform complete filtering pipeline of a slide collection.

    This function takes a completely unfiltered and unprocessed (in raw counts) slide collection and filters it
    (both samples and genes) according to the ``param_dict`` argument.
    A summary list of the steps is the following:

        1. Filter out observations with ``total_counts`` outside the range ``[param_dict['cell_min_counts'], param_dict['cell_max_counts']]``.
           This filters out low quality observations not suitable for analysis.
        2. Compute the ``exp_frac`` for each gene. This means that for each slide in the collection we compute the fraction of the spots that express each gene and then take the minimum across all the slides (see ``get_exp_frac`` function for more details).
        3. Compute the ``glob_exp_frac`` for each gene. This is similar to the ``exp_frac`` but instead of computing for each
           slide and taking the minimum we compute it for the whole collection. Slides don't matter here
           (see ``get_glob_exp_frac`` function for more details).
        4. Filter out genes depending on the ``param_dict['wildcard_genes']`` value, the options are the following:

            a. ``param_dict['wildcard_genes'] == 'None'``:

                - Filter out genes that are not expressed in at least ``param_dict['min_exp_frac']`` of spots in each slide.
                - Filter out genes that are not expressed in at least ``param_dict['min_glob_exp_frac']`` of spots in the whole collection.
                - Filter out genes with counts outside the range ``[param_dict['gene_min_counts'], param_dict['gene_max_counts']]``
            b. ``param_dict['wildcard_genes'] != 'None'``:

                - Read ``.txt`` file specified by ``param_dict['wildcard_genes']`` and leave only the genes that are in this file.
        5. If there are spots with zero counts in all genes after gene filtering, remove them.
        6. Compute quality control metrics using scanpy's ``sc.pp.calculate_qc_metrics`` function.

    Args:
        adata (ad.AnnData): An unfiltered (unexpressed genes are encoded as ``0`` on the ``adata.X matrix``) slide collection.
        param_dict (dict): Dictionary that contains filtering and processing parameters. Keys that must be present are:

            - ``'cell_min_counts'`` (*int*):      Minimum total counts for a spot to be valid.
            - ``'cell_max_counts'`` (*int*):      Maximum total counts for a spot to be valid.
            - ``'gene_min_counts'`` (*int*):      Minimum total counts for a gene to be valid.
            - ``'gene_max_counts'`` (*int*):      Maximum total counts for a gene to be valid.
            - ``'min_exp_frac'`` (*float*):       Minimum fraction of spots in any slide that must express a gene for it to be valid.
            - ``'min_glob_exp_frac'`` (*float*):  Minimum fraction of spots in the whole collection that must express a gene for it to be valid.
            - ``'wildcard_genes'`` (*str*):       Path to a ``.txt`` file with the genes to keep or ``'None'`` to filter genes based on the other keys.

    Returns:
        ad.AnnData: The filtered adata collection.
    """

    # Start tracking time
    print('Starting data filtering...')
    start = time()

    # Get initial gene and observation numbers
    n_genes_init = adata.n_vars
    n_obs_init = adata.n_obs

    ### Filter out samples:

    # Find indexes of cells with total_counts outside the range [cell_min_counts, cell_max_counts]
    sample_counts = np.squeeze(np.asarray(adata.X.sum(axis=1)))
    bool_valid_samples = (sample_counts > param_dict['cell_min_counts']) & (sample_counts < param_dict['cell_max_counts'])
    valid_samples = adata.obs_names[bool_valid_samples]

    # Subset the adata to keep only the valid samples
    adata = adata[valid_samples, :].copy()

    ### Filter out genes:

    # Compute the min expression fraction for each gene across all the slides
    adata = get_exp_frac(adata)
    # Compute the global expression fraction for each gene
    adata = get_glob_exp_frac(adata)
    
    # If no wildcard genes are specified then filter genes based in min_exp_frac and total counts
    if param_dict['wildcard_genes'] == 'None':
        
        gene_counts = np.squeeze(np.asarray(adata.X.sum(axis=0)))
                    
        # Find indexes of genes with total_counts inside the range [gene_min_counts, gene_max_counts]
        bool_valid_gene_counts = (gene_counts > param_dict['gene_min_counts']) & (gene_counts < param_dict['gene_max_counts'])
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
        num_genes = np.where(df_exp['vol_real_data'] >= param_dict['real_data_percentage'])[0][-1]
        valid_genes = df_exp.iloc[:num_genes + 1]['gene_ids']
        # Subset the adata to keep only the valid genes
        adata = adata[:, valid_genes].copy()
    
    # If there are wildcard genes then read them and subset the dataset to just use them
    else:
        # Read valid wildcard genes
        genes = pd.read_csv(param_dict['wildcard_genes'], sep=" ", header=None, index_col=False)
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

### Expression data processing functions:

# FIXME: Put the organism as a parameter instead of in the name of the dataset
def tpm_normalization(organism: str, adata: ad.AnnData, from_layer: str, to_layer: str) -> ad.AnnData:
    """Normalize expression using TPM normalization.

    This function applies TPM normalization to an AnnData object with raw counts. It also removes genes that are not fount in the ``.gtf`` annotation file.
    The counts are taken from ``adata.layers[from_layer]`` and the results are stored in ``adata.layers[to_layer]``. It can perform the normalization
    for `human <https://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_human/release_43/gencode.v43.basic.annotation.gtf.gz>`_ and `mouse
    <https://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_mouse/release_M33/gencode.vM33.basic.annotation.gtf.gz>`_ reference genomes.
    To specify which GTF annotation file should be used, the string parameter ``'organism'`` must be ``'mouse'`` or ``'human'``.

    Args:
        organism (str): Organism of the dataset. Must be 'mouse' or 'human'.
        adata (ad.Anndata): The Anndata object to normalize.
        from_layer (str): The layer to take the counts from. The data in this layer should be in raw counts.
        to_layer (str): The layer to store the results of the normalization.
    Returns:
        ad.Anndata: The updated Anndata object with TPM values in ``adata.layers[to_layer]``.
    """
    
    # Get the number of genes before filtering
    initial_genes = adata.shape[1]

    # Automatically download the human gtf annotation file if it is not already downloaded
    if not os.path.exists(os.path.join(SPARED_PATH, 'data', 'annotations', 'gencode.v43.basic.annotation.gtf.gz')):
        print('Automatically downloading human gtf annotation file...')
        os.makedirs(os.path.join(SPARED_PATH, 'data', 'annotations'), exist_ok=True)
        wget.download(
            'https://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_human/release_43/gencode.v43.basic.annotation.gtf.gz',
            out = os.path.join(SPARED_PATH, 'data', 'annotations', 'gencode.v43.basic.annotation.gtf.gz'))

    # Define gtf human path
    gtf_path = os.path.join(SPARED_PATH, 'data', 'annotations', 'gencode.v43.basic.annotation.gtf')

    # Unzip the data in annotations folder if it is not already unzipped
    if not os.path.exists(gtf_path):
        with gzip.open(os.path.join(SPARED_PATH, 'data', 'annotations', 'gencode.v43.basic.annotation.gtf.gz'), 'rb') as f_in:
            with open(gtf_path, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)

    # FIXME: Set up automatic download of mouse gtf file (DONE)
    # Automatically download the mouse gtf annotation file if it is not already downloaded
    if not os.path.exists(os.path.join(SPARED_PATH, 'data', 'annotations', 'gencode.vM33.basic.annotation.gtf.gz')):
        print('Automatically downloading mouse gtf annotation file...')
        os.makedirs(os.path.join(SPARED_PATH, 'data', 'annotations'), exist_ok=True)
        wget.download(
            'https://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_mouse/release_M33/gencode.vM33.basic.annotation.gtf.gz',
            out = os.path.join(SPARED_PATH, 'data', 'annotations', 'gencode.vM33.basic.annotation.gtf.gz'))
    
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
    if organism.lower() == "mouse":
        glength_df = pd.read_csv(gene_length_path_mouse, delimiter='\t', usecols=['gene', 'merged'])
    elif organism.lower() == "human":
        glength_df = pd.read_csv(gene_length_path, delimiter='\t', usecols=['gene', 'merged'])
    else:
        assert "Organism not valid"

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
    """Perform :math:`\log_2(x+1)` transformation 

    Performs logarithmic transformation over ``adata.layers[from_layer]``. Simple wrapper of scanpy's ``sc.pp.log1p()``
    (base 2) to transform data from ``adata.layers[from_layer]`` and save it into ``adata.layers[to_layer]``.

    Args:
        adata (ad.AnnData): The AnnData object to transform.
        from_layer (str): The layer to take the data from.
        to_layer (str): The layer to store the results of the transformation.

    Returns:
        ad.AnnData: The updated AnnData object with transformed data in ``adata.layers[to_layer]``.
    """

    # Transform the data with log1p
    transformed_adata = sc.pp.log1p(adata, base= 2.0, layer=from_layer, copy=True)

    # Add the log1p transformed data to adata.layers[to_layer]
    adata.layers[to_layer] = transformed_adata.layers[from_layer]

    # Return the transformed AnnData object
    return adata

# FIXME: CHeck this is adequately included in documentation
### Define function to get spatial neighbors in an AnnData object
def get_spatial_neighbors(adata: ad.AnnData, n_hops: int, hex_geometry: bool) -> dict:
    """ Compute neighbors dictionary for an AnnData object.
    
    This function computes a neighbors dictionary for an AnnData object. The neighbors are computed according to topological distances over
    a graph defined by the hex_geometry connectivity. The neighbors dictionary is a dictionary where the keys are the indexes of the observations
    and the values are lists of the indexes of the neighbors of each observation. The neighbors include the observation itself and are found
    inside an n_hops neighborhood (vicinity) of the observation.

    Args:
        adata (ad.AnnData): The AnnData object to process. Importantly it is only from a single slide. Can not be a collection of slides.
        n_hops (int): The size of the neighborhood to take into account to compute the neighbors.
        hex_geometry (bool): Whether the graph is hexagonal or not. If True, then the graph is hexagonal. If False, then the graph is a grid. Only
                                true for visium datasets.

    Returns:
        dict: The neighbors dictionary. The keys are the indexes of the observations and the values are lists of the indexes of the neighbors of each observation.
    """

    # Compute spatial_neighbors
    if hex_geometry:
        sq.gr.spatial_neighbors(adata, coord_type='generic', n_neighs=6) # Hexagonal visium case
    else:
        sq.gr.spatial_neighbors(adata, coord_type='grid', n_neighs=8) # Grid dataset case

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

def clean_noise(collection: ad.AnnData, from_layer: str, to_layer: str, n_hops: int, hex_geometry: bool) -> ad.AnnData:
    """Remove noise with median filter.

    Function that cleans noise (missing data) with a modified adaptive median filter for each slide in an AnnData collection.
    Details of the adaptive median filter can be found in the ``adaptive_median_filter_pepper()`` function inside the source code.
    The data will be taken from ``adata.layers[from_layer]`` and the results will be stored in ``adata.layers[to_layer]``.

    Args:
        collection (ad.AnnData): The AnnData collection to process.
        from_layer (str): The layer to compute the adaptive median filter from. Where to clean the noise from.
        to_layer (str): The layer to store the results of the adaptive median filter. Where to store the cleaned data.
        n_hops (int): The maximum number of concentric rings in the neighbors graph to take into account to compute the median. Analogous to the maximum window size.
        hex_geometry (bool): ``True``, if the graph has hexagonal spatial geometry (Visium technology). If False, then the graph is a grid.

    Returns:
        ad.AnnData: New AnnData collection with the results of the adaptive median filter stored in the layer ``adata.layers[to_layer]``.
    """

    ### Define cleaning function for single slide:
    def adaptive_median_filter_pepper(adata: ad.AnnData, from_layer: str, to_layer: str, n_hops: int, hex_geometry: bool) -> ad.AnnData:
        """
        This function computes a modified adaptive median filter for pairs (obs, gene) with a zero value (peper noise) in the layer 'from_layer' and
        stores the result in the layer 'to_layer'. The max window size is a neighborhood of n_hops defined by the conectivity (hexagonal or grid).
        This means the number of concentric rings in a graph to take into account to compute the median.

        The adaptive median filter denoises each gene independently. In other words gene A has no influence on the denoising of gene B.

        Args:
            adata (ad.AnnData): The AnnData object to process. Importantly it is only from a single slide. Can not be a collection of slides.
            from_layer (str): The layer to compute the adaptive median filter from.
            to_layer (str): The layer to store the results of the adaptive median filter.
            n_hops (int): The maximum number of concentric rings in the graph to take into account to compute the median. Analogous to the max window size.
            hex_geometry (bool): Whether the graph is hexagonal or not. If True, then the graph is hexagonal. If False, then the graph is a grid. Only
                                 true for visium datasets.

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
    """ Perform batch correction with ComBat

    Compute batch correction using the `pycombat <https://github.com/epigenelabs/pyComBat?tab=readme-ov-file>`_ package. The batches are defined by ``adata.obs[batch_key]`` so
    the user can define which variable to use as batch identifier. The input data for the batch correction is ``adata.layers[from_layer]`` and the output is stored in
    ``adata.layers[to_layer]``.

    Args:
        adata (ad.AnnData): The AnnData object to transform. Must have logarithmically transformed data in ``adata.layers[from_layer]``.
        batch_key (str): The column in ``adata.obs`` that defines the batches.
        from_layer (str): The layer to take the data from.
        to_layer (str): The layer to store the results of the transformation.

    Returns:
        ad.AnnData: The updated AnnData object with batch corrected data in ``adata.layers[to_layer]``.
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
    """ Get expression deltas from the mean.

    Compute the deviations from the mean expression of each gene in ``adata.layers[from_layer]`` and save them
    in ``adata.layers[to_layer]``. Also add the mean expression of each gene to ``adata.var[f'{from_layer}_avg_exp']``.
    Average expression is computed using only train data determined by the ``adata.obs['split']`` column. However, deltas
    are computed for all observations.

    Args:
        adata (ad.AnnData): The AnnData object to update. Must have expression values in ``adata.layers[from_layer]``. Must also have the ``adata.obs['split']`` column with ``'train'`` values.
        from_layer (str): The layer to take the data from.
        to_layer (str): The layer to store the results of the transformation.

    Returns:
        ad.AnnData: The updated AnnData object with the deltas (``adata.layers[to_layer]``) and mean expression (``adata.var[f'{from_layer}_avg_exp']``) information.
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

def compute_moran(adata: ad.AnnData, from_layer: str, hex_geometry: bool) -> ad.AnnData:
    """Compute Moran's I statistic for each gene.

    Compute average Moran's I statistic for a collection of slides. Internally cycles over each slide in the ``adata`` collection
    and computes the Moran's I statistic for each gene. After that, it averages the Moran's I for each gene across all
    slides and saves it in ``adata.var[f'{from_layer}_moran']``.The input data for the Moran's I computation is ``adata.layers[from_layer]``.

    Args:
        adata (ad.AnnData): The AnnData object to update. Must have expression values in ``adata.layers[from_layer]``.
        from_layer (str): The key in ``adata.layers`` with the values used to compute Moran's I.
        hex_geometry (bool): Whether the geometry is hexagonal or not. This is used to compute the spatial neighbors before computing Moran's I. Only ``True`` for visium datasets.

    Returns:
        ad.AnnData: The updated AnnData object with the average Moran's I for each gene in ``adata.var[f'{from_layer}_moran']``.
    """

    print(f'Computing Moran\'s I for each gene over each slide using data of layer {from_layer}...')

    # Get the unique slide_ids
    slide_ids = adata.obs['slide_id'].unique()

    # Create a dataframe to store the Moran's I for each slide
    moran_df = pd.DataFrame(index = adata.var.index, columns=slide_ids)

    # Cycle over each slide
    for slide in slide_ids:
        # Get the annData for the current slide
        slide_adata = get_slide_from_collection(adata, slide)
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

# TODO: Fix documentation when the internal fixme is solved (VERIFICATION REQUIRED)
def filter_by_moran(adata: ad.AnnData, n_keep: int, from_layer: str) -> ad.AnnData:
    """ Filter prediction genes by Moran's I.

    This function filters the genes in ``adata.var`` by the Moran's I statistic. It keeps the ``n_keep`` genes with the highest Moran's I.
    The Moran's I values will be selected from ``adata.var[f'{from_layer}_moran']`` which must be already present in the ``adata``.
    If ``n_keep <= 0``, it means the number of genes is no specified and wee proceed to automatically compute it in the following way:
    
        a. If ``adata.n_vars > 320`` then ``n_keep = 128``.
        b. else, ``n_keep = 32``. 

    Args:
        adata (ad.AnnData): The AnnData object to update. Must have ``adata.var[f'{from_layer}_moran']`` column.
        n_keep (int): The number of genes to keep. I less than ``0`` the number of genes to keep is computed automatically.
        from_layer (str): Layer for which the Moran's I was already computed (``adata.var[f'{from_layer}_moran']``).

    Returns:
        ad.AnnData: The updated AnnData object with the filtered genes.
    """

    # Assert that the number of genes is at least n_keep
    assert adata.n_vars >= n_keep, f'The number of genes in the AnnData object is {adata.n_vars}, which is less than n_keep ({n_keep}).'

    # FIXME: This part is weird, we can define a simple threshold without all the computation
    # Select amount of top genes depending on the available amount if n_keep is not specified (Verify: n_keep would be equal to n_vars?)
    # threshold: (calcular el threshold segun np.abs(n_keep - 128) > np.abs(n_keep - 32)) --> 320
    if n_keep <= 0:
        n_keep = adata.n_vars
        """
        n_keep = round(adata.n_vars * 0.25, 0)
        if np.abs(n_keep - 128) > np.abs(n_keep - 32):
            n_keep = 32
        else:
            n_keep = 128
        """

    print(f"Filtering genes by Moran's I. Keeping top {n_keep} genes.")
    
    # Sort the genes by Moran's I
    sorted_genes = adata.var.sort_values(by=f'{from_layer}_moran', ascending=False).index

    # Get genes to keep list
    genes_to_keep = list(sorted_genes[:n_keep])

    # Filter the genes andata object
    adata = adata[:, genes_to_keep]

    # Return the updated AnnData object
    return adata

# TODO: Fix documentation when the internal fixme is solved. (NOT NECESARRY) 
# Another option is to put this function inside the initial preprocessing.
# TODO: When cheking this function on its own, try using a prediction layer "delta"
def add_noisy_layer(adata: ad.AnnData, prediction_layer: str) -> ad.AnnData:
    """ Add an artificial noisy layer.
    This function should only be used for experimentation/ablation purposes. It adds a noisy layer to the adata object by the name of 'noisy_d'
    or 'noisy' depending on the prediction layer. The noisy layer is created by returning the missing values to an already denoised layer of the adata.
    In the case the source layer is on logarithmic scale, the noisy layer is created by assigning zero values to the missing values. In the case the source
    layer is on delta scale, the noisy layer is created by assigning the negative mean of the gene to the missing values. missing values are specified by
    the binary ada.layers['mask'] layer.

    Args:
        adata (ad.AnnData): The AnnData object to update. Must have the prediction layer, the gene means if its a delta layer, and the mask layer.
        prediction_layer (str): The layer that will be corrupted to create the noisy layer.

    Returns:
        ad.AnnData: The updated AnnData object with the noisy layer added.
    """
    if 'delta' in prediction_layer or 'noisy_d' in prediction_layer:
    # FIXME: make it flexible for other prediction layers different to c_d_log1p (DONE)
        # Get vector with gene means
        gene_means = adata.var[f"{prediction_layer}_avg_exp"].values 
        # Expand gene means to the shape of the layer
        gene_means = np.repeat(gene_means.reshape(1, -1), adata.n_obs, axis=0)
        # Get valid mask
        valid_mask = adata.layers['mask']
        # Initialize noisy deltas
        noisy_deltas = -gene_means 
        delta_key = prediction_layer.split("log1p")
        # Assign delta values in positions where valid mask is true
        noisy_deltas[valid_mask] = adata.layers[f'{delta_key[0]}deltas'][valid_mask]
        # Add the layer to the adata
        adata.layers['noisy_d'] = noisy_deltas

        # Add a var column of used means of the layer
        mean_key = f'{prediction_layer}_avg_exp'
        adata.var['used_mean'] = adata.var[mean_key]

    else:
        # Copy the cleaned layer to the layer noisy
        noised_layer = adata.layers[prediction_layer].copy()
        # Get zero mask
        zero_mask = ~adata.layers['mask']
        # Zero out the missing values
        noised_layer[zero_mask] = 0
        # Add the layer to the adata
        adata.layers['noisy'] = noised_layer

    # Give warning to say that the noisy layer is being used
    print('Using noisy_delta layer for training. This will probably yield bad results.')

    return adata

# FIXME: hex_geometry default set to True (?). Check if this is needed.
# FIXME: Shouldn't dataset be inside the param_dict and not a parameter (DONE)
# FIXME: Maybe the organism should be a key of the param_dict (DONE)
# FIXME: Hex geometry should also be inside the param_dict (How do I define this for every dataset?) --> True siempre
# TODO: Fix documentation when the above fixmes are solved.
def process_dataset(adata: ad.AnnData, param_dict: dict, hex_geometry: bool = True) -> ad.AnnData:
    """ Perform complete processing pipeline.
    This function performs the complete processing pipeline for a dataset. It only computes over the expression values of the dataset
    (adata.X). The processing pipeline is the following:

        1. Normalize the data with tpm normalization (tpm layer)
        2. Transform the data with log1p (log1p layer)
        3. Denoise the data with the adaptive median filter (d_log1p layer)
        4. Compute moran I for each gene in each slide and average moranI across slides (add results to .var['d_log1p_moran'])
        5. Filter dataset to keep the top param_dict['top_moran_genes'] genes with highest moran I.
        6. Perform ComBat batch correction if specified by the 'combat_key' parameter (c_d_log1p layer)
        7. Compute the deltas from the mean for each gene (computed from log1p, d_log1p and c_log1p, c_d_log1p layer if batch correction was performed)
        8. Add a binary mask layer specifying valid observations for metric computation (mask layer, True for valid observations, False for missing values).

    Args:
        adata (ad.AnnData): The AnnData object to process. Should be already filtered with the filter_dataset() function.
        param_dict (dict): Dictionary that contains filtering and processing parameters. Keys that must be present are:

                            - 'top_moran_genes': (int) The number of genes to keep after filtering by Moran's I. If set to 0, then the number of genes is internally computed.
                            - 'combat_key': (str) The column in adata.obs that defines the batches for ComBat batch correction. If set to 'None', then no batch correction is performed.
        
        hex_geometry (bool): Whether the graph is hexagonal or not. If True, then the graph is hexagonal. If False, then the graph is a grid. Only true for visium datasets.

    Returns:
        ad.Anndata: The processed AnnData object with all the layers and results added. A list of includded layers in adata.layers is:

                    - 'counts': Raw counts of the dataset.
                    - 'tpm': TPM normalized data.
                    - 'log1p': Log1p transformed data (base 2.0).
                    - 'd_log1p': Denoised data with adaptive median filter.
                    - 'c_log1p': Batch corrected data with ComBat (only if combat_key is not 'None').
                    - 'c_d_log1p': Batch corrected and denoised data with adaptive median filter (only if combat_key is not 'None').
                    - 'deltas': Deltas from the mean expression for log1p.
                    - 'd_deltas': Deltas from the mean expression for d_log1p.
                    - 'c_deltas': Deltas from the mean expression for c_log1p (only if combat_key is not 'None').
                    - 'c_d_deltas': Deltas from the mean expression for c_d_log1p (only if combat_key is not 'None').
                    - 'mask': Binary mask layer. True for valid observations, False for imputed missing values.
    """

    ### Compute all the processing steps
    # NOTE: The d prefix stands for denoised
    # NOTE: The c prefix stands for combat corrected

    # Start the timer and print the start message
    start = time()
    print('Starting data processing...')

    # First add raw counts to adata.layers['counts']
    adata.layers['counts'] = adata.X.toarray()
    
    # 1. Make TPM normalization
    adata = tpm_normalization(param_dict["organism"], adata, from_layer='counts', to_layer='tpm')

    # 2. Transform the data with log1p (base 2.0)
    adata = log1p_transformation(adata, from_layer='tpm', to_layer='log1p')

    # 3. Denoise the data with adaptive median filter
    adata = clean_noise(adata, from_layer='log1p', to_layer='d_log1p', n_hops=4, hex_geometry=hex_geometry)

    # 4. Compute average moran for each gene in the layer d_log1p 
    adata = compute_moran(adata, hex_geometry=hex_geometry, from_layer='d_log1p')

    # 5. Filter genes by Moran's I
    adata = filter_by_moran(adata, n_keep=param_dict['top_moran_genes'], from_layer='d_log1p')

    # 6. If combat key is specified, apply batch correction
    if param_dict['combat_key'] != 'None':
        adata = combat_transformation(adata, batch_key=param_dict['combat_key'], from_layer='log1p', to_layer='c_log1p')
        adata = combat_transformation(adata, batch_key=param_dict['combat_key'], from_layer='d_log1p', to_layer='c_d_log1p')

    # 7. Compute deltas and mean expression for all log1p, d_log1p, c_log1p and c_d_log1p
    adata = get_deltas(adata, from_layer='log1p', to_layer='deltas')
    adata = get_deltas(adata, from_layer='d_log1p', to_layer='d_deltas')
    
    if param_dict['combat_key'] != 'None':
        adata = get_deltas(adata, from_layer='c_log1p', to_layer='c_deltas')
        adata = get_deltas(adata, from_layer='c_d_log1p', to_layer='c_d_deltas')

    # 8. Add a binary mask layer specifying valid observations for metric computation
    adata.layers['mask'] = adata.layers['tpm'] != 0
    
    # Print with the percentage of the dataset that was replaced
    print('Percentage of imputed observations with median filter: {:5.3f}%'.format(100 * (~adata.layers["mask"]).sum() / (adata.n_vars*adata.n_obs)))

    # Print the number of cells and genes in the dataset
    print(f'Processing of the data took {time() - start:.2f} seconds')
    print(f'The processed dataset looks like this:')
    print(adata)
    
    return adata


### Patch processing function:

# FIXME: keep patch_scale as parameter? NOP
# TODO: assert: que exista la escala
# TODO: assert solo una llave que corresponda a patch_scale 
# TODO: Fix documentation when the above fixme is solved. (DONE)
def compute_patches_embeddings_and_predictions(adata: ad.AnnData, backbone: str ='densenet', model_path:str="None", preds: bool=True, patch_size: int = 224) -> None:
    """ Compute embeddings or predictions for patches.

    This function computes embeddings or predictions for a given backbone model and adata object. It can optionally
    compute using a stored model in model_path or a pretrained model from pytorch. The embeddings or predictions are
    stored in adata.obsm[f'embeddings_{backbone}'] or adata.obsm[f'predictions_{backbone}'] respectively. The patches
    must be stored in adata.obsm[f'patches_scale_{patch_scale}'] and must be of shape (n_patches, patch_size*patch_size*3).

    The function only modifies the AnnData object in place.

    Args:
        adata (ad.AnnData): The AnnData object to process.
        backbone (str, optional): A string specifiying the backbone model to use. Must be one of the following ['resnet', 'resnet50', 'ConvNeXt', 'EfficientNetV2', 'InceptionV3', 'MaxVit', 'MobileNetV3', 'ResNetXt', 'ShuffleNetV2', 'ViT', 'WideResnet', 'densenet', 'swin']. Defaults to 'densenet'.
        model_path (str, optional): The path to a stored model. If set to 'None', then a pretrained model is used. Defaults to "None".
        preds (bool, optional): True if predictions are to be computed, False if embeddings are to be computed. Defaults to True.
        patch_size (int, optional): The size of the patches. Defaults to 224.

    Raises:
        ValueError: If the backbone is not supported.
    """

    # Define a cuda device if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Define the model
    model = im_encoder.ImageEncoder(backbone=backbone, use_pretrained=True, latent_dim=adata.n_vars, patch_scale=1.0)

    if model_path != "None":
        saved_model = torch.load(model_path)
        # Check if state_dict is inside a nested dictionary
        if 'state_dict' in saved_model.keys():
            saved_model = saved_model['state_dict']

        model.load_state_dict(saved_model)
    
    # Define the weights for the model depending on the backbone
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

    # Pass model to device and put in eval mode
    model.to(device)
    model.eval()

    # Perform specific preprocessing for the model
    preprocess = weights.transforms(antialias=True)

    # Get the patches
    patch_scale = 1.0
    flat_patches = adata.obsm[f'patches_scale_{patch_scale}']

    # Reshape all the patches to the original shape
    all_patches = flat_patches.reshape((-1, patch_size, patch_size, 3))
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

    # Pass embeddings or predictions to cpu and add to data.obsm
    if preds:
        adata.obsm[f'predictions_{backbone}'] = outputs.cpu().numpy()
    else:
        adata.obsm[f'embeddings_{backbone}'] = outputs.cpu().numpy()


### Adata dataloader building function:

# TODO: Fix the internal fixme (DISCUSS AGAIN)
def get_pretrain_dataloaders(adata: ad.AnnData, layer: str = 'c_d_log1p', batch_size: int = 128, shuffle: bool = True, use_cuda: bool = False) -> Tuple[AnnLoader, AnnLoader, AnnLoader]:
    """ Get dataloaders for pretraining an image encoder.
    This function returns the dataloaders for training an image encoder. This means training a purely vision-based model on only
    the patches to predict the gene expression of the patches.

    Dataloaders are returned as a tuple, if there is no test set for the dataset, then the test dataloader is None.

    Args:
        adata (ad.AnnData): The AnnData object that will be processed.
        layer (str, optional): The layer to use for the pre-training. The adata.X will be set to that of 'layer'. Defaults to 'deltas'.
        batch_size (int, optional): The batch size of the loaders. Defaults to 128.
        shuffle (bool, optional): Whether to shuffle the data in the loaders. Defaults to True.
        use_cuda (bool, optional): True for using cuda in the loader. Defaults to False.

    Returns:
        Tuple[AnnLoader, AnnLoader, AnnLoader]: The train, validation and test dataloaders. If there is no test set, the test dataloader is None.
    """
    # Get the sample indexes for the train, validation and test sets
    idx_train, idx_val, idx_test = adata.obs[adata.obs.split == 'train'].index, adata.obs[adata.obs.split == 'val'].index, adata.obs[adata.obs.split == 'test'].index

    ##### Addition to handle noisy training #####

    # FIXME: Put this in a part of the complete processing pipeline instead of the dataloader function. 
    # Handle noisy training
    # Add this function in procces_data function and automaticaaly generate noisy layers for this layers:
    # c_d_log1p, c_t_log1p, c_d_deltas, c_t_deltas
    adata = add_noisy_layer(adata=adata, prediction_layer=layer)

    # Set the X of the adata to the layer casted to float32
    adata.X = adata.layers[layer].astype(np.float32)

    imp_model_str = 'transformer model' if layer in ['c_t_log1p', 'c_t_deltas'] else 'median filter'

    # Print with the percentage of the dataset that was replaced
    imp_pct = 100 * (~adata.layers["mask"]).sum() / (adata.n_vars*adata.n_obs)
    print('Percentage of imputed observations with {}: {:5.3f}%'.format(imp_model_str, imp_pct))

    # If the prediction layer is some form of deltas, add the used mean of the layer as a column in the var
    if 'deltas' in layer:
        # Add a var column of used means of the layer
        mean_key = f'{layer}_avg_exp'.replace('deltas', 'log1p')
        adata.var['used_mean'] = adata.var[mean_key]

    # Subset the global data handle also the possibility that there is no test set
    adata_train, adata_val = adata[idx_train, :], adata[idx_val, :]
    adata_test = adata[idx_test, :] if len(idx_test) > 0 else None

    # Declare dataloaders
    train_dataloader = AnnLoader(adata_train, batch_size=batch_size, shuffle=shuffle, use_cuda=use_cuda)
    val_dataloader = AnnLoader(adata_val, batch_size=batch_size, shuffle=shuffle, use_cuda=use_cuda)
    test_dataloader = AnnLoader(adata_test, batch_size=batch_size, shuffle=shuffle, use_cuda=use_cuda) if adata_test is not None else None

    return train_dataloader, val_dataloader, test_dataloader

# funcion de complete with spackle para que queden las capas de transformers (REVISAR COMO HACER ESTO)
# PARAMETROS SEGUN LO QUE FUNCIONÓ MEJOR PARA LA MAYORIA
### Graph building functions:

def get_graphs_one_slide(adata: ad.AnnData, n_hops: int, layer: str, hex_geometry: bool) -> Tuple[dict,int]:
    """ Get neighbor graphs for a single slide.
    This function receives an AnnData object with a single slide and for each node computes the graph in an
    n_hops radius in a pytorch geometric format. The AnnData object must have both embeddings and predictions in the
    adata.obsm attribute.

    It returns a dictionary where the patch names are the keys and a pytorch geometric graph for each one as
    values. NOTE: The first node of every graph is the center.

    Args:
        adata (ad.AnnData): The AnnData object with the slide data.
        n_hops (int): The number of hops to compute the graph.
        layer (str): The layer of the graph to predict. Will be added as y to the graph.
        hex_geometry (bool): Whether the slide has hexagonal geometry or not.

    Returns:
        Tuple(dict,int)
        dict: A dictionary where the patch names are the keys and pytorch geometric graph for each one as values. The first node of every graph is the center.
        int: Max column or row difference between the center and the neighbors. Used for positional encoding.                   
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
    layers_dict = {key: torch.from_numpy(adata.layers[key]).type(torch.float32) for key in adata.layers.keys()} # Get global layers
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
    used_mean = torch.from_numpy(adata.var[f'{layer}_avg_exp'.replace('deltas', 'log1p')].values).type(torch.float32) if 'deltas' in layer else None

    # Define the empty graph dict
    graph_dict = {}
    max_abs_d_pos=-1

    # Cycle over each obs
    for i in tqdm(range(len(neighbors_dict_index)), leave=False, position=1):
        central_node_name = index_to_obs[i]                                                 # Get the name of the central node
        curr_nodes_idx = torch.tensor(neighbors_dict_index[i])                              # Get the indexes of the nodes in the graph
        curr_adj_matrix = matrices_dict[central_node_name]                                  # Get the adjacency matrix of the graph (precomputed)
        curr_edge_index, _ = from_scipy_sparse_matrix(curr_adj_matrix)                      # Get the edge index and edge attribute of the graph
        curr_layers = {key: layers_dict[key][curr_nodes_idx] for key in layers_dict.keys()} # Get the layers of the graph filtered by the nodes
        curr_pos = pos[curr_nodes_idx]                                                      # Get the positions of the nodes in the graph
        curr_d_pos = curr_pos - curr_pos[0]                                                 # Get the relative positions of the nodes in the graph

        # Define the graph
        graph_dict[central_node_name] = geo_Data(
            y=curr_layers[layer],
            edge_index=curr_edge_index,
            pos=curr_pos,
            d_pos=curr_d_pos,
            embeddings=embeddings[curr_nodes_idx],
            predictions=predictions[curr_nodes_idx] if predictions is not None else None,
            used_mean=used_mean if used_mean is not None else None,
            num_nodes=len(curr_nodes_idx),
            mask=layers_dict['mask'][curr_nodes_idx]
        )

        max_curr_d_pos=curr_d_pos.abs().max()
        if max_curr_d_pos>max_abs_d_pos:
            max_abs_d_pos=max_curr_d_pos

    #cast as int
    max_abs_d_pos=int(max_abs_d_pos)
    
    # Return the graph dict
    return graph_dict, max_abs_d_pos


def get_sin_cos_positional_embeddings(graph_dict: dict, max_d_pos: int) -> dict:
    """ Get positional encodings for a neighbor graph.
    This function adds a transformer-like positional encodings to each graph in a graph dict. It adds the positional
    encodings under the attribute 'positional_embeddings' for each graph. 

    Args:
        graph_dict (dict): A dictionary where the patch names are the keys and a pytorch geometric graphs for each one are values.
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


def get_graphs(adata: ad.AnnData, n_hops: int, layer: str, hex_geometry: bool=True) -> dict:
    """ Get graphs for all the slides in a dataset.
    This function wraps the get_graphs_one_slide function to get the graphs for all the slides in the dataset.
    After computing the graph dicts for each slide it concatenates them into a single dictionary which is then used to compute
    the positional embeddings for each graph.

    For details see get_graphs_one_slide and get_sin_cos_positional_embeddings functions.

    Args:
        adata (ad.AnnData): The AnnData object used to build the graphs.
        n_hops (int): The number of hops to compute each graph.
        layer (str): The layer of the graph to predict. Will be added as y to the graph.
        hex_geometry (bool): Whether the graph is hexagonal or not. Only true for visium datasets. Defaults to True.

    Returns:
        dict: A dictionary where the spots' names are the keys and pytorch geometric graphs are values.
    """

    print('Computing graphs...')

    # Get unique slide ids
    unique_ids = adata.obs['slide_id'].unique()

    # Global dictionary to store the graphs (pytorch geometric graphs)
    graph_dict = {}
    max_global_d_pos=-1

    # Iterate through slides
    for slide in tqdm(unique_ids, leave=True, position=0):
        curr_adata = get_slide_from_collection(adata, slide)
        curr_graph_dict, max_curr_d_pos = get_graphs_one_slide(curr_adata, n_hops, layer, hex_geometry)
        
        # Join the current dictionary to the global dictionary
        graph_dict = {**graph_dict, **curr_graph_dict}

        if max_curr_d_pos>max_global_d_pos:
            max_global_d_pos=max_curr_d_pos
    
    graph_dict = get_sin_cos_positional_embeddings(graph_dict, max_global_d_pos)

    # Return the graph dict
    return graph_dict

# TODO: Fix the internal fixme (DEPENDS ON THE PREVIOUS DISCUSSION)
def get_graph_dataloaders(adata: ad.AnnData, dataset_path: str='', layer: str = 'c_t_log1p', n_hops: int = 2, backbone: str ='None', model_path: str = "best_stnet.pt", batch_size: int = 128, 
                          shuffle: bool = True, hex_geometry: bool=True, patch_size: int=224, patch_scale: float=1.0) -> Tuple[geo_DataLoader, geo_DataLoader, geo_DataLoader]:
    """ Get dataloaders for the graphs of a dataset.
    This function performs all the pipeline to get graphs dataloaders for a dataset. It does the following steps:

        1. Computes embeddings and predictions for the patches using the specified backbone and model.
        2. Computes the graph dictionaries for the dataset using the embeddings and predictions.
        3. Saves the graphs in the dataset_path folder.
        4. Returns the train, validation and test dataloaders for the graphs.
    
    The function also checks if the graphs are already saved in the dataset_path folder. If they are, it loads them instead of recomputing them. In case 
    the dataset has no test set, the test dataloader is set to None.

    Args:
        adata (ad.AnnData): The AnnData object to process.
        dataset_path (str, optional): The path to the dataset (where the graphs will be stored). Defaults to ''.
        layer (str, optional): Layer to predict. Defaults to 'c_t_log1p'.
        n_hops (int, optional): Number of hops to compute the graph. Defaults to 2.
        backbone (str, optional): Backbone model to use. Defaults to 'densenet'.
        model_path (str, optional): Path to the model to use. Defaults to "None".
        batch_size (int, optional): Batch size of the dataloaders. Defaults to 128.
        shuffle (bool, optional): Whether to shuffle the data in the dataloaders. Defaults to True.
        hex_geometry (bool, optional): Whether the graph is hexagonal or not. Defaults to True.
        patch_size (int, optional): Size of the patches. Defaults to 224.
        patch_scale (float, optional): Scale of the patches. Defaults to 1.0.

    Returns:
        Tuple[geo_DataLoader, geo_DataLoader, geo_DataLoader]: _description_
    """
    # Get dictionary of parameters to get the graphs
    curr_graph_params = {
        'n_hops': n_hops,
        'layer': layer,
        'backbone': backbone,
        'model_path': model_path
    }        

    # Create graph directory if it does not exist
    os.makedirs(os.path.join(dataset_path, 'graphs'), exist_ok=True)
    # Get the filenames of the parameters of all directories in the graph folder
    filenames = glob.glob(os.path.join(dataset_path, 'graphs', '**', 'graph_params.json' ), recursive=True)

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

        # FIXME: Again this should be in the processing part and not in the dataloader
        adata = add_noisy_layer(adata=adata, prediction_layer=layer)

        # We compute the embeddings and predictions for the patches
        compute_patches_embeddings_and_predictions(preds=True, backbone=backbone, model_path=model_path, patch_size=patch_size, patch_scale=patch_scale)
        compute_patches_embeddings_and_predictions(preds=False, backbone=backbone, model_path=model_path, patch_size=patch_size, patch_scale=patch_scale)
        
        # Get graph dicts
        general_graph_dict = get_graphs(adata=adata, n_hops=n_hops, layer=layer, hex_geometry=hex_geometry)

        # Get the train, validation and test indexes
        idx_train, idx_val, idx_test = adata.obs[adata.obs.split == 'train'].index, adata.obs[adata.obs.split == 'val'].index, adata.obs[adata.obs.split == 'test'].index

        # Get list of graphs
        train_graphs = [general_graph_dict[idx] for idx in idx_train]
        val_graphs = [general_graph_dict[idx] for idx in idx_val]
        test_graphs = [general_graph_dict[idx] for idx in idx_test] if len(idx_test) > 0 else None

        print('Saving graphs...')
        # Create graph directory if it does not exist with the current time
        graph_dir = os.path.join(dataset_path, 'graphs', datetime.now().strftime("%d-%m-%Y-%H-%M-%S"))
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

# Get the nearest neighbors dist and ids and add this information in the adata
def get_nn_images(adata) -> None:
    
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
    train_subset = adata[adata.obs['split'] == 'train']
    val_subset = adata[adata.obs['split'] == 'val']
    test_subset = adata[adata.obs['split'] == 'test']
    
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
    all_distances_df = all_distances_df.reindex(adata.obs.index.values)
    all_ids_df = all_ids_df.reindex(adata.obs.index.values)
    
    # Add the dataframes to the obsm
    adata.obsm['image_nn_distances'] = all_distances_df
    adata.obsm['image_nn_ids'] = all_ids_df

    end = time()
    print(f'Finished getting image nearest neighbors in {end - start:.2f} seconds')

# Obtain ResNet50 embedding and add this information in the adata
def obtain_embeddings_resnet50(adata, patch_scale: float = 1.0, patch_size: int = 224):

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
    flat_patches = adata.obsm[f'patches_scale_{patch_scale}']

    # Reshape all the patches to the original shape
    all_patches = flat_patches.reshape((-1, patch_size, patch_size, 3))
    torch_patches = torch.from_numpy(all_patches).permute(0, 3, 1, 2).float()    # Turn all patches to torch
    rescaled_patches = egn_transforms(torch_patches / 255)                       # Rescale patches to [0, 1]

    img_embedding = [extract(resnet_encoder, single_patch.unsqueeze(dim=0).to(device)) for single_patch in tqdm(rescaled_patches)]
    img_embedding = torch.cat(img_embedding).contiguous()             
    
    adata.obsm['resnet50_embeddings'] = img_embedding.cpu().numpy()