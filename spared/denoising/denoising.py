import anndata as ad
from tqdm import tqdm
import numpy as np
import sys
import pathlib
# Get the path of the spared database
# SPARED_PATH = pathlib.Path(__file__).parent

# El path a spared es ahora diferente
SPARED_PATH = pathlib.Path(__file__).resolve().parent.parent

# Agregar el directorio padre al sys.path para los imports
sys.path.append(str(SPARED_PATH))
# Import im_encoder.py file
from spot_features import spot_features
# Remove the path from sys.path
sys.path.remove(str(SPARED_PATH))

#TODO: esto debe quedar como dos funciones, una que limpie por medio del filtro mediano y otra por medio del transformer
#clean noise limpia con medianas (ajustar)
#crear la de transformer
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
            curr_neighbors_dict = spot_features.get_spatial_neighbors(adata, i, hex_geometry)

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