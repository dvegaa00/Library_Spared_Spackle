import anndata as ad
import numpy as np
# FIXME: merge get_metrics functions from metrics.py and metrics_ids
from metrics_ids import get_metrics
import matplotlib
import matplotlib.pyplot as plt
import squidpy as sq
import wandb

def log_pred_image(adata: ad.AnnData, n_genes: int = 3, slide = ""):
    """
    This function receives an adata with the prediction layers of the median imputation model and transformer
    imputation model and plots the visualizations to compare the performance of both methods.

    Args:
        adata (ad.AnnData): adata containing the predictions, masks and groundtruth of the imputations methods.
        n_genes (int, optional): number of genes to plot (top and bottom genes).
        slide (str, optional): slide to plot. If none is given it plots the first slide of the adata.
    """
    # Get prediction and groundtruth layers from layers keys in the anndata
    # Define prediction layers
    pred_layer = [l for l in adata.layers.keys() if 'predictions' in l]
    trans_pred_layer = pred_layer[0] if "transformer" in pred_layer[0] else pred_layer[1]
    med_pred_layer = pred_layer[1] if "median" in pred_layer[1] else pred_layer[0]
    # Define groundtruth layer
    gt_layer = trans_pred_layer.split(',')[1]
    # Define input and random mask
    input_mask_layer = "masked_map"
    random_mask_layer = "random_mask"

    # Handle delta prediction in normal scale
    if 'deltas' in gt_layer:
        gt_layer = gt_layer.replace('deltas', 'log1p')
    # Be sure the prediction layer and gt layer is present in dataset
    assert not (pred_layer is None), 'predictions layer not present in the adata'
    assert not (gt_layer is None), 'groundtruth layer not present in the adata'

    # Get detailed metrics from partition (change layer "mask" for mask_layer)
    for p in pred_layer:
        name_model = p.split(',')[2]
        detailed_metrics = get_metrics(
            gt_mat = adata.to_df(layer=gt_layer).values, #layer=f"{gt_layer},{name_model}"
            pred_mat = adata.to_df(layer=p).values,
            mask = adata.to_df(layer="random_mask").values, #layer=f"mask,{name_model}"
            detailed=True
        ) 
        
        # Add global metrics to adata
        adata.var[f'{name_model}_pcc_global'] = detailed_metrics['PCC-Gene']
        adata.var[f'{name_model}_mse_global'] = detailed_metrics['MSE']
        # Add detalied metrics to adata
        adata.var[f'{name_model}_pcc_test'] = detailed_metrics['detailed_PCC-Gene']
        adata.var[f'{name_model}_mse_test'] = detailed_metrics['detailed_mse_gene']
        
        if name_model == "transformer":
            gene_len = len(detailed_metrics['detailed_PCC-Gene'])
    
    # Get selected genes based on the best and worst pcc
    selected_genes = []
    n_top = adata.var.nlargest(gene_len, columns='transformer_pcc_test').index.to_list()
    n_bottom = adata.var.nsmallest(gene_len, columns='transformer_pcc_test').index.to_list()
    
    # Get the selected slides. NOTE: Only first slide is always selected in case slides is not specified by parameter.
    if slide == "":
        slide = list(adata.obs.slide_id.unique())[0]
    
    # Get adata for slide
    slide_adata = adata[adata.obs['slide_id'] == slide].copy()
    # Modify the uns dictionary to include only the information of the slide
    slide_adata.uns['spatial'] = {slide: adata.uns['spatial'][slide]}
    
    # Takes top and worst genes that contain at leats 10% of missing spots
    top_genes = []
    bottom_genes = []
    print('Extracting top and bottom performing genes ...')
    for g in n_top: 
        gene_adata = slide_adata[:,g].copy()
        true_values = np.count_nonzero(gene_adata.layers[random_mask_layer])
        if true_values > (adata.shape[0]*0.05):
            top_genes.append(g)
        if len(top_genes) == n_genes:
            break
    
    for g in n_bottom: 
        gene_adata = slide_adata[:,g].copy()
        true_values = np.count_nonzero(gene_adata.layers[random_mask_layer])
        if true_values > (adata.shape[0]*0.05):
            bottom_genes.append(g)
        if len(bottom_genes) == n_genes:
            break

    # Best and worst genes to plot
    selected_genes.append(top_genes)
    selected_genes.append(bottom_genes)

    # TODO: finish documentation for the log_genes_for_slides function
    def log_genes_for_slide(top_bottom, genes, slide_adata, gt_layer, trans_pred_layer, med_pred_layer, random_mask_layer, input_mask_layer):
        """
        This function receives a slide adata and the names of the prediction, groundtruth and masking layers 
        and logs the visualizations for the top and bottom genes

        Args:
            genes (list): genes to visualize
            gene_to_df (dictionary): dictionary of genes and genes in dataframe that correspond to accurate metrics
            slide_adata (AnnData): slide AnnData
            gt_layer (str): name of groundtruth layer
            trans_pred_layer (str): name of the prediction layer of the transformer imputation model
            med_pred_layer (str): name of the prediction layer of the median imputation model
            random_mask_layer (str): name of the random mask
            input_mask_layer (str): name of the input mask for visualizations

        """
        # Get the slide
        slide = list(slide_adata.obs.slide_id.unique())[0]
        # Replace 0 with nan to plot mask in visualizations (nans and black dots)
        slide_adata.layers[input_mask_layer][slide_adata.layers[input_mask_layer]==0] = np.nan
        # Define order of rows in dict
        order_dict = {}
        for i, gene in enumerate(genes):
            order_dict[gene] = i
  
        # Declare figure
        fig, ax = plt.subplots(nrows=len(genes), ncols=5, layout='constrained')
        fig.set_size_inches(25, 5 * len(genes))

        # Iterate over the genes
        for g in genes:         
            # Get current row
            row = order_dict[g]
            # Get min and max of the selected top genes in the slide
            gene_min_pred_1 = slide_adata[:, g].layers[trans_pred_layer].min() 
            gene_max_pred_1 = slide_adata[:, g].layers[trans_pred_layer].max()
            
            gene_min_pred_2 = slide_adata[:, g].layers[med_pred_layer].min() 
            gene_max_pred_2 = slide_adata[:, g].layers[med_pred_layer].max() 
            
            gene_min_gt = slide_adata[:, g].layers[gt_layer].min() 
            gene_max_gt = slide_adata[:, g].layers[gt_layer].max() 

            gene_min_mask = np.nanmin(slide_adata[:, g].layers[input_mask_layer])
            gene_max_mask = slide_adata[:, g].layers[input_mask_layer].max()
            
            gene_min = min([gene_min_pred_1, gene_min_pred_2, gene_min_gt, gene_min_mask])
            gene_max = max([gene_max_pred_1, gene_max_pred_2, gene_max_gt, gene_max_mask])

            # Define color normalization
            norm = matplotlib.colors.Normalize(vmin=gene_min, vmax=gene_max)
            # Set PCC
            pcc_trans = str(round(slide_adata.var["transformer_pcc_test"][g], 3))
            pcc_med = str(round(slide_adata.var["median_pcc_test"][g], 3))
            # Set MSE
            mse_trans = str(round(slide_adata.var["transformer_mse_test"][g], 3))
            mse_med = str(round(slide_adata.var["median_mse_test"][g], 3))
            # Plot gt and pred of gene in the specified slides
            sq.pl.spatial_scatter(slide_adata, color=[g], layer=input_mask_layer, ax=ax[row,0], cmap='jet', norm=norm, colorbar=False, na_color="black", title="")
            sq.pl.spatial_scatter(slide_adata, color=[g], layer=trans_pred_layer, ax=ax[row,1], cmap='jet', norm=norm, colorbar=False, title="")
            sq.pl.spatial_scatter(slide_adata, color=[g], layer=med_pred_layer, ax=ax[row,2], cmap='jet', norm=norm, colorbar=False, title="")
            sq.pl.spatial_scatter(slide_adata, color=[g], layer=gt_layer, ax=ax[row,3], cmap='jet', norm=norm, colorbar=True, title="")
            
            # Set titles
            ax[row, 1].set_title(f'PCC = {pcc_trans} & MSE = {mse_trans}', fontsize='xx-large')
            ax[row, 2].set_title(f'PCC = {pcc_med} & MSE = {mse_med}', fontsize='xx-large')
            
            # Set y labels
            ax[row,0].set_ylabel(f'{g}:\n{slide}\n', fontsize='xx-large')
            ax[row,0].set_xticks([])
            ax[row,0].set_yticks([])
            ax[row,1].set_ylabel('')
            ax[row,2].set_ylabel('')
            ax[row,3].set_ylabel('')
            
            # Set x labels 
            ax[row,0].set_xlabel('')
            ax[row,1].set_xlabel('')
            ax[row,2].set_xlabel('')
            ax[row,3].set_xlabel('')
            
            # Define gene adata
            gene_adata = slide_adata[:,g].copy()
            # Define models prediction and ground truth (only masked spots)
            ceros_gt_trans = [gene_adata.layers[gt_layer][gene_adata.layers[random_mask_layer]==True]][0]
            ceros_trans_pred = [gene_adata.layers[trans_pred_layer][gene_adata.layers[random_mask_layer]==True]][0]
            ceros_gt_med = [gene_adata.layers[gt_layer][gene_adata.layers[random_mask_layer]==True]][0]
            ceros_med_pred = [gene_adata.layers[med_pred_layer][gene_adata.layers[random_mask_layer]==True]][0]
            
            # Plot gen predictions and ground truth
            ax[row,4].plot(ceros_gt_trans, ceros_gt_trans, color="black", linestyle="-", label="Ground Truth")
            ax[row,4].plot(ceros_gt_trans, ceros_trans_pred, color="blue", marker="o",  markersize=3, linestyle="None", label=f"Transformer\nPCC = {pcc_trans} & MSE = {mse_trans}")
            ax[row,4].plot( ceros_gt_med, ceros_med_pred, color="red", marker="o",  markersize=3, linestyle="None", label=f"Median Filter\nPCC = {pcc_med} & MSE = {mse_med}")
            ax[row,4].legend(markerfirst=3, framealpha=0.4, loc="lower right")
            ax[row,4].set_xlabel("Ground Truth")
            ax[row,4].set_ylabel("Prediction")
        
        
        # Format figure
        for i, axis in enumerate(ax.flatten()):
            axis.spines['top'].set_visible(False)
            axis.spines['right'].set_visible(False)
            if ((i+1)%5) != 0:
                axis.spines['bottom'].set_visible(False)
                axis.spines['left'].set_visible(False)
        
        # Set PCC
        pcc_trans = str(round(slide_adata.var["transformer_pcc_test"][genes[0]], 3))
        pcc_med = str(round(slide_adata.var["median_pcc_test"][genes[0]], 3))
        # Set MSE
        mse_trans = str(round(slide_adata.var["transformer_mse_test"][genes[0]], 3))
        mse_med = str(round(slide_adata.var["median_mse_test"][genes[0]], 3))
        # Set titles
        ax[0, 0].set_title('Masked Map', fontsize='xx-large')
        ax[0, 1].set_title(f'Prediction Transformer\nPCC = {pcc_trans} & MSE = {mse_trans}', fontsize='xx-large')
        ax[0, 2].set_title(f'Prediction Median Filter\nPCC = {pcc_med} & MSE = {mse_med}', fontsize='xx-large')
        ax[0, 3].set_title('Ground Truth', fontsize='xx-large')
        ax[0, 4].set_title('Pred vs Trues', fontsize='xx-large')
        
        # Log plot 
        #wandb.log({top_bottom: fig})
        breakpoint()
        
    top_bottom = ["Top Genes", "Bottom Genes"]

    print('Logging visualization plots ...')
    for i, gene in enumerate(selected_genes):
        log_genes_for_slide(top_bottom=top_bottom[i],genes=gene, slide_adata=slide_adata, gt_layer=gt_layer, trans_pred_layer=trans_pred_layer, med_pred_layer=med_pred_layer, random_mask_layer=random_mask_layer, input_mask_layer=input_mask_layer)

    