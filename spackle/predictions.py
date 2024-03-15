import anndata as ad
from torch.utils.data import DataLoader
from tqdm import tqdm
from dataset import ImputationDataset
import numpy as np
import torch
import pandas as pd
import os

#TODO: Document
def get_predictions(adata: ad.AnnData, args, model, split_name, layer='c_d_log1p', method='transformer', device="cuda", save_path = '')->None:
    """
    This function receives a trained model and an adata for which it will predict the imputed gene expression matrix using a trained model.
    The predictions are saved in the original adata for visualization purposes.

    Args:
        adata (ad.AnnData): adata with the data split of interest.
        args (argparse): parser with the values necessary for data processing.
        model (model): model used to get imputation predictions.
        split_name (str): name of the split being processed.
        layer (str, optional): layer for prediction. Defaults to 'c_d_log1p'.
        method (str): imputation method to which the model corresponds. Either 'median' or 'transformer'.
        device (torch.device): device in which tensors will be processed.
        save_path (str, optional): path were a dictionary of the predictions, masked map, ground truth and median imputation results will be saved.
    """    
    # Prepare data 
    dataset = ImputationDataset(adata, args, split_name, pre_masked = True)
    # Get complete dataloader
    # FIXME: ValueError: Shape of passed values is (1536, 128), indices imply (1788, 128) if drop_last=True
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, pin_memory=True, drop_last=False, num_workers=args.num_workers)
    
    # Define global variables
    glob_masked_map = None
    glob_expression_pred = None
    glob_expression_gt = None
    glob_ids = None
    glob_model_mask = None
    glob_ids = adata.obs['unique_id'].tolist() if glob_ids is None else glob_ids + adata.obs['unique_id'].tolist()
    
    # Set model to eval mode
    model=model.to(device)
    model.eval()

    # Get complete predictions
    with torch.no_grad():
        for batch in tqdm(dataloader):
            # TODO: confirm if this returns predictions in model.eval() mode
            # Get the outputs from model
            prediction, expression_gt, random_mask = model.pred_outputs_from_batch(batch, pre_masked_batch = True)
            # Input expression matrix
            input_genes = batch['pre_masked_exp'].clone().detach() #changed .clone for .copy
            # Get predictions only for missing values
            pred_miss = np.zeros(shape=(prediction.shape))
            for i in range(prediction.shape[1]):
                pred_miss[:, i, :] = np.where(input_genes.cpu()[:, i, :] == 0, prediction.cpu()[:, i, :], expression_gt.cpu()[:, i, :])
            # Convert prediction to tensor
            exp_pred = torch.tensor(pred_miss[:,0,:])
            # Concat batch to get global predictions 
            glob_expression_pred = exp_pred if glob_expression_pred is None else torch.cat((glob_expression_pred, exp_pred))
            # Se toma el primer input_genes (parche central)
            masked_map = input_genes.cpu()[:,0,:]
            # Concat batch to get global masked map 
            glob_masked_map = masked_map if glob_masked_map is None else torch.cat((glob_masked_map, masked_map))
               
        # Handle delta prediction
        if 'deltas' in layer:
            mean_key = f'{layer}_avg_exp'.replace('deltas', 'log1p')
            means = torch.tensor(adata.var[mean_key], device=glob_expression_pred.device)
            glob_expression_pred = glob_expression_pred+means  
        
        glob_expression_pred, glob_masked_map = glob_expression_pred.view(-1, glob_expression_pred.shape[-1]), glob_masked_map.view(-1, glob_masked_map.shape[-1])
        
        # Put complete predictions in a single dataframe
        pred_matrix = glob_expression_pred
        pred_df = pd.DataFrame(pred_matrix, index=glob_ids, columns=adata.var_names)
        pred_df = pred_df.reindex(adata.obs.index)
        
        # Put complete masked map in a single dataframe
        mask_matrix = glob_masked_map
        mask_df = pd.DataFrame(mask_matrix, index=glob_ids, columns=adata.var_names)
        mask_df = mask_df.reindex(adata.obs.index)

        # Log predictions in wandb
        #wandb_df = pred_df.reset_index(names='sample')
        #wandb.init()
        #wandb.log({'predictions': wandb.Table(dataframe=wandb_df)})

        # Add layer to adata
        adata.layers[f'predictions,{layer},{method}'] = pred_df
        adata.layers["masked_map"] = mask_df

        if save_path != '':
            ## Save prediction DataFrames
            os.makedirs(os.path.join(save_path, 'predictions', split_name), exist_ok=True)

            # Save model's predictions and initial mask
            pred_df.to_csv(os.path.join(save_path, 'predictions', split_name, f'transformer_preds_{split_name}.csv'))
            mask_df.to_csv(os.path.join(save_path, 'predictions', split_name, f'masked_map_{split_name}.csv'))

            # Save median preditions
            median_imputation_results = adata.layers['median_imputed_expression_matrix'] # Values that are False in layers['random_mask'] are the same as in glob_expression_pred (i.e. ground truth)
            median_df = pd.DataFrame(median_imputation_results, index=glob_ids, columns=adata.var_names)
            median_df = median_df.reindex(adata.obs.index)
            median_df.to_csv(os.path.join(save_path, 'predictions', split_name, f'median_preds_{split_name}.csv'))

            # Save ground truths
            ground_truth_matrix = adata.layers[args.prediction_layer] # Values that are False in layers['random_mask'] are the same as in glob_expression_pred (i.e. ground truth)
            ground_truth_df = pd.DataFrame(ground_truth_matrix, index=glob_ids, columns=adata.var_names)
            ground_truth_df = ground_truth_df.reindex(adata.obs.index)
            ground_truth_df.to_csv(os.path.join(save_path, 'predictions', split_name, f'gt_{split_name}.csv'))