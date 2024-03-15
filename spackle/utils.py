import numpy as np
import pandas as pd
import os
import squidpy as sq
import torch
from tqdm import tqdm
import argparse
import anndata as ad
import glob
from spared.metrics import get_metrics
#from metrics import get_metrics
from sklearn.preprocessing import StandardScaler

# Auxiliary function to use booleans in parser
str2bool = lambda x: (str(x).lower() == 'true')
str2intlist = lambda x: [int(i) for i in x.split(',')]
str2floatlist = lambda x: [float(i) for i in x.split(',')]
str2h_list = lambda x: [str2intlist(i) for i in x.split('//')[1:]]

# Function to get global parser
def get_main_parser():
    parser = argparse.ArgumentParser(description='Code for gene expression imputation.')
    # Dataset parameters #####################################################################################################################################################################
    parser.add_argument('--dataset',                        type=str,           default='10xgenomic_human_brain',   help='Dataset to use.')
    parser.add_argument('--prediction_layer',               type=str,           default='c_d_log1p',                help='The prediction layer from the dataset to use.')
    parser.add_argument('--hex_geometry',                   type=bool,          default=True,                       help='Whether the geometry of the spots in the dataset is hexagonal or not.')
    # Data masking parameters ################################################################################################################################################################
    parser.add_argument('--masking_method',                 type=str,           default='mask_prob',              help='The masking method to use.', choices=['prob_median', 'mask_prob', 'scale_factor'])
    parser.add_argument('--mask_prob',                      type=float,         default=0.3,                        help='The probability of masking a gene for imputation when mas_prob masking methos is selected.')
    parser.add_argument('--scale_factor',                   type=float,         default=0.8,                        help='The scale factor to use for masking when scale_factor masking method is selected.')
    parser.add_argument('--neighborhood_type',              type=str,           default='nn_distance',              help='The method used to select the neighboring spots.', choices=['circular_hops', 'nn_distance'])
    parser.add_argument('--num_neighs',                     type=int,           default=18,                          help='Amount of neighbors to consider for context during imputation.')
    parser.add_argument('--num_hops',                       type=int,           default=1,                          help='Amount of graph hops to consider for context during imputation if neighborhoods are built based on proximity rings.')
    # Imputation model parameters ############################################################################################################################################################
    parser.add_argument('--base_arch',                      type=str,           default='transformer_encoder',      help='Base architecture chosen for the imputation model.', choices=['transformer_encoder', 'MLP'])
    parser.add_argument('--transformer_dim',                type=int,           default=128,                        help='The number of expected features in the encoder/decoder inputs of the transformer.')
    parser.add_argument('--transformer_heads',              type=int,           default=1,                          help='The number of heads in the multiheadattention models of the transformer.')
    parser.add_argument('--transformer_encoder_layers',     type=int,           default=2,                          help='The number of sub-encoder-layers in the encoder of the transformer.')
    parser.add_argument('--transformer_decoder_layers',     type=int,           default=1,                          help='The number of sub-decoder-layers in the decoder of the transformer.')
    parser.add_argument('--include_genes',                  type=str2bool,      default=True,                       help='Whether or not to to include the gene expression matrix in the data inputed to the transformer encoder when using visual features.')
    parser.add_argument('--use_visual_features',            type=str2bool,      default=False,                      help='Whether or not to use visual features to guide the imputation process.')
    parser.add_argument('--use_double_branch_archit',       type=str2bool,      default=False,                      help='Whether or not to use the double branch transformer architecture when using visual features to guide the imputation process.')
    # Model parameters #######################################################################################################################################################################
    parser.add_argument('--sota',                           type=str,           default='pretrain',                 help='The name of the sota model to use. "None" calls main.py, "nn_baselines" calls nn_baselines.py, "pretrain" calls pretrain_backbone.py, and any other calls main_sota.py', choices=['None', 'pretrain', 'stnet', 'nn_baselines', "histogene"])
    parser.add_argument('--img_backbone',                   type=str,           default='ViT',                      help='Backbone to use for image encoding.', choices=['resnet', 'ConvNeXt', 'MobileNetV3', 'ResNetXt', 'ShuffleNetV2', 'ViT', 'WideResNet', 'densenet', 'swin'])
    parser.add_argument('--use_pretrained_ie',              type=str,           default=True,                       help='Whether or not to use a pretrained image encoder model to get the patches embeddings.')
    parser.add_argument('--freeze_img_encoder',             type=str2bool,      default=False,                      help='Whether to freeze the image encoder. Only works when using pretrained model.')
    parser.add_argument('--matrix_union_method',            type=str,           default='concatenate',              help='Method used to combine the output of the gene processing transformer and the visual features processing transformer.', choices=['concatenate', 'sum'])
    parser.add_argument('--num_mlp_layers',                 type=int,           default=5,                          help='Number of layers stacked in the MLP architecture.')
    parser.add_argument('--ae_layer_dims',                  type=str2intlist,   default='512,384,256,128,64,128,256,384,512',                          help='Layer dimensions for ae in MLP base architecture.')
    parser.add_argument('--mlp_act',                        type=str,           default='ReLU',                     help='Activation function to use in the MLP architecture. Case sensitive, options available at: https://pytorch.org/docs/stable/nn.html#non-linear-activations-weighted-sum-nonlinearity')
    parser.add_argument('--mlp_dim',                        type=int,           default=512,                        help='Dimension of the MLP layers.')
    parser.add_argument('--graph_operator',                 type=str,           default='None',                     help='The convolutional graph operator to use. Case sensitive, options available at: https://pytorch-geometric.readthedocs.io/en/latest/modules/nn.html#convolutional-layers', choices=['GCNConv','SAGEConv','GraphConv','GATConv','GATv2Conv','TransformerConv', 'None'])
    parser.add_argument('--pos_emb_sum',                    type=str2bool,      default=False,                      help='Whether or not to sum the nodes-feature with the positional embeddings. In case False, the positional embeddings are only concatenated.')
    parser.add_argument('--h_global',                       type=str2h_list,    default='//-1//-1//-1',             help='List of dimensions of the hidden layers of the graph convolutional network.')
    parser.add_argument('--pooling',                        type=str,           default='None',                     help='Global graph pooling to use at the end of the graph convolutional network. Case sensitive, options available at but must be a global pooling: https://pytorch-geometric.readthedocs.io/en/latest/modules/nn.html#pooling-layers')
    parser.add_argument('--dropout',                        type=float,         default=0.0,                        help='Dropout to use in the model to avoid overfitting.')
    # Train parameters #######################################################################################################################################################################
    parser.add_argument('--num_workers',                    type=int,           default=0,                          help='DataLoader num_workers parameter - amount of subprocesses to use for data loading.')
    parser.add_argument('--num_assays',                     type=int,           default=10,                         help='Number of experiments used to test the model.')
    parser.add_argument('--optim_metric',                   type=str,           default='MSE',                      help='Metric that should be optimized during training.', choices=['PCC-Gene', 'MSE', 'MAE', 'Global'])
    parser.add_argument('--max_steps',                      type=int,           default=10000,                      help='Number of steps to train de model.')
    parser.add_argument('--val_check_interval',             type=int,           default=10,                         help='Number of steps to do valid checks.')
    parser.add_argument('--batch_size',                     type=int,           default=256,                        help='The batch size to train model.')
    parser.add_argument('--shuffle',                        type=str2bool,      default=True,                       help='Whether or not to shuffle the data in dataloaders.')
    parser.add_argument('--lr',                             type=float,         default=1e-3,                       help='Learning rate to use.')
    parser.add_argument('--optimizer',                      type=str,           default='Adam',                     help='Optimizer to use in training. Options available at: https://pytorch.org/docs/stable/optim.html It will just modify main optimizers and not sota (they have fixed optimizers).')
    parser.add_argument('--momentum',                       type=float,         default=0.9,                        help='Momentum to use in the optimizer if it receives this parameter. If not, it is not used. It will just modify main optimizers and not sota (they have fixed optimizers).')
    parser.add_argument('--average_test',                   type=str2bool,      default=False,                      help='If True it will compute the 8 symmetries of an image during test and the prediction will be the average of the 8 outputs of the model.')
    parser.add_argument('--cuda',                           type=str,           default='0',                        help='CUDA device to run the model.')
    parser.add_argument('--exp_name',                       type=str,           default='None',                     help='Name of the experiment to save in the results folder. "None" will assign a date coded name.')
    parser.add_argument('--train',                          type=str2bool,      default=True,                       help='If true it will train, if false it only tests')
    parser.add_argument('--load_ckpt_path',                 type=str,           default='',                         help='Path to the checkpoints that will be tested when not training from scratch.')
    ##########################################################################################################################################################################################

    return parser

def seed_everything(seed: int):
    import random, os
    import numpy as np
    import torch
    
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

### Define function to get spatial neighbors in an AnnData object
def get_spatial_neighbors(adata: ad.AnnData, n_hops: int, hex_geometry: bool) -> dict:
    """
    This function computes a neighbors dictionary for an AnnData object. The neighbors are computed according to topological distances over
    a graph defined by the hex_geometry connectivity. The neighbors dictionary is a dictionary where the keys are the indexes of the observations
    and the values are lists of the indexes of the neighbors of each observation. The neighbors include the observation itself and are found
    inside a n_hops neighborhood of the observation.

    Args:
        adata (ad.AnnData): the AnnData object to process. Importantly it is only from a single slide. Can not be a collection of slides.
        n_hops (int): the size of the neighborhood to take into account to compute the neighbors.
        hex_geometry (bool): whether the graph is hexagonal or not. If True, then the graph is hexagonal. If False, then the graph is a grid. Only
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

def get_mask_prob_tensor(masking_method, dataset, mask_prob=0.3, scale_factor=0.8):
    """
    This function calculates the probability of masking each gene present in the expression matrix. 
    Within this function, there are three different methods for calculating the masking probability, 
    which are differentiated by the 'masking_method' parameter. 
    The return value is a vector of length equal to the number of genes, where each position represents
    the masking probability of that gene.
    
    Args:
        masking_method (str): parameter used to differenciate the method for calculating the probabilities.
        dataset (SpatialDataset): the dataset in a SpatialDataset object.
        mask_prob (float): masking probability for all the genes. Only used when 'masking_method = mask_prob' 
        scale_factor (float): maximum probability of masking a gene if masking_method == 'scale_factor'
    Return:
        prob_tensor (torch.Tensor): vector with the masking probability of each gene for testing. Shape: n_genes  
    """

    # Convert glob_exp_frac to tensor
    glob_exp_frac = torch.tensor(dataset.adata.var.glob_exp_frac.values, dtype=torch.float32)
    # Calculate the probability of median imputation
    prob_median = 1 - glob_exp_frac

    if masking_method == "prob_median":
        # Calculate masking probability depending on the prob median
        # (the higher the probability of being replaced with the median, the higher the probability of being masked).
        prob_tensor = prob_median/(1-prob_median)

    elif masking_method == "mask_prob":
        # Calculate masking probability according to mask_prob parameter
        # (Mask everything with the same probability)
        prob_tensor = mask_prob/(1-prob_median)

    elif masking_method == "scale_factor":
        # Calculate masking probability depending on the prob median scaled by a factor
        # (Multiply by a factor the probability of being replaced with median to decrease the masking probability).
        prob_tensor = prob_median/(1-prob_median)
        prob_tensor = prob_tensor*scale_factor
        
    # If probability is more than 1, set it to 1
    prob_tensor[prob_tensor>1] = 1

    return prob_tensor

def mask_exp_matrix(adata: ad.AnnData, pred_layer: str, mask_prob_tensor: torch.Tensor, device):
    """
    This function recieves an adata and masks random values of the pred_layer based on the masking probability of each gene, then saves the masked matrix in the corresponding layer. 
    It also saves the final random_mask for metrics computation. True means the values that are real in the dataset and have been masked for the imputation model development.
    
    Args:
        adata (ad.AnnData): adata of the data split that will be masked and imputed.
        pred_layer (str): indicates the adata.layer with the gene expressions that should be masked and later reconstructed. Shape: spots_in_adata, n_genes
        mask_prob_tensor (torch.Tensor):  tensor with the masking probability of each gene for testing. Shape: n_genes
    
    Return:
        adata (ad.AnnData): adata of the data split with the gene expression matrix already masked and the corresponding random_mask in adata.layers.
    """

    # Extract the expression matrix
    expression_mtx = torch.tensor(adata.layers[pred_layer])
    # Calculate the mask based on probability tensor
    random_mask = torch.rand(expression_mtx.shape).to(device) < mask_prob_tensor.to(device)
    median_imp_mask = torch.tensor(adata.layers['mask']).to(device)
    # Combine random mask with the median imputation mask
    random_mask = random_mask.to(device) & median_imp_mask
    # Mask chosen values.
    expression_mtx[random_mask] = 0
    # Save masked expression matrix in the data_split annData
    adata.layers['masked_expression_matrix'] = np.asarray(expression_mtx)
    #Save final mask for metric computation
    adata.layers['random_mask'] = np.asarray(random_mask.cpu())

    return adata

def get_mean_performance(complete_imputation_function, n_assays: int, model, trainer, best_model_path: str, args: argparse, prob_tensor: torch.Tensor, device: torch.device, train_split: ad.AnnData, val_split: ad.AnnData, test_split: ad.AnnData = None) -> dict:
    """
    This function receives the data before being imputed and performs n_assays experiments imputing through both methods. 
    Each experiment uses a different random mask and the results are used to obtain more accurate metrics and calculate 
    the variation in the performance of both methods.

    Args:
        complete_imputation_function (function): adata with the data split of interest.
        n_assays (int): number of experiments considered to test the model's performance.
        model (model): imputation model with loaded weights to test perfomance.
        trainer (lightning.Trainer): pytorch lightning trainer used for model training and testing.
        best_model_path (str): path to the checkpoints that will be tested.
        args (argparse): parser with the values necessary for data processing.
        prob_tensor (torch.Tensor): vector with the masking probability of each gene for testing. Shape: n_genes  
        device (torch.device): device in which tensors will be processed.
        train_split (ad.AnnData): adata of the train data split before being masked and imputed through median and trained model.
        val_split (ad.AnnData): adata of the val data split before being masked and imputed through median and trained model.
        test_split (ad.AnnData, optional): if available, adata of the test data split before being masked and imputed through median and trained model.

    Returns:
        final_metrics (dict): dictionary with the mean and standard deviation of each evaluation metric for both the median imputation method and the model.
    """
    def calculate_final_stats(grouped_metrics_dict: dict):
        """
        This function gets the results of all the evaluation metrics in each experiment done to then calculate the mean and
        standard deviation of each metric in both methods.

        Args:
            grouped_metrics_dict (dict): dictionary of the performance metrics for both imputation methods in each data split. 
            The splits have a list of values for every evaluation metric obtained in each experiment.

        Returns:
            final_metrics (dict): dictionary with the mean and standard deviation of each evaluation metric for both the median imputation method and the model.
        """
        final_metrics = {'train':{'model':{}},
                        'val':{'model':{}},
                        'test':{'model':{}}}
        
        for split, methods_dict in grouped_metrics_dict.items():
            for method, metrics_dict in methods_dict.items():
                for metric, assays_results in metrics_dict.items():
                    # Add metric to the keys of the dictionary if it hasn't been added yet
                    final_metrics[split][method].setdefault(metric, {})
                    # Calculate and save mean and standard deviation of the metric
                    final_metrics[split][method][metric]['mean'] = np.mean(assays_results)
                    final_metrics[split][method][metric]['std_dev'] = np.std(assays_results)

        return final_metrics
    
    stats_per_split = {'train':{'model':{}},
                       'val':{'model':{}},
                       'test':{'model':{}}}
    
    # Test both imputation methods in n_assays random masking problems of all data splits
    for i in range(n_assays):
        results = complete_imputation_function(
                                model = model, 
                                trainer = trainer, 
                                best_model_path = best_model_path, 
                                args = args,
                                prob_tensor = prob_tensor, 
                                device = device, 
                                train_split = train_split, 
                                val_split = val_split, 
                                test_split = test_split
                                )
        # Get quantitative results and adatas with the last random masking performed saved in layers
        results, train_split, val_split, test_split = results

        # Iterate over the 6 results dictionaries (one for every data split in each imputation method)
        for split_results, values_dict in results.items():
            split_name = split_results.split("_")[0]
            method_name = split_results.split("_")[1]

            # Iterate over the 7 metrics to regroup the results of each assay in stats_per_split
            for metric in values_dict.keys():
                metric_name = metric.split("_")[-1]
                # Add metric to the keys of the dictionary if it hasn't been added yet
                stats_per_split[split_name][method_name].setdefault(metric_name, [])
                # Append the value of the metric to the list of n_assays results
                stats_per_split[split_name][method_name][metric_name].append(values_dict[metric])

    final_metrics = calculate_final_stats(stats_per_split)

    return final_metrics, train_split, val_split, test_split 

### Function to get expression deltas from the mean expression of a gene expression matrix
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

# To test the code
if __name__=='__main__':
    hello = 0