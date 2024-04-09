import os
os.environ['USE_PYGEOS'] = '0' # To supress a warning from geopandas
import json
import torch
import anndata as ad
from utils import *
from spared.datasets import get_dataset
from model import GeneImputationModel
from lightning.pytorch import seed_everything
from dataset import ImputationDataset
from torch.utils.data import DataLoader
import warnings

# Get parser and parse arguments
parser = get_main_parser()
args = parser.parse_args()
args_dict = vars(args)

# Set manual seeds and get cuda
seed_everything(42, workers=True)
# Set cuda visible devices
if "CUDA_VISIBLE_DEVICES" in os.environ.keys():
    args.cuda = os.environ["CUDA_VISIBLE_DEVICES"]
os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda
use_cuda = torch.cuda.is_available()

# Declare device
device = torch.device("cuda" if use_cuda else "cpu")

def complete_real_missing():
    # Get dataset from the values defined in args
    dataset = get_dataset(args.dataset)
    adata = dataset.adata

    # Declare model
    vis_features_dim = adata.obsm[f'embeddings_{args.img_backbone}'].shape[-1] if args.use_visual_features else 0
    model = GeneImputationModel(
        args=args, 
        data_input_size=dataset.adata.n_vars,
        vis_features_dim=vis_features_dim
        ).to(device)  
    
    # Get path to best checkpoints 
    model_method = 'with_visual_fts' if args.use_visual_features else 'only_gene_inputs'
    best_model_dir = os.path.join('optimal_imputation_models', model_method, args.dataset, '**', '')
    best_model_dir = glob.glob(best_model_dir)[-1]

    with open(os.path.join(best_model_dir, 'script_params.json'), 'r') as f:
        saved_script_params = json.load(f)
        # Check that the parameters of the loaded model agree with the current inference process
        if (saved_script_params['prediction_layer'] != args_dict['prediction_layer']) or (saved_script_params['prediction_layer'] != args_dict['prediction_layer']):
            warnings.warn("Saved model's parameters differ from those of the current argparse.")

    best_model_path = glob.glob(os.path.join(best_model_dir, '*.ckpt'))[0]

    # Load best checkpoints
    state_dict = torch.load(best_model_path)
    state_dict = state_dict['state_dict']
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    print(f"Finished loading model with weights from {best_model_path}")

    # Prepare data and dataloader
    data = ImputationDataset(adata, args, 'complete')
    dataloader = DataLoader(
        data, 
        batch_size=args.batch_size, 
        shuffle=False, 
        pin_memory=True, 
        drop_last=False, 
        num_workers=args.num_workers)
    
    # Get gene imputations for missing values
    all_exps = []
    all_masks = []
    exp_with_imputation = []
    with torch.no_grad():
        for batch in tqdm(dataloader):
            del batch['split_name']
            # Extract batch variables
            batch = {k: v.to(device) for k, v in batch.items()}
            expression_gt = batch['exp_matrix_gt']
            mask = batch['real_missing']
            # Remove median imputations from gene expression matrix
            input_genes = expression_gt.clone()
            input_genes[~mask] = 0

            # Get predictions
            prediction = model.forward(input_genes)

            # Imput predicted gene expression only in missing data for 'main spot' in the neighborhood
            imputed_exp = torch.where(mask[:,0,:], expression_gt[:,0,:], prediction[:,0,:])

            all_exps.append(expression_gt[:,0,:])
            all_masks.append(batch['real_missing'][:,0,:])
            exp_with_imputation.append(imputed_exp)

    # Concatenate output tensors into complete data expression matrix
    all_exps = torch.cat(all_exps)
    all_masks = torch.cat(all_masks)
    exp_with_imputation = torch.cat(exp_with_imputation) 

    # Add imputed data to adata
    adata.layers['c_t_log1p'] = np.asarray(exp_with_imputation.cpu().double())
    # Get deltas layer from new imputed gene values matrix
    adata = get_deltas(adata, 'c_t_log1p', 'c_t_deltas')
    # Replace adata file in spared > preprocessed_data
    author = args.dataset.split('_')[0]
    if '10x' in author:
        author = 'Visium'

    elif 'fan' in author:
        author = 'fan_mouse_brain'

    preprocessed_dataset_path = os.path.join('spared', 'processed_data', f'{author}_data', args.dataset, '**', 'adata.h5ad')

    # FIXME: parigi_data directory skips the dataset name inner directory 'parigi_mouse_intestine'
    if 'parigi' in author:
        preprocessed_dataset_path = os.path.join('spared', 'processed_data', f'{author}_data', '**', 'adata.h5ad')

    preprocessed_dataset_path = glob.glob(preprocessed_dataset_path)[0]
    print('Replacing adata file ...')
    adata.write(os.path.join(preprocessed_dataset_path))
    print(f'Dataset: {args.dataset}')
    print("Transformer imputation for layers c_d_log1p and c_d_deltas into c_t_log1p and c_t_deltas done.")
    print('-----------------------------------------------------------------------------------------------------------------------------------')

# Perform gene imputation inference
if __name__=='__main__':
    complete_real_missing()
    
