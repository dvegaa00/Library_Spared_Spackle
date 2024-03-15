import os
os.environ['USE_PYGEOS'] = '0' # To supress a warning from geopandas
import json
from utils import *
from spared.datasets import get_dataset
from model import GeneImputationModel
from lightning.pytorch import Trainer
from dataset import ImputationDataset
from torch.utils.data import DataLoader
from lightning.pytorch.callbacks import ModelCheckpoint
from predictions import get_predictions
from visualize_data import log_pred_image


## Set of auxiliary functions for model test and comparison
def get_imputation_results_from_trained_model(trainer, model, best_model_path, train_loader, val_loader, test_loader = None):
    """
    This function tests the incoming model in all data splits available using pytorch lightning.

    Args:
        trainer (lightning.Trainer): pytorch lightning trainer used for model training and testing.
        model (model): imputation model with loaded weights to test perfomance.
        best_model_path (str): path to the checkpoints that will be tested.
        train_loader (torch.DataLoader): DataLoader of the train data split. 
        val_loader (torch.DataLoader): DataLoader of the val data split. 
        test_loader (torch.DataLoader, optional): if available, DataLoader of the test data split. 

    Return:
        train_model_imputation_metrics (dict): Evaluation metrics when testing the model on train split.
        val_model_imputation_metrics (dict): Evaluation metrics when testing the model on val split.
        test_model_imputation_metrics (dict): returned if test data is provided, else is None. Evaluation metrics when testing the model on test split.
    """
    ## Results for imputation model
    train_model_imputation_metrics = trainer.test(model = model, dataloaders = train_loader, ckpt_path = best_model_path)[0]
    val_model_imputation_metrics = trainer.test(model = model, dataloaders = val_loader, ckpt_path = best_model_path)[0]
    test_model_imputation_metrics = None

    # Use test_split too if available
    if test_loader is not None:
        test_model_imputation_metrics = trainer.test(model = model, dataloaders = test_loader, ckpt_path = best_model_path)[0]

    return train_model_imputation_metrics, val_model_imputation_metrics, test_model_imputation_metrics

def get_complete_imputation_results(model, trainer, best_model_path, args, prob_tensor, device, train_split, val_split, test_split = None):
    """
    This function gets the evaluation metrics of both the median filter and the trained model in all data splits available.

    Args:
        model (model): imputation model with loaded weights to test perfomance.
        trainer (lightning.Trainer): pytorch lightning trainer used for model training and testing.
        best_model_path (str): path to the checkpoints that will be tested.
        args (argparse): parser with the values necessary for data processing.
        prob_tensor (torch.Tensor): vector with the masking probabilities for each gene. Shape: n_genes  
        device (torch.device): device in which tensors will be processed.
        train_split (ad.AnnData): adata of the train data split before being masked and imputed through median and trained model.
        val_split (ad.AnnData): adata of the val data split before being masked and imputed through median and trained model.
        test_split (ad.AnnData, optional): if available, adata of the test data split before being masked and imputed through median and trained model.

    Return:
        complete_imputation_results (dict): contains the evaluation metrics of the imputation through both methods (median and model) in all data splits available.
        train_split (ad.AnnData): updated train adata with the prediction layers included.
        val_split (ad.AnnData): updated val adata with the prediction layers included.
        test_split (ad.AnnData): if not None, updated test adata with the prediction layers included.

    """
    complete_imputation_results = {}
    ## Results for median filter
    train_split, train_median_imputation_results = apply_median_imputation_method(
        data_split = train_split, 
        split_name = 'train', 
        prediction_layer = args.prediction_layer, 
        prob_tensor = prob_tensor, 
        device = device)
    
    print('Median imputation results on train data split: ', train_median_imputation_results)
    
    val_split, val_median_imputation_results = apply_median_imputation_method(
        data_split = val_split, 
        split_name = 'val', 
        prediction_layer = args.prediction_layer, 
        prob_tensor = prob_tensor, 
        device = device)
    
    print('Median imputation results on val data split: ', val_median_imputation_results)
    
    # Use test_split too if available
    test_median_imputation_results = None
    if test_split is not None:
        # Results for median filter
        test_split, test_median_imputation_results = apply_median_imputation_method(
            data_split = test_split, 
            split_name = 'test', 
            prediction_layer = args.prediction_layer, 
            prob_tensor = prob_tensor, 
            device = device)
        
        print('Median imputation results on test data split: ', test_median_imputation_results)
        
    ## Prepare DataLoaders for testing on trained model
    train_data = ImputationDataset(train_split, args, 'train', pre_masked = True)
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=False, pin_memory=True, drop_last=True, num_workers=args.num_workers)

    val_data = ImputationDataset(val_split, args, 'val', pre_masked = True)
    val_loader = DataLoader(val_data, batch_size=args.batch_size, shuffle=False, pin_memory=True, drop_last=True, num_workers=args.num_workers)
    test_loader = None
    if test_split is not None:
        test_data = ImputationDataset(test_split, args, 'test', pre_masked = True)
        test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False, pin_memory=True, drop_last=True, num_workers=args.num_workers)

    ## Results for trained model
    trained_model_results = get_imputation_results_from_trained_model(
        trainer, model, best_model_path, 
        train_loader, val_loader, test_loader = test_loader)

    # Build dictionary with results from median and model
    complete_imputation_results = {
        'train_median_imputation_results': train_median_imputation_results,
        'val_median_imputation_results': val_median_imputation_results,
        'train_model_results': trained_model_results[0],
        'val_model_results': trained_model_results[1]
        }
    
    if test_split is not None:
        complete_imputation_results['test_median_imputation_results'] = test_median_imputation_results
        complete_imputation_results['test_model_results'] = trained_model_results[2]
    
    return complete_imputation_results, train_split, val_split, test_split


def main():
    """
    Parameters:
    train_split, val_split, test_split (optional), batch_size, shuffle, pin_memory=True, drop_last=True, num_workers=0
    masking_method, mask_prob, scale_factor, optim_metric, val_check_interval, max_steps, num_assays
    """
    # Create saving directory
    os.makedirs(save_path, exist_ok=True)

    # TODO: adata entra por parámetro (entran un adata por split)
    # Get dataset from the values defined in args
    dataset = get_dataset(args.dataset)

    # TODO: eliminar opción con features visuales
    if args.use_visual_features:
        # Get the path to the optimized model that will work as image encoder if using pre trained weights
        img_encoder_ckpts_path = get_img_encoder_ckpts(dataset=args.dataset, args=args) if args.use_pretrained_ie else "None"
        # Get embeddings of patch images
        dataset.compute_patches_embeddings_and_predictions(
            backbone = args.img_backbone, 
            model_path = img_encoder_ckpts_path, 
            preds=False
            )

    # TODO: quitar splits internos
    # Check if test split is available
    test_data_available = True if 'test' in dataset.adata.obs['split'].unique() else False
    # Declare data splits
    train_split = dataset.adata[dataset.adata.obs['split']=='train']
    val_split = dataset.adata[dataset.adata.obs['split']=='val']
    test_split = dataset.adata[dataset.adata.obs['split']=='test'] if test_data_available else None

    # Prepare data and create dataloaders
    train_data = ImputationDataset(train_split, args, 'train')
    val_data = ImputationDataset(val_split, args, 'val')

    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=args.shuffle, pin_memory=True, drop_last=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_data, batch_size=args.batch_size, shuffle=args.shuffle, pin_memory=True, drop_last=True, num_workers=args.num_workers)

    # Get masking probability tensor for training
    train_prob_tensor = get_mask_prob_tensor(args.masking_method, dataset, args.mask_prob, args.scale_factor)
    # Get masking probability tensor for validating and testing with fixed method 'prob_median'
    # TODO: definir si masking de test es igual al método de train o es fijo en prob_median
    val_test_prob_tensor = get_mask_prob_tensor('prob_median', dataset, args.mask_prob, args.scale_factor)
    # FIXME: change masking method for test to args.masking_method (i.e. 'mask_prob') when testing on a specifik masking proportion (i.e. progressive masking experiment)
    #val_test_prob_tensor = get_mask_prob_tensor(args.masking_method, dataset, args.mask_prob, args.scale_factor)

    # Declare model
    model = GeneImputationModel(
        args=args, 
        data_input_size=dataset.adata.n_vars,
        train_mask_prob_tensor=train_prob_tensor.to(device), 
        val_test_mask_prob_tensor = val_test_prob_tensor.to(device)
        ).to(device)        
        
    print(model.model)

    # Define dict to know whether to maximize or minimize each metric
    max_min_dict = {'PCC-Gene': 'max', 'PCC-Patch': 'max', 'MSE': 'min', 'MAE': 'min', 'R2-Gene': 'max', 'R2-Patch': 'max', 'Global': 'max'}

    # Define checkpoint callback to save best model in validation
    checkpoint_callback = ModelCheckpoint(
        dirpath=save_path,
        monitor=f'val_{args.optim_metric}', # Choose your validation metric
        save_top_k=1, # Save only the best model
        mode=max_min_dict[args.optim_metric], # Choose "max" for higher values or "min" for lower values
    )

    # Define the pytorch lightning trainer
    trainer = Trainer(
        max_steps=args.max_steps,
        val_check_interval=args.val_check_interval,
        log_every_n_steps=args.val_check_interval,
        check_val_every_n_epoch=None,
        devices=1,
        callbacks=[checkpoint_callback],
        enable_progress_bar=True,
        enable_model_summary=True,
    )
    
    # Train the model
    trainer.fit(
        model=model,
        train_dataloaders=train_loader,
        val_dataloaders=val_loader
    )
    # Load the best model after training
    best_model_path = checkpoint_callback.best_model_path

    ### TEST
    # Load the checkpoints that will be tested
    best_model_path = args.load_ckpt_path
    
    # Test median imposition and trained/loaded model on the same masked data
    mean_performance = get_mean_performance(
                get_complete_imputation_results, 
                n_assays = args.num_assays, 
                model = model, 
                trainer = trainer, 
                best_model_path = best_model_path, 
                args = args,
                prob_tensor = val_test_prob_tensor, 
                device = device, 
                train_split = train_split, 
                val_split = val_split, 
                test_split = test_split)

    # Get quantitative results and adatas with the last random masking performed saved in layers
    mean_performance_results, train_split, val_split, test_split = mean_performance

    # Save results in a txt file
    test_description = f"Gene imputation through median and {args.base_arch} model.\n"\
                        f"Checkpoints restored from {best_model_path}"
    
    file_path = os.path.join(save_path, 'testing_results.txt')
    with open(file_path, 'w') as txt_file:
        txt_file.write(test_description)
        # Convert the dictionary to a formatted string
        dict_str = json.dumps(mean_performance_results, indent=4)
        txt_file.write(dict_str)
    
    if test_data_available:
        get_predictions(adata = test_split, 
                    args = args, 
                    model = model, 
                    split_name = 'test', 
                    layer = args.prediction_layer, 
                    method = 'transformer', 
                    device = device, 
                    #save_path = save_path # uncomment to save csv files with predictions
                    )
        
    else:
        get_predictions(adata = val_split, 
                    args = args, 
                    model = model, 
                    split_name = 'val', 
                    layer = args.prediction_layer, 
                    method = 'transformer', 
                    device = device
                    )
        
    return train_split, val_split, test_split

def test_completion_model(test_split, ckpt_path, save_results_path):
    ### TEST
    # Load the checkpoints that will be tested
    best_model_path = ckpt_path
    
    # Test median imposition and trained/loaded model on the same masked data
    mean_performance = get_mean_performance(
                get_complete_imputation_results, 
                n_assays = args.num_assays, 
                model = model, 
                trainer = trainer, 
                best_model_path = best_model_path, 
                args = args,
                prob_tensor = val_test_prob_tensor, 
                device = device, 
                train_split = train_split, 
                val_split = val_split, 
                test_split = test_split)

    # Get quantitative results and adatas with the last random masking performed saved in layers
    mean_performance_results, train_split, val_split, test_split = mean_performance

    # Save results in a txt file
    test_description = f"Gene imputation using SpaCKLE model.\n"\
                        f"Checkpoints restored from {best_model_path}"
    
    file_path = os.path.join(save_results_path, 'testing_results.txt')
    with open(file_path, 'w') as txt_file:
        txt_file.write(test_description)
        # Convert the dictionary to a formatted string
        dict_str = json.dumps(mean_performance_results, indent=4)
        txt_file.write(dict_str)
    
    # add predictions layer to adata
    save_predictions_path = save_results_path if save_predictions else ''
    get_predictions(adata = test_split, 
                args = args, 
                model = model, 
                split_name = 'test', 
                layer = args.prediction_layer, 
                method = 'transformer', 
                device = device, 
                save_path = save_predictions_path # if != '', then the predictions will be saved to csv file
                )
    
    # TODO: print metrics results
        
    return train_split, val_split, test_split
        
    # Only run on test set to log in WandB
    adata_for_visuals = test_split if test_data_available else val_split
    prepare_preds_for_visuals(adata_for_visuals, args)

    # Log prediction images
    log_pred_image(adata = adata_for_visuals, n_genes = 3)  
    

# Run gene imputation model
if __name__=='__main__':
    main()