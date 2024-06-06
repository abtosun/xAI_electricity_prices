# %% [markdown]
### 1. Setup configuration 
# %% 
import warnings
warnings.filterwarnings("ignore")

import pandas as pds
import numpy as np
import os
import time

from argparse import ArgumentParser
import gc ; gc.enable()
import torch
from torch import nn
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from torchmetrics import Accuracy,MeanAbsolutePercentageError
from pytorch_forecasting import SMAPE
from sklearn.model_selection import TimeSeriesSplit
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from src.config import load_default, dataset_name_check, setup_config_with_method


config = load_default()
dataset_name_check(config,config.dataset_name)
setup_config_with_method(config)

if torch.cuda.is_available():
    n_gpu = torch.cuda.device_count()
    config.gpu_numb = [n_gpu-1]

print("CUDA Available:", torch.cuda.is_available())
print("Number of GPUs:", torch.cuda.device_count())
print("GPU Number Config:", config.gpu_numb)

device = torch.device('cuda:{}'.format(config.gpu_numb[0])) 

# %% [markdown]
### 2. Load dataset
# %%
raw_df = pds.read_csv(config.data_path)

if 'sample_data' in config.data_path:
    raw_df = raw_df.drop(columns = 'Date')
    
# %% [markdown]
### 3. Standart Scaling & Preparing Train-Validation Size

# %%
scaler = StandardScaler()
scaled_df = scaler.fit_transform(raw_df)

nsamples = len(raw_df)
train_size = int(nsamples * config.training_size)
val_size = int(nsamples * 0.1)


# Custom split generator
def custom_time_series_split(data, n_splits, train_size, val_size):
    total_size = train_size + val_size
    indices = np.arange(len(data))
    splits = []
    step = (len(data) - total_size) // (n_splits - 1)
    for i in range(n_splits):
        start = i * step
        end = start + total_size
        if end <= len(data):
            train_indices = indices[start:start + train_size]
            val_indices = indices[start + train_size:start + total_size]
            splits.append((train_indices, val_indices))
    return splits

splits = custom_time_series_split(scaled_df, config.n_fold, train_size, val_size)

# %% [markdown]
### 4. Defining Evaluation Metrics
# %%
def fn_smape(y, y_pred):
    return ((2 * np.abs(y - y_pred)) / (np.abs(y) + np.abs(y_pred))).mean()

def compute_metric(y_true, y_pred):
    metrics = {
        'R2': r2_score(y_true, y_pred),
        'RMSE': mean_squared_error(y_true, y_pred) ** 0.5,
        'MAE': mean_absolute_error(y_true, y_pred),
        'SMAPE': fn_smape(y_true, y_pred)
    }
    return metrics

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# %% [markdown]
### 5. Trainig & Evaluation

# %%
# Loop for Cross-Validation

for fold, (train_index, val_index) in enumerate(splits):   
    print(f"Training fold {fold + 1}/{config.n_fold}")
    
    train_df = scaled_df[train_index]
    valid_df = scaled_df[val_index]

    print(f"Fold {fold + 1} - Training indices: {train_index[:5]}...{train_index[-5:]}")
    print(f"Fold {fold + 1} - Validation indices: {val_index[:5]}...{val_index[-5:]}")
    
    # Sequences Slicing 
    train_seqs = np.lib.stride_tricks.sliding_window_view(
        x=train_df,
        window_shape=(config.input_length + config.output_length),
        axis=0
    ).transpose([0, 2, 1])
    
    valid_seqs = np.lib.stride_tricks.sliding_window_view(
        x=valid_df,
        window_shape=(config.input_length + config.output_length),
        axis=0
    ).transpose([0, 2, 1])
    
    dataset_dict = dict(
        train=(train_seqs[:, :-1], train_seqs[:, -1]),
        valid=(valid_seqs[:, :-1], valid_seqs[:, -1])
    )
    
    from src.models.model_selector import get_model
    from src.data_prepare import pl_DataModule
    
    # Preparing the model 
    pldm = pl_DataModule(dataset_dict, config) 
    
    config.input_feature = raw_df.shape[1]
    config.output_feature = raw_df.shape[1]
    
    model = get_model(config) # Model Selection NATM (Feature, Time, Independent), DNN, LSTM, SCINet
    
    model.train()
    save_name = '_'.join([config.exp_name, config.method, config.dataset_name, str(config.input_length)])
    
    callbacks = [
        ModelCheckpoint(
            dirpath=os.path.join('.', config.save_ckpt_dirs, save_name, f'fold_{fold + 1}'),
            filename='{epoch:03d}-{val_loss:.3f}-{val_SMAPE:.3f}',
            save_last=True,
            save_top_k=config.save_top_k,
            monitor='val_loss',
        ),
        EarlyStopping(
            monitor='val_loss',
            patience=config.ealry_stop_round,
        )
    ]
    
    trainer = pl.Trainer(
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        enable_progress_bar=config.prog_bar,
        devices=config.gpu_numb,
        max_epochs=config.epochs,
        callbacks=callbacks,
    )
    
    # Count the number of parameters
    num_params = count_parameters(model)
    print(f"Fold {fold + 1} - Number of parameters: {num_params}")

    # Measure the training time
    start_time = time.time()

    # Training 
    trainer.fit(model, datamodule=pldm)

    end_time = time.time()
    training_time = end_time - start_time
    print(f"Fold {fold + 1} - Training time: {training_time:.2f} seconds")
    
    # Logging
    model.eval()
    outputs = trainer.predict(model, pldm.val_dataloader())
    
    return_trues = []
    return_preds = []
    
    for output in outputs:
        if len(output) == 4:
            yt, yp, met, w = output
        else:
            yt, yp, w = output
        
        return_trues.append(yt.numpy())
        return_preds.append(yp.numpy())
    
    return_trues = scaler.inverse_transform(np.concatenate(return_trues))
    return_preds = scaler.inverse_transform(np.concatenate(return_preds))
    
    joblib.dump(
    dict(
        scaler = scaler,
        config = config,
        columns = raw_df.columns
    ), os.path.join('.',config.save_ckpt_dirs, save_name, f'fold_{fold + 1}', 'log.joblib')
    )

    joblib.dump(
    dict(
        train = train_seqs,
        valid = valid_seqs,
    ), os.path.join('.',config.save_ckpt_dirs, save_name, f'fold_{fold + 1}', 'data_samples.joblib')
    )

    # Save train and validation indices
    indices_path = os.path.join('.', config.save_ckpt_dirs, save_name, f'fold_{fold + 1}', 'indices.joblib')
    joblib.dump({'train_index': train_index, 'val_index': val_index}, indices_path)
    
    # Extracting Price Feature
    feature_true = return_trues[:, 0]
    feature_pred = return_preds[:, 0]
    
    #Print sizes of feature_true and feature_pred for inspection
    print(f"Fold {fold + 1} - feature_true size: {feature_true.shape}, feature_pred size: {feature_pred.shape}")
    
    # Evaluation
    metrics = compute_metric(feature_true, feature_pred)
    metrics.update({
        'method': config.method,
        'input_length': config.input_length,
        'dataset_name': config.dataset_name,
        'fold': fold + 1,
        'num_params': num_params,
        'training_time': training_time
    })


    df_metrics = pds.DataFrame([metrics])

    results_dir = 'results'
    csv_path = os.path.join(results_dir, 'evaluation_metrics_results.csv')

    os.makedirs(results_dir, exist_ok=True)
    print("Writing to CSV now...") # Controlling
    print(df_metrics) # Controlling

    if os.path.isfile(csv_path):
        df_metrics.to_csv(csv_path, mode='a', header=False, index=False)
    else:
        df_metrics.to_csv(csv_path, mode='w', header=True, index=False)

