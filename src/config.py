from argparse import ArgumentParser
import os
from glob import glob

def load_default():
    parser = ArgumentParser()

    
    parser.add_argument('--natm_type', default = None, type = str)
    parser.add_argument('--save_ckpt_dirs', default = 'ckpt', type = str)

    
    ## Default model param
    parser.add_argument('--input_length', default=8, type=int)
    parser.add_argument('--output_length', default=1, type=int)
    parser.add_argument('--seed', default=2022, type=int)


    parser.add_argument('--dataset_list', default=[
                            'aggregated_dataframe_germany','aggregated_dataframe_france',
                            'aggregated_dataframe_belgium', 'aggregated_dataframe_netherlands',
                            'aggregated_dataframe_spain'
                        ], type = list)

    parser.add_argument('--dataset_path', 
                        default= os.path.join('.','sample_data'),
                        type = str)
    parser.add_argument('--data_path', default=None, type = str)
    parser.add_argument('--dataname', default=None, type = str)
    parser.add_argument('--dataset_name', default = 'spain', type = str)
    parser.add_argument('--n_fold', default=5, type=int)
    parser.add_argument('--training_size', default=0.5, type=float)
    parser.add_argument('--multi_thread', default = 6)

    ## Exp variable params 
    parser.add_argument('--exp_name', default = 'test', type = str)
    parser.add_argument('--fold_numb', default = None, type = str)
    parser.add_argument('--gpu_numb', default = None, type = list)
    parser.add_argument('--method', default = 'Feature', type = str)

    ## Training params
    parser.add_argument('--prog_bar', default = True, type = bool)
    parser.add_argument('--regression', default = True, type = bool)
    parser.add_argument('--epochs', default = 1000, type = int)
    parser.add_argument('--lr', default = 1e-3, type = float)
    parser.add_argument('--batch_size', default = 128, type = int)
    parser.add_argument('--num_workers', default = 0, type = int)
    parser.add_argument('--ealry_stop_round', default = 10, type = int)
    parser.add_argument('--save_top_k', default = 1, type = int)

    ## Global model params
    parser.add_argument('--dropout', default = 0.1, type = float)

    ## NAM modules params
    parser.add_argument('--nam_hidden_sizes', default = [128], type = list)
    parser.add_argument('--nam_activation', default = 'linear', type = str)
    parser.add_argument('--nam_basis_functions', default = None, type = int)
    parser.add_argument('--nam_output_bias', default = False, type = bool)
    parser.add_argument('--nam_output_dropout', default = .0, type = float)
    parser.add_argument('--activation', default = 'leaky', type = str)
    parser.add_argument('--shuffle', default = True, type = bool)
    parser.add_argument('--mean', default = .0, type = float)
    parser.add_argument('--std', default = 1., type = float)

    ## Nam regularization techniques
    parser.add_argument('--feature_dropout', default = 0., type = float)
    parser.add_argument('--l2_norm', default = 0., type = float)
    parser.add_argument('--output_regularization', default = 0., type = float)

    ## LSTM modules param
    parser.add_argument('--lstm_hidden', default = 100, type = int)
    parser.add_argument('--lstm_num_layers', default = 2, type = int)
    parser.add_argument('--lstm_bidirectional', default = True, type = bool)

    ## DNN modules param
    parser.add_argument('--dnn_hiddens', default = [128, 128, 128], type = list)


    config = parser.parse_args([])
    
    
    config.nam_basis_functions = config.nam_hidden_sizes[0]
    if len(config.nam_hidden_sizes) == 1:
        config.nam_hidden_sizes = None
    else:
        config.nam_hidden_sizes = config.nam_hidden_sizes[1:]

    
    return config


def dataset_name_check(config, dataset_name):
    
    if 'germany' in dataset_name:
        config.data_path = os.path.join(config.dataset_path,'germany','aggregated_dataframe_germany.csv')
        
    elif dataset_name == 'france':
        config.data_path = os.path.join(config.dataset_path,'france','aggregated_dataframe_france.csv')
        
    elif dataset_name == 'belgium':
        config.data_path = os.path.join(config.dataset_path,'belgium','aggregated_dataframe_belgium.csv')
        
    elif dataset_name == 'netherlands':
        config.data_path = os.path.join(config.dataset_path,'netherlands','aggregated_dataframe_netherlands.csv')
        
    elif dataset_name == 'spain':
        config.data_path = os.path.join(config.dataset_path,'spain','aggregated_dataframe_spain.csv')
        
    else:
        config.data_path = os.path.join(config.dataset_path)
        
    return config

def setup_config_with_method(config):
    if config.method in ['Feature', 'Independent', 'Time']:
        config.natm_type = config.method  # Set the NATM type based on the method
    return config

