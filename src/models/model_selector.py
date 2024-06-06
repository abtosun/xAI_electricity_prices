from src.models.natm import pl_natm
from src.models.DNN import pl_DNN
from src.models.LSTM import pl_LSTM
from src.models.SCINet import pl_SCINet 

def get_model(config):
    if config.method in ['Feature', 'Independent', 'Time']:
        return pl_natm(config)  # NATM model uses config.natm_type internally
    elif config.method == 'DNN':
        return pl_DNN(config)
    elif config.method == 'LSTM':
        return pl_LSTM(config)
    elif config.method == 'SCINet':
        return pl_SCINet (config)
    else:
        raise ValueError("Unsupported model type")