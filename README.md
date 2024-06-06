# Explainable Deep Learning for Analysis and Forecasting of Multivariate Time Series: Applications to Electricity Markets

**Type:** Master's Thesis 

**Author:** Ali Burak Tosun

**1st Examiner:** Prof. Dr. Stefan Lessmann

**2nd Examiner:** Prof. Dr. Benjamin Fabian

###  Experimental Results in Terms of Rank

| Length | Metric | NATM<sub>Feature</sub> | DNN | NATM | LSTM | SCINet | NATM<sub>Time</sub> |
|--------|--------|------------------------|-----|------|------|--------|---------------------|
| 8      | MAE    | 2.6                    | 5.8 | 3.6  | 5.2  | 1.0    | 2.8                 |
|        | RMSE   | 2.8                    | 5.6 | 3.8  | 5.4  | 1.0    | 2.4                 |
|        | SMAPE  | 3.2                    | 5.8 | 3.2  | 4.6  | 1.0    | 3.2                 |
|        | R<sup>2</sup> | 3.0            | 5.4 | 3.4  | 5.6  | 1.0    | 2.6                 |
| 16     | MAE    | 2.4                    | 6.0 | 3.6  | 5.0  | 1.0    | 3.0                 |
|        | RMSE   | 2.2                    | 5.8 | 3.6  | 5.2  | 1.0    | 3.2                 |
|        | SMAPE  | 3.0                    | 6.0 | 3.0  | 4.6  | 1.0    | 3.4                 |
|        | R<sup>2</sup> | 2.4            | 5.8 | 2.8  | 5.2  | 1.0    | 3.8                 |
| 24     | MAE    | 2.6                    | 6.0 | 3.0  | 5.0  | 1.0    | 3.4                 |
|        | RMSE   | 2.4                    | 5.6 | 3.2  | 5.4  | 1.0    | 3.4                 |
|        | SMAPE  | 2.2                    | 6.0 | 3.0  | 4.6  | 1.2    | 4.0                 |
|        | R<sup>2</sup> | 2.4            | 5.8 | 3.0  | 5.2  | 1.0    | 3.6                 |
| **Overall Avg. Rank** |        | **2.6**              | **5.8** | **3.3** | **5.1** | **1.0** | **3.2**              |

## Table of Content

- [Summary](#summary)
- [Working with the repo](#working-with-the-repo)
    - [Dependencies](#dependencies)
    - [Setup](#setup)
- [Reproducing results](#reproducing-results)
    - [Training and Evaluation code](#training-and-evaluation-code)
    - [Visualization code](#visualization-code)
    - [Pretrained models](#pretrained-models)
- [Results](#results)
- [Project structure](#project-structure)

## Summary

Electricity Price Forecasting (EPF) is crucial for renewable energy stakeholders. This thesis explores Neural Additive Time-series Models (NATMs)[^1] for forecasting electricity prices in Germany, France, Belgium, Spain, and the Netherlands. NATMs offer accurate predictions with interpretability, identifying contributions of different input variables. Using datasets from the ENTSO-E Transparency Platform[^3] (2020-2023), NATMs are compared with Long Short-Term Memory (LSTM) networks, Deep Neural Networks (DNNs), and SCINet[^2] using metrics like MAE, RMSE, SMAPE, and R². Results show SCINet achieves the highest predictive accuracy, but NATMs perform well with added interpretability. Contribution maps from NATMs show significant insights into the impact of historical prices and generation forecasts on electricity prices. This research demonstrates NATMs' potential in enhancing EPF accuracy and transparency.

**Keywords**: Electiricity Price Forecasting, Explainable Artificial Intelligence, Deep Learning, Multivariate Time-series Prediction, Neural Additive Time-series Models

**Full text**: The full text for this work is available here.

[^1]: [Neural Additive Time-series Models (NATMs)](https://github.com/merchen911/NATM)
[^2]: [SCINet](https://github.com/cure-lab/SCINet)
[^3]: [ENTSO-E Transparency Platform](https://transparency.entsoe.eu/)

## Working with the repo

### Dependencies

The environment was set up using Python 3.8 and Conda 4.12, though it should be compatible with similar versions.

### Setup

1. **Clone this repository:**

    ```bash
    git clone https://github.com/abtosun/xAI_electricity_prices.git
    cd xAI_electricity_prices
    ```
2. **Create and activate a virtual environment:**

    ```bash
    conda env create -f environment.yml
    conda activate myenv
    ```
## Reproducing results

The reproduction of results is divided into two main parts: Training & Evaluation, and Visualization. Follow the steps below to replicate the results for each part:

### Training and Evaluation code

The `Learning.py` script includes both the training and evaluation phases of the models. You can run the `Learning.py` script with different methods, datasets, and input lengths. Alternatively, you can run the `Learning.ipynb` notebook, but you will need to change parameters in `config.py`. Here are the details:

#### Methods, Datasets, and Input Lengths

| method        | dataset_name  | input_length  |
|---------------|---------------|---------------|
| `Independent` | `germany`     | `8`           |
| `Feature`     | `netherlands` | `16`          |
| `Time`        | `belgium`     | `24`          |
| `DNN`         | `spain`       |               |
| `SCINet`      | `france`      |               |
| `LSTM`        |               |               |


1. **Make sure the Conda environment is activated or selected as a kernel**

2. **Prepare the data**

    Ensure the sample data is correctly placed in the `sample_data` directory. This directory should contain subdirectories for each dataset (e.g., `germany`, `netherlands`, etc.) with the respective CSV files.

3. **Run the `Learning.py` script**

#### Example Commands

Ensure that you have the necessary permissions and access to the GPUs on your servers to execute the experiments efficiently.
To run the code for different combinations, use the following commands:

```bash
# Method: Independent, Dataset: Germany, Input Length: 24
python Learning.py --method Independent --dataset_name germany --input_length 24 

# Method: Feature, Dataset: Spain, Input Length: 24
python Learning.py --method Feature --dataset_name spain --input_length 24 

# Method: SCINet, Dataset: Belgium, Input Length: 24
python Learning.py --method SCINet --dataset_name belgium --input_length 24 
```
In each run of `Learning.py`, the models will be trained and evaluated for 5 different folds. After each run, a results folder will be created, and the evaluation metrics will be saved in a CSV file.

### Visualization Code

The visualization code is used to generate visualizations for the results of the NATMs (Independent, Feature, Time) in EPF. Contribution maps for NATMs are created to provide insights into the model's behavior and feature importance.

The visualizations are generated using a Jupyter Notebook:
1. **Make sure the Conda environment is activated or selected as a kernel**

2. **Unzip the checkpoints and logs:**
    ```bash
    unzip ckpt.zip -d ckpt
    ```

3. **Run the Jupyter Notebook:**
    
    In the Notebook:
    - Change the `base_path` to the desired method, dataset, and input length. For example:
    
      ```python
      base_path = 'ckpt/test_Feature_netherlands_8'
      ```

    - Set a valid date in the validation set for `input_date`. For example:
    
      ```python
      input_date = '2023-07-17 09:00:00'
      ```

These changes will ensure that contribution maps are generated for the specified method, dataset, and input length.

### Pretrained models

You can access the pretrained model weights and logs of NATMs in the `ckpt.zip` file. This archive are essential for visualization and understanding the model performance.

## Results

All evaluation metrics of the experiments can be found in the `results` folder named as [`evaluation_metrics_results.csv`](results/evaluation_metrics_results.csv). This file provides detailed metrics for each experiment, allowing for easy reproduction and analysis of the results. 

## Project structure

The project has a following structure:
```bash
.
├── ckpt.zip                                        # Model checkpoints
├── environment.yml                                 # Environment configuration file
├── Learning.ipynb                                  # Jupyter notebook for learning experiments
├── Learning.py                                     # Python script for learning experiments
├── README.md                                       # The main documentation file for the project
├── results                                         # Directory for storing result files
│   └── evaluation_metrics_results.csv              # CSV file containing evaluation metrics results
├── sample_data                                     # Directory for storing sample data files
│   ├── belgium
│   │   └── aggregated_dataframe_belgium.csv        # Aggregated data for Belgium
│   ├── france
│   │   └── aggregated_dataframe_france.csv         # Aggregated data for France
│   ├── germany
│   │   └── aggregated_dataframe_germany.csv        # Aggregated data for Germany
│   ├── netherlands
│   │   └── aggregated_dataframe_netherlands.csv    # Aggregated data for Netherlands
│   └── spain
│       └── aggregated_dataframe_spain.csv          # Aggregated data for Spain
├── src                                             # Source code directory containing scripts and modules
│   ├── config.py                                   # Configuration file for the project
│   ├── data_prepare.py                             # Data preparation script
│   ├── __init__.py                                 # Init file for the src module
│   └── models                                      # Directory for model-related scripts
│       ├── activation
│       │   ├── exu.py                              # EXU activation function
│       │   ├── __init__.py                         # Init file for activation functions
│       │   └── relu.py                             # ReLU activation function
│       ├── base.py                                 # Base model script
│       ├── DNN.py                                  # Deep Neural Network model script
│       ├── featurenn.py                            # Feature neural network script
│       ├── __init__.py                             # Init file for the models module
│       ├── LSTM.py                                 # Long Short-Term Memory model script
│       ├── model_selector.py                       # Model selection script
│       ├── natm.py                                 # NATM model script
│       └── SCINet.py                               # SCINet model script
├── structure.txt                                   # File containing the project structure
└── Visualization.ipynb                             # Jupyter notebook for visualization experiments
```