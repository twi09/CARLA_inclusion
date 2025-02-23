# VCNet and CROCO - Implementation in the CARLA framework

This repository proposed the inclusion of VCNet: Variational Counter Net, and CROCO: Cost-efficient RObust COunterfactuals into the CARLA framework.   
The detailed research papers for VCNet and CROCO are available at the following links:  [VCNet](https://link.springer.com/chapter/10.1007/978-3-031-26387-3_27), [CROCO](https://arxiv.org/abs/2304.12943).  

Both method are implemented in the CARLA framework to conduct comparisons with other counterfactual methods.   

CARLA is a python library to benchmark counterfactual explanation and recourse models. It comes out-of-the box with commonly used datasets and various machine learning models. Designed with extensibility in mind: Easily include your own counterfactual methods, new machine learning models or other datasets. Find extensive documentation [here](https://carla-counterfactual-and-recourse-library.readthedocs.io/en/latest/)! The arXiv paper can be found [here](https://arxiv.org/pdf/2108.00783.pdf).


### Available Datasets

| Name                | Source                                                                                       |
|---------------------|----------------------------------------------------------------------------------------------|
| Adult               | [Source](https://archive.ics.uci.edu/ml/datasets/adult)                                      |
| COMPAS              | [Source](https://www.kaggle.com/danofer/compass)                                             |
| Give Me Some Credit | [Source](https://www.kaggle.com/c/GiveMeSomeCredit/data)                                     |
| HELOC               | [Source](https://community.fico.com/s/explainable-machine-learning-challenge?tabset-158d9=2) |

### Provided Machine Learning Models

| Model        |                                 Description                                  | Tensorflow | Pytorch | Sklearn | XGBoost |
|--------------|:----------------------------------------------------------------------------:|:----------:|:-------:|:-------:|:-------:|
| ANN          | Artificial Neural Network with 2 hidden layers and ReLU activation function. |     X      |    X    |         |         |
| LR           |        Linear Model with no hidden layer and no activation function.         |     X      |    X    |         |         |
| RandomForest |                             Tree Ensemble Model.                             |            |         |    X    |         |
| XGBoost      |                              Gradient boosting.                              |            |         |         |    X    |

### Implemented Counterfactual methods
The framework a counterfactual method currently works with is dependent on its underlying implementation.
It is planned to make all recourse methods available for all ML frameworks . The latest state can be found here:

| Recourse Method                                            | Paper                                                            | Tensorflow | Pytorch | SKlearn | XGBoost |
|------------------------------------------------------------|:-----------------------------------------------------------------|:----------:|:-------:|:-------:|:-------:|
| Actionable Recourse (AR)                                   | [Source](https://arxiv.org/pdf/1809.06514.pdf)                   |     X      |    X    |         |         |
| Causal Recourse                                            | [Source](https://arxiv.org/abs/2002.06278.pdf)                   |     X      |    X    |         |         |
| CCHVAE                                                     | [Source](https://arxiv.org/pdf/1910.09398.pdf)                   |            |    X    |         |         |
| Contrastive Explanations Method (CEM)                      | [Source](https://arxiv.org/pdf/1802.07623.pdf)                   |     X      |         |         |         |
| Counterfactual Latent Uncertainty Explanations (CLUE)      | [Source](https://arxiv.org/pdf/2006.06848.pdf)                   |            |    X    |         |         |
| CRUDS                                                      | [Source](https://finale.seas.harvard.edu/files/finale/files/cruds-_counterfactual_recourse_using_disentangled_subspaces.pdf)                                                       |            |    X    |         |         |
| Diverse Counterfactual Explanations (DiCE)                 | [Source](https://arxiv.org/pdf/1905.07697.pdf)                   |     X      |    X    |         |         |
| Feasible and Actionable Counterfactual Explanations (FACE) | [Source](https://arxiv.org/pdf/1909.09369.pdf)                   |     X      |    X    |         |         |
| FeatureTweak                                               | [Source](https://arxiv.org/pdf/1706.06691.pdf)                   |            |         |    X    |    X    |
| FOCUS                                                      | [Source](https://arxiv.org/pdf/1911.12199.pdf)                   |            |         |    X    |    X    |
| Growing Spheres (GS)                                       | [Source](https://arxiv.org/pdf/1712.08443.pdf)                   |     X      |    X    |         |         |
| Revise                                                     | [Source](https://arxiv.org/pdf/1907.09615.pdf)                   |            |    X    |         |         |
| Wachter                                                    | [Source](https://arxiv.org/ftp/arxiv/papers/1711/1711.00399.pdf) |            |    X    |         |         |
| VCNet                                                      | [Source](https://arxiv.org/abs/2212.10847)                       |            |    X    |         |         |
| CROCO                                                      | [Source](https://arxiv.org/abs/2304.12943)                       |            |    X    |         |         |
## Installation

### Requirements

- `python3.7`
- `pip`

### Install 
Run in the root folder (where the setup.py file is):
```sh
pip install -e .
```
## VCNet functioning

### Select a VCNet model 
Two versions of VCNet can be selected: 
* The original version of the arcticle (immutable=False)
* A specific version that handle immutable features (immutable=True)
```python
from carla.self_explaining_model import VCNet
ml_model = VCNet(data_catalog,hyperparams,immutable_features,immutable=False)
```

### Hyperparameters and models 
The hyperparameters values for each dataset are provided in [HYPERPARAMETERS_original](carla/self_explaining_model/catalog/vcnet/library/vcnet_tabular_data_v0/hyperparameters) and [HYPERPARAMETERS_immutable](carla/self_explaining_model/catalog/vcnet/library/vcnet_tabular_data_v1/hyperparameters) for the original and specific version respectively.

The models weights are provided in [MODELS_original](carla/self_explaining_model/catalog/vcnet/library/vcnet_tabular_data_v0/save_models) and [MODELS_immutable](carla/self_explaining_model/catalog/vcnet/library/vcnet_tabular_data_v1/save_models) respectively. 

### Run benchmark with VCNet


```python
from carla.models.negative_instances import predict_negative_instances
from carla.data.catalog import OnlineCatalog
from carla import MLModelCatalog
from carla import Benchmark
from carla.self_explaining_model.catalog.vcnet.library.utils import *
from carla.self_explaining_model import VCNet
from carla.self_explaining_model.catalog.vcnet.library.utils import fix_seed
from carla.recourse_methods import Face
from carla.evaluation.catalog import Distance, Redundancy, SuccessRate, AvgTime, ConstraintViolation, YNN 
import os 


# Load a dataset from the carla framework 
name = "adult"
data_catalog = OnlineCatalog(name,encoding_method="OneHot_drop_binary")


# Immutable features 
immutable_features = data_catalog.immutables

# Fix the seed
fix_seed()
# Define hyperparams for counterfactual search : 
hyperparams = {   
    "name" : name ,
    "vcnet_params" : {
    "train" : False,
    "lr":  1.14e-5,
    "batch_size": 91,
    "epochs" : 174,
    "lambda_1": 0,
    "lambda_2": 0.93,
    "lambda_3": 1,
    "latent_size" : 19,
    "latent_size_share" :  304, 
    "mid_reduce_size" : 152,
    "kld_start_epoch" : 0.112,
    "max_lambda_1" : 0.034
    }
}

# Instantiate a VCNet model 
ml_model = VCNet(data_catalog,hyperparams,immutable_features,immutable=False)

# Select a subsample (here the 100 first) of test instances that are predicted class 0
factuals_drop = predict_negative_instances(ml_model, data_catalog.df_test.drop(columns=[data_catalog.target])).iloc[:100].reset_index(drop=True)


# Benchmark VCNet 
benchmark = Benchmark(ml_model,ml_model,factuals_drop)

# Load metrics from carla 
distances = Distance(ml_model)
success_rate = SuccessRate()
constraint_violation = ConstraintViolation(ml_model)
ynn = YNN(ml_model,{"y" : 5, "cf_label" : 1})

# Run the benchmark 
results = benchmark.run_benchmark([success_rate,distances,constraint_violation,ynn])

# Save the results 
outname = f'results_VCNet.csv'
outdir = f'./carla_results/vcnet/{data_catalog.name}'
if not os.path.exists(outdir):
    os.makedirs(outdir)
fullname = os.path.join(outdir, outname)
results.to_csv(fullname)
```
## CROCO functioning 

### Run benchmark with CROCO 
```python
from carla.data.catalog import OnlineCatalog
from carla.models.catalog import MLModelCatalog
from carla.models.negative_instances import predict_negative_instances
from carla.recourse_methods.catalog.croco import CROCO
import os 
from carla import Benchmark
from carla.evaluation.catalog import Distance, Redundancy, SuccessRate, AvgTime, ConstraintViolation, YNN, Robustness


# Load a dataset from the carla framework 
name = "adult"
data_catalog = OnlineCatalog(name,encoding_method="OneHot_drop_binary")

## Load and train a machine learning model from the framework 
model_type = "ann"


# Params for neural network model 
training_params_ann = {
    "adult": {"lr": 0.002,
              "epochs": 30,
              "batch_size": 1024},
    "compas": {"lr": 0.002,
               "epochs": 25,
               "batch_size": 25},
    "give_me_some_credit": {"lr": 0.002,
                            "epochs": 50,
                            "batch_size": 2048},
    "breast_cancer": {"lr": 0.002,
               "epochs": 25,
               "batch_size": 32} }

params = training_params_ann[name]
model = MLModelCatalog(
        data_catalog, model_type, load_online=False, backend="pytorch"
    )
model.train(
        learning_rate=params["lr"],
        epochs=params["epochs"],
        batch_size=params["batch_size"],
        hidden_size=[50],
    )
model.use_pipeline = False


# Select a subsample (here the 10 first) of test instances that are predicted class 0
factuals_drop = predict_negative_instances(ml_model, data_catalog.df_test.drop(columns=[data_catalog.target])).iloc[:10].reset_index(drop=True)


 # Hyperparameters for croco 
hyperparams_croco = {"n_samples" : 10_000,
                "binary_cat_features": True,
                "sigma2": 0.01,
                "n_iter": 200,
                "robustness_target" : 0.35,
                "m" : 0.1,
                "robustness_epsilon" : 0.01,
                "lambda_param" : 1,
                "distribution" : "gaussian"}

# CROCO method 
croco = CROCO(model,hyperparams_croco)


# Benchmark CROCOs 
benchmark = Benchmark(model,croco,factuals_drop)

# Load metrics from carla 
distances = Distance(model)
success_rate = SuccessRate()
constraint_violation = ConstraintViolation(model)
ynn = YNN(model,{"y" : 5, "cf_label" : 1})
# Robustness metric (recourse invalidation rate)
robustness = Robustness(model,{"n_samples" : 10_000, "sigma2" : 0.01, "distribution" : "gaussian"  })

# Run the benchmark 
results = benchmark.run_benchmark([success_rate,distances,constraint_violation,robustness,ynn])


# Save the results 
outname = f'results_CROCO.csv'
outdir = f'./carla_results/croco/{data_catalog.name}'
if not os.path.exists(outdir):
    os.makedirs(outdir)
fullname = os.path.join(outdir, outname)
results.to_csv(fullname)
```


## Contributing

### Requirements

- `python3.7-venv` (when not already shipped with python3.7)
- Recommended: [GNU Make](https://www.gnu.org/software/make/)

### Installation

Using make:

```sh
make requirements
```

Using python directly or within activated virtual environment:

```sh
pip install -U pip setuptools wheel
pip install -e .
```

### Testing

Using make:

```sh
make test
```

Using python directly or within activated virtual environment:

```sh
pip install -r requirements-dev.txt
python -m pytest test/*
```


## Licence

VCNet is under the MIT Licence. See the [LICENCE](github.com/indyfree/carla/blob/master/LICENSE) for more details.
CROCO is under the MIT Licence. See the [LICENCE](github.com/indyfree/carla/blob/master/LICENSE) for more details.

## Citation

VCNet came from a paper published to ECML/PKDD 2022.
If you conduct comparison with it, please cite: 
```sh
@inproceedings{Guyomard2022VCNetAS,
  title={{VCNet}: A self-explaining model for realistic counterfactual generation},
  author={Victor Guyomard and Françoise Fessant and Thomas Guyet},
  booktitle={Proceedings of the European Conference on Machine Learning and Principles and Practice of Knowledge Discovery in Databases (ECML/PKDD)},
  pages={437--453},
  location={Grenoble, Fr},
  year={2022}
}
```
CROCO came from a paper published to ECML/PKDD 2023.
If you conduct comparison with it, please cite: 
```sh
@inproceedings{CROCO,
  title={Generating robust counterfactual explanations},
  author={Victor Guyomard and Françoise Fessant and Thomas Guyet},
  booktitle={Proceedings of the European Conference on Machine Learning and Principles and Practice of Knowledge Discovery in Databases (ECML/PKDD)},
  location={Turin, Italy},
  year={2023}
}
```
The CARLA framwork that is used for implementation, came from a project accepted to NeurIPS 2021 (Benchmark & Data Sets Track).
If you use this codebase, please cite:

```sh
@misc{pawelczyk2021carla,
      title={CARLA: A Python Library to Benchmark Algorithmic Recourse and Counterfactual Explanation Algorithms},
      author={Martin Pawelczyk and Sascha Bielawski and Johannes van den Heuvel and Tobias Richter and Gjergji Kasneci},
      year={2021},
      eprint={2108.00783},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```
