from carla.recourse_methods.api import RecourseMethod
from carla.recourse_methods.catalog.robust_counterfactuals.library import robust_counterfactuals_recourse_v2
import pandas as pd 
from carla.recourse_methods.processing import (
    check_counterfactuals,
    merge_default_parameters,
)
import numpy as np

class Robust_counterfactuals(RecourseMethod) : 
    """
    This is a description 
    """
    
    _DEFAULT_HYPERPARAMS = {
        "n_samples" : 500,
        "feature_cost": "_optional_",
        "lr": 0.01,
        "t" : 0.5, 
        "m" : 0.1,
        "lambda_param": 1,
        "n_iter": 1000,
        "t_max_min": 0.5,
        "loss_type": "BCE",
        "y_target": [0, 1],
        "binary_cat_features": True,
        "clamp" : True,
        "sigma2" : 0.01, 
        "init_random" : False,
        "version" : "v1",
        "robustness_target" : 0.3,
        "robustness_epsilon" : 0.01,
        "distribution" : "gaussian"
    }
    
    
    def __init__(self, mlmodel, hyperparams):
        super().__init__(mlmodel)
    
        checked_hyperparams = merge_default_parameters(
            hyperparams, self._DEFAULT_HYPERPARAMS
        )
        self.n_samples = checked_hyperparams["n_samples"]
        self._feature_costs = checked_hyperparams["feature_cost"]
        self._lr = checked_hyperparams["lr"]
        self._lambda_param = checked_hyperparams["lambda_param"]
        self._n_iter = checked_hyperparams["n_iter"]
        self._t_max_min = checked_hyperparams["t_max_min"]
        self._loss_type = checked_hyperparams["loss_type"]
        self._y_target = checked_hyperparams["y_target"]
        self._binary_cat_features = checked_hyperparams["binary_cat_features"]
        self._sigma2 =  checked_hyperparams["sigma2"]
        self._clamp = checked_hyperparams["clamp"]
        self._init_random =  checked_hyperparams["init_random"]
        self._version = checked_hyperparams["version"]
        self.robustness_target = checked_hyperparams["robustness_target"]
        self.robustness_epsilon = checked_hyperparams["robustness_epsilon"]
        self.distribution = checked_hyperparams["distribution"]
        self.m = checked_hyperparams["m"]
        self.t = checked_hyperparams["t"]
        
    def get_counterfactuals(self,factuals: pd.DataFrame) -> pd.DataFrame :
            
        encoded_feature_names = self._mlmodel.data.encoder.get_feature_names(
            self._mlmodel.data.categorical
        )
        cat_features_indices = [
            factuals.columns.get_loc(feature)
            for feature in encoded_feature_names
        ]
        
        
        # Number of numerical features 
        number_numerical = len(self._mlmodel.data.continuous)
        
        # Compute robust counterfactuals for every x instance based on the perturbation c outuputed by a given counterfactual algorithm 
        df_cfs_new = factuals.copy()
        for index, x in factuals.iterrows() :
                # Init perturb as zeros 
                perturb_init = np.zeros(x.shape)
                
                df_cfs_new.loc[index] = robust_counterfactuals_recourse_v2(self._mlmodel.raw_model,
                                                      np.array(x).reshape((1, -1)),
                                                      perturb_init,
                                                      number_numerical,
                                                      cat_features_indices,
                                                      binary_cat_features=self._binary_cat_features,
                                                      n_samples = self.n_samples,
                                                      feature_costs=self._feature_costs,
                                                      lr=self._lr,
                                                      lambda_param=self._lambda_param,
                                                      sigma2 = self._sigma2,
                                                      clamp=self._clamp,
                                                      robustness_target = self.robustness_target,
                                                      robustness_epsilon = self.robustness_epsilon,
                                                      y_target = self._y_target,
                                                      n_iter=self._n_iter,
                                                      t_max_min=self._t_max_min,
                                                      t = self.t,
                                                      m = self.m, 
                                                      distribution = self.distribution
                                                      )
            
        
        
        df_cfs_new = check_counterfactuals(self._mlmodel, df_cfs_new, factuals.index)
            
        return(df_cfs_new)
            
            
            
            
            
 