
from carla.models.api import MLModel
import pandas as pd 
import torch
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.distributions import Uniform
import numpy as np 
import torch.distributions.normal as normal_distribution
from carla.recourse_methods.catalog.w_rip.library import compute_invalidation_rate_closed
from torch.autograd import Variable

# Perturb samples with a normal distribution 
def perturb_sample(x, n_samples, sigma2,distrib="gaussian"):
    if distrib=="gaussian" : 
        return(perturb_sample_gaussian(x, n_samples, sigma2))
    
    elif distrib=="uniform" : 
        return(perturb_sample_uniform(x, n_samples, sigma2))

# Perturb samples with a gaussian distribution  
def perturb_sample_gaussian(x, n_samples, sigma2) :
    # stack copies of this sample, i.e. n rows of x.
    X = x.repeat(n_samples, 1)
    
    # sample normal distributed values
    Sigma = torch.eye(x.shape[1]) * sigma2
    eps = MultivariateNormal(
        loc=torch.zeros(x.shape[1]), covariance_matrix=Sigma
    ).sample((n_samples,))
    
    return X + eps, Sigma



# Perturb samples with an uniform distribution 
def perturb_sample_uniform(x, n_samples, sigma2):
    # stack copies of this sample, i.e. n rows of x.
    X = x.repeat(n_samples, 1)
    
    # sample uniform distribution [x+sigma,x-sigma]
    eps = Uniform(-sigma2*torch.ones(x.shape[1]),sigma2*torch.ones(x.shape[1])).sample((n_samples,))
    
    
    return X + eps, None


def compute_estimate_wachter_rip(df_cfs: pd.DataFrame,model,folder_name : str,n_samples,sigma2,backend) : 
    results = []
    for i, x in df_cfs.iterrows() :
        x = torch.Tensor(x).unsqueeze(0)
        x_new = Variable(x.clone(), requires_grad=True)
        ir_estimate = compute_invalidation_rate_closed(model.raw_model, x_new, torch.tensor(sigma2))
        results.append(ir_estimate.detach().numpy())
    
    df = pd.DataFrame(results)
    df.to_csv(
        f"recourse_invalidation_results/{folder_name}/estimate_ir_test.csv",
        sep=",",
        index=False,
    )
    



def compute_recourse_invalidation_rate(df_cfs: pd.DataFrame,model: MLModel, folder_name : str,n_samples,sigma2,backend,distribution="gaussian") : 

    result = []
    cf_predictions = []
    
    for i, x in df_cfs.iterrows():
        x = torch.Tensor(x).unsqueeze(0)
        X_pert, _ = perturb_sample(x, n_samples, sigma2=sigma2,distrib=distribution)
        if backend == "pytorch":
            prediction = (model.predict(x).squeeze() > 0.5).int()
            cf_predictions.append(prediction.item())
            delta_M = torch.mean(
                (1 - (model.predict(X_pert).squeeze() > 0.5).int()).float()
            ).item()
        else:
            prediction = (model.predict(x).squeeze() > 0.5).astype(int)
            cf_predictions.append(prediction)
            delta_M = np.mean(
                1 - (model.predict(X_pert).squeeze() > 0.5).astype(int)
            )
        
        result.append(delta_M)
    df_cfs["prediction"] = cf_predictions
    
    
    
    
    df = pd.DataFrame(result)
    df.to_csv(
        f"recourse_invalidation_results/{folder_name}/delta_testtest.csv",
        sep=",",
        index=False,
    )
    
    return(df_cfs)
    


def compute_estimator(df_cfs: pd.DataFrame, model: MLModel, folder_name : str,n_samples,sigma2,backend,distribution="gaussian") : 
    
    result = []
    cf_predictions = []
    
    for i, x in df_cfs.iterrows():
        x = torch.Tensor(x).unsqueeze(0)
        X_pert, _ = perturb_sample(x, n_samples, sigma2=sigma2,distrib=distribution)
        if backend == "pytorch":
            delta_M = torch.mean(
                1 - model.predict_proba(X_pert)[:,1].float()
            ).item()
        else :
            delta_M = np.mean(
               1 - model.predict_proba(X_pert)[:,1].float()
            )
        
        result.append(delta_M)
    
    
    
    df = pd.DataFrame(result)
    df.to_csv(
        f"recourse_invalidation_results/{folder_name}/estimator_testtest.csv",
        sep=",",
        index=False,
    )
    
    return(df_cfs)
    
    
 
    




    
    
    