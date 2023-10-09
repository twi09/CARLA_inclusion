 
import numpy as np  
from typing import List, Optional
import torch 
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.distributions import Uniform
import datetime 
from torch.autograd import Variable
from torch import nn
import torch.optim as optim
from carla.recourse_methods.processing import reconstruct_encoding_constraints

 
def reparametrization_trick(mu,sigma2,device,n_samples,distrib) :
    if distrib=="gaussian" : 
        return(reparametrization_trick_gaussian(mu, sigma2, device,n_samples))
    
    elif distrib=="uniform" : 
        return(reparametrization_trick_uniform(mu, sigma2, device,n_samples))

 

def reparametrization_trick_gaussian(mu, sigma2, device,n_samples):
    
    #var = torch.eye(mu.shape[1]) * sigma2
    std = torch.sqrt(sigma2)
    epsilon = MultivariateNormal(loc=torch.zeros(mu.shape[1]), covariance_matrix=torch.eye(mu.shape[1]))
    epsilon = epsilon.sample((n_samples,))  # standard Gaussian random noise
    ones = torch.ones_like(epsilon)
    random_samples = mu.reshape(-1) * ones.to(device) + (std * epsilon).to(device)
    
    return random_samples


# if U(a,b) --> a+(b-a)x where x in U(0,1)
def reparametrization_trick_uniform(x, sigma2, device,n_samples):
    
    epsilon = Uniform(torch.zeros(x.shape[1]),torch.ones(x.shape[1])).sample((n_samples,))
    ones = torch.ones_like(epsilon)
    random_samples = (x.reshape(-1) - sigma2*ones) + 2*sigma2*ones * epsilon
    
    return random_samples




def compute_invalidation_rate(torch_model, random_samples):
    yhat = torch_model(random_samples.float())[:, 1]
    hat = (yhat > 0.5).float()
    ir = 1 - torch.mean(hat, 0)
    return ir




# Find a perturbation delta for a group G assigned to class pred_class
def Optimize_v2(model,x0,pred_class,delta,sigma2,n_samples,lr,max_iter,robustness_target,robustness_epsilon,cat_feature_indices,binary_cat_features,clamp,t,m,device,lambda_param,number_numerical,distribution,t_max_min=0.5) :
    y_target = torch.zeros(1, n_samples) + 1
    y_target_class = torch.tensor([0,1]).float().to(device)
    # Tensors
    G_target = torch.tensor(y_target).float().to(device)
    lamb = torch.tensor(lambda_param).float()
    # init perturb with wachter counterfactual 
    Perturb = Variable(torch.clone(delta.to(device)), requires_grad=True)
    
    
    #print("x_0",x0)
    #print("number_numerical",number_numerical)
    #print(x0[:,number_numerical:])
    x_cf_new = reconstruct_encoding_constraints(x0+Perturb,cat_feature_indices,binary_cat_features).to(device)
    
    # Set optimizer 
    optimizer = optim.Adam([Perturb], lr, amsgrad=True)
    
    # MSE loss for class term 
    loss_fn = torch.nn.MSELoss()
    
    # Prediction for the counterfactual 
    f_x_binary = model(x_cf_new.float())
    f_x = f_x_binary[:,1-pred_class]
    

    # Compute true invalidation rate on random samples 
    random_samples = reparametrization_trick(x_cf_new, torch.tensor(sigma2), device, n_samples=n_samples,distrib=distribution)
    invalidation_rate = compute_invalidation_rate(model, random_samples)
    
    G = random_samples
    
    # Translated group with reconstruct constraint such as categorical data is either 0 or 1 
    G_new = reconstruct_encoding_constraints(
        G, cat_feature_indices, binary_cat_features
    ).to(device)

    # Mean prediction close to 1 
    compute_robutness = (m + torch.mean(G_target- model((G_new).float())[:,1-pred_class])) / (1-t)
    
   
    t0 = datetime.datetime.now()
    t_max = datetime.timedelta(minutes=t_max_min)
    
    #Lambda = []
    #Dist = []
    #Rob = []
    while (f_x <=t) and (compute_robutness > robustness_target + robustness_epsilon) : 
        it=0
        for it in range(max_iter) :
            
            optimizer.zero_grad()
            
            
            
            x_cf_new = reconstruct_encoding_constraints(x0+Perturb,cat_feature_indices,binary_cat_features)
            
            
            # Prediction for the counterfactual 
            f_x_binary = model(x_cf_new.float())
            f_x = f_x_binary[:,1-pred_class]
             
        
            
            # Hinge loss (negative if under 0)
            #robustness_invalidation[robustness_invalidation < 0] = 0
            
            # Compute true invalidation rate on random samples 
            random_samples = reparametrization_trick(x_cf_new, torch.tensor(sigma2), device, n_samples=n_samples,distrib=distribution)
            invalidation_rate = compute_invalidation_rate(model, random_samples)
            
            
            # New perturbated group translated 
            G = random_samples
            G_new = reconstruct_encoding_constraints(
                G, cat_feature_indices, binary_cat_features
            )
            
            # Compute (m + theta) / (1-t)
            mean_proba =  torch.mean(model((G_new).float())[:,pred_class])
            #compute_robutness = (m + torch.mean(G_target- (model((G_new).float())[:,1-pred_class]))) /(1-t)
            compute_robutness = (m + mean_proba) /(1-t)
            
            # Diff between robustness and targer robustness 
            robustness_invalidation = compute_robutness - robustness_target
            
            
            
            #loss = robustness_invalidation  + (1 - (model((x_cf_new).float())[:,1-pred_class])) + lamb* torch.norm(Perturb,p=1)
            loss = robustness_invalidation**2 + loss_fn(f_x_binary,y_target_class) + lamb* torch.norm(Perturb,p=1)
            
            loss.backward()
            optimizer.step()
            
            
               
                
            
            it += 1
            
        #print("Theta",mean_proba)
        print("compute_robutness",compute_robutness)
        print("robustness_target",robustness_target)
        #print("lambda",lamb)
        print("predicted_class",f_x)
        #Lambda.append(lamb.clone().detach().cpu().numpy())
        #Rob.append((robustness_invalidation**2).detach().cpu().numpy())
        #Dist.append(torch.norm(Perturb,p=1).detach().cpu().numpy())
        if (f_x > t) and ((compute_robutness < robustness_target + robustness_epsilon))  :
            print("Counterfactual Explanation Found")
            break
        
        
        lamb -= 0.25
        
        # Stop if no solution found for different lambda values 
        if lamb <=0 :
            print("No Counterfactual Explanation Found for these lambda values")
            break
        
    final_perturb = Perturb.clone()
    
    #np.savetxt("lambda_values_costERC",np.vstack(Lambda))
    #np.savetxt("rob_cost_ERC",np.vstack(Rob))
    #np.savetxt("dist_cost_ERC",np.vstack(Dist))
    return(final_perturb)




 

def robust_counterfactuals_recourse_v2(torch_model,
    x: np.ndarray,
    delta : np.ndarray,
    number_numerical : int,
    cat_feature_indices: List[int],
    binary_cat_features: bool = True,
    n_samples : int = 500,
    feature_costs: Optional[List[float]] = None,
    lr: float = 0.01,
    lambda_param: float = 1,
    sigma2 : float = 0.01,
    robustness_target : float = 0.3,
    robustness_epsilon : float = 0.01,
    y_target: List[int] = [0, 1],
    n_iter: int = 1000,
    clamp: bool = False,
    t_max_min: float = 0.5,
    t : float = 0.5,
    m : float = 0.1,
    distribution : bool = "gaussian"
    ) -> np.ndarray:
    
    """ 
    This is a description of my algorithm 
    """
    device = "cpu"
    #device = "cuda" if torch.cuda.is_available() else "cpu"
    #device = "cuda:1"
    # Input example as a tensor 
    x0 = torch.from_numpy(x).float().to(device)
    y_target = torch.tensor(y_target).float().to(device)
    # Target class probability
    pred_class = 0
    # Perturbation outputed by wachter 
    delta = torch.from_numpy(delta)
    # Compute the optimization problem 
    perturb = Optimize_v2(torch_model,x0,pred_class,delta,sigma2,n_samples,lr,n_iter,robustness_target,robustness_epsilon,cat_feature_indices,binary_cat_features,clamp,t,m,device,lambda_param,number_numerical,distribution=distribution)
    # New counterfactual
    #print(x0+delta)
    x_new =(x0 + perturb).cpu().detach().numpy().squeeze(axis=0)
    
    return(x_new)









