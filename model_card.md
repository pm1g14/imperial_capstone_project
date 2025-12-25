# Model Card BBO Capstone Project

## 1. Overview

**Name:** Heteroscedastic Gaussian Processes Bayesian Optimization  
**Type:** Bayesian Optimization surrogate model  
**Version:** 2.0

## 2. Intended Use

- Hyperparameter optimization of ML models
- Black-box optimization where gradients are unavailable
- Noisy objective functions with variable noise
- Moderate dimensionality objective functions up to 8 dimensions
- Problems with limited evaluation budgets

**Note:** The model should not be used for very high-dimensional problems (> 11 dimensions). It should also not be used for discrete or categorical-only search spaces (without encoding). Finally, it cannot be used for objective functions with discontinuities or non-stationary structure that cannot be captured by a stationary kernel.

## 3. Details

The model fits two SingleTask Gaussian Processes and uses an ARD Matern kernel for both of them, with configurable nu. The two models are:
- The mean Gaussian Process is used to estimate the latent objective function f(x).
- The noise Gaussian Process is used to model heteroscedastic noise and is trained on log residuals.  
The predictive posterior combines the mean GP ovariance and the learned, input-dependent noise variance.  
Earlier versions (rounds 1-5), assumed a constant noise across the entire input space and did not account for non-stationarity. The 2.0 version described above (rounds 6-10), accounts for variable noise and uses a warping technique to account for non-stationarity of the black-box function.  
Rounds 1-3 focused on exploration using UCB with rounds 4-6 turning towards NEI and exploitation. Rounds 7 and 8 were more exploratory using Integrated Posterior Variance, before turning towards exploitation on round 9.

## 4. Performance

**Metrics:** 
- Evaluated via improvement over best observed objective value.
- Evaluated using the **Optuna framework** against functions with known best value.
- Diagnostic metrics include acquisition value progression, residuals to measure surrogate quality (RMSE), GP lengthscale evolution, and trust-region radius dynamics.

## 5. Assumptions and Limitations

**Assumptions:** The model assumes variable noise and non-stationarity across the input space X. It also assumes input variables are normalised in the [0, 1] range.  

**Limitations:** The model can handle up to 11 dimensions. The current approach, does not scale well for higher input dimensions (> 11) or input values that are not normalised in the [0, 1] range. Acquisition optimization can fail under numerical instability. This is true, especially when trying to maximize the acquisition over smaller areas. The model is often unstable and sensitive to kernel hyperparameter bounds and trust-region radius changes. These need to be carefully tuned and monitored to ensure a relatively stable outcome. The model also exhibits some bias because of gaps in the underlying data and poor sampling in earlier rounds of the competition. 

## 6. Ethical considerations

The model has no direct impact on any group of people or individuals. The risks relate mostly on wasted compute or poor optimization efficiency. 

**Transparency:** A Gaussian Processes surrogate model can be easily explained in contrast with surrogate models that are based on neural networks. It has clear command line arguments and hyperparameters, such as lengthscales and noise variance that can be inspected and monitored over time. The acquisition functions used are well-known in the literature and the process does not involve complex or hidden heuristics. This transparency enables users to understand the *Why?* behind specific query points that are proposed, diagnose failure modes, and reason about the exploration-exploitation trade-off.

**Adaptation:** The transparency of the model facilitates adaptation to new similar problem settings (variable noise, non-stationarity). Acquisition functions and trust-region parameters can be changed to reflect different optimization goals. As a result, the approach is not a fixed black-box optimizer, but a configurable and extensible optimization framework. 

**Reproducibility:** The results can be reproduced via deterministic model training and acquisition optimization when random seeds are fixed. Because the BO loop *(data -> surrogate -> acquisition -> candidate)* is deterministic given seeds and configuration, experiments can be reliably repeated and compared across runs and environments.

