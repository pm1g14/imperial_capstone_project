## Professional Certificate in Machine Learning and AI - Capstone Project
This repository is my implementation of the capstone project for the ML and Artificial Intelligence Professional Certificate.

The model is used to optimize eight black box functions resembling real-world problems, like finding contamination sources in a field, or optimizing allocation of resources in a warehouse. Since these problems are expensive and costly to evaluate, we want to find the maximum of each function in the least number of trials. My model relies on early exploration and builds up on the results of each round to find more optimal setups for the aforementioned problems. 

## Data
The **input data** were given as part of the challenge. Depending on the input features, they have the following format:
`x1, x2, ....., y`
where y is the objective function evaluation for the corresponding x values.

The **output data**, when in the evaluation mode, is the next suggested point to evaluate in the format:
`x1-x2-x3-...xn` for a n-dimensional problem.

More information on the dataset and its format is provided in the datasheet.md in the [Supporting Documentation](#Supporting Documentation) section

## Model
**Model Type:** Heteroscedastic Gaussian Processes Surrogate Model.

The model is a Gaussian Processes using two surrogate models, one to model the posterior mean and another to model the variable noise. It also uses popular algorithms such as gradient ascent + trust region BO
for exploitation of promising regions. The model accounts for non-stationarity via input warping + trust region BO. 
I chose this particular model because I found it more suitable for the capstone project setup (small datasets, few evaluations) compared to other approaches like deep neural networks that are more data-hungry. I also found it easier to experiment with more advanced features of GP BO due to initial limited knowledge of the field of hyperparameter optimization.

## Hyperparameters
Below is a list of the hyperparameters for my model:

| Hyperparameters | Description |
| :---: | :---: |
| nu_mean | The estimated wiggliness of the function |
| nu_noise | The estimated wiggliness of the noise function |
| warp_inputs | Whether to warp the input x |
| warp_outputs | Whether to warp the outputs y |
| bounds | The area for which to calculate the acquisition function max |
| acquisition_function_str | The acquisition function to maximize (qLogNEI, UCB or qNegIntegratedPosteriorVariance) |
| raw_samples | The number of starting points for the gradient ascent algorithm |
| num_restarts | The number of restarts when running the gradient ascent algorithm |

## How to Run
Clone the git repo and run:

`pipenv install`

to install dependencies.

There are 2 modes that can run, the "evaluate" mode that suggests the next evaluation point and the "update" mode that's applicable for Trust Region BO, to update the search area bounds.
For the "evaluate" mode, run:

`python capstone_imperial/src/app.py evaluate`

Input parameters for this mode are:
| Name | Description |
| :---: | :---: |
| function-number | A unique identifier for the function to evaluate. |
| dimensions | The number of input parameters. |
| total-budget | The total number of trials. |
| trial-no | The trial number now. |
| input-dataset-path | The path to the input dataset. Must be in a .npy file format. |
| output-dataset-path | The existing evaluations file, if any. Must be an .npy file. |
| submission-path | The path to the trial evaluations file. |


EX.
`python capstone_imperial/src/app.py evaluate 1 2 13 --trial-no 11`

For the "update" mode, run:

`python capstone_imperial/src/app.py update`

Input parameters for this mode are:
| Name | Description |
| :---: | :---: |
| y-new | The latest trial evaluation. |
| function-number | A unique identifier for the function to update. Must match the identifier used in the evaluate mode. |
| dimensions | The number of input parameters. |

EX.
`python capstone_imperial/src/app.py update -1.2481615507832458e-21 1 2`

## Results
Below is the summary of the model's performance for the 8 functions:
| Function | Max | Trial reached | Improvement |
| :---: | :---: | :---: | :---: |
| function 1 | 2.5485375317843807e-7 | 11 | |
| function 2 | 0.6918590194346371 | 13 | |
| function 3 | -0.0030721660466671236 | 13 | |
| function 4 | -1.8207573488402562 | 13 | |
| function 5 | 8662.4825 | 7 | |
| function 6 | -0.30782271999440225 | 12 | |
| function 7 | 2.0505603049107615 | 13 | |
| function 8 | 9.9537948611856 | 11 | |

Most max values were identified close to the end of the capstone project, an indication that exploitation should have started at an earlier iteration. Trust Region BO was applied on round 9 for most of the functions, and there were not enough trials left to get better results. 

## Supporting Documentation
[Datasheet for the Dataset](https://github.com/pm1g14/imperial_capstone_project/edit/main/src/datasheet.md) - Details about the dataset, data collection, format of the data, and intended uses.  
[Model Card](https://github.com/pm1g14/imperial_capstone_project/blob/main/src/model_card.md) - Details about the model, the strategy, performance, and limitations.
