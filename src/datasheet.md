# Datasheet

## 1. Motivation

The dataset creation supports the **capstone project**, a task to maximize eight black-box functions using surrogate models. It serves as a history log for the queries and black-box function evaluations to track progress. It can also be used as a benchmarking tool for other regression models.

## 2. Composition

The dataset contains samples for each of the eight separate black-box functions. More specifically, each row of the dataset has the input vector to the model (x1, x2, ..., xn), which is a vector of coordinates depending on the black-box function dimensionality, as well as the black-box function evaluation Y at the specific input coordinates. Input coordinates are in the [0, 1] range and the black-box evaluation value Y is a scalar that can be positive or negative.
**format:** The data is stored as an .npy file (numpy array)  

**size:** The size of the dataset is 80 query-evaluation pairs (10 rounds x 8 functions)  

**dimensions:** The black-box functions range from 2D to 8D

**Note:** There are gaps in the dataset, especially for the high-dimensional functions since they create a hypercube and input data is scarce. The gaps are generally influenced by the acquisition function strategy used to generate the dataset.

## 3. Collection Process

The queries were generated using an iterative Bayesian Optimisation Gaussian Processes approach.  

**strategy:** 
- Rounds (1-4) queries were generated using a surrogate model assuming constant noise and stationary functions.
- Rounds (5-7) queries were generated using a surrogate model assuming variable noise.
- Rounds (7-13) queries were generated using a surrogate model assuming variable noise and non-stationarity. A Trust-region BO is also used to focus on promising areas.

**timeframe:** The dataset was created over the course of the competition from September to December 2025. A new query point and evaluation pair for each function was generated each week.

## 4. Preprocessing and Uses

**Preprocessing:** 
- Inputs are normalized in the [0, 1] range
- Outputs are raw function evaluations

For the generation of the datapoints, both inputs and outputs are warped to help the surrogate model perform better on non-Gaussian like distributions. However, this transformation is not directly reflected in the resulting dataset.

**Intended Uses:** The dataset can serve as part of a benchmarking process to evaluate the performance of similar surrogate models

**Innapropriate Uses:** The dataset cannot be used to perform general statistical inference or global predictive modeling.

## 5. Distribution and Maintenance

The dataset is publicly available on https://github.com/pm1g14/imperial_capstone_project  

**Licence:** MIT Licence  

**Maintainer:** The author of the specific github repository.

