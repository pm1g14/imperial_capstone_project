# Datasheet

## 1. Motivation

The dataset creation supports the **capstone project**, a task to maximize eight black-box functions using surrogate models. It serves as a history log for the queries and black-box function evaluations to track progress. It can also be used as a benchmarking tool for other regression models.

## 2. Composition

The dataset contains samples for each of the eight separate black-box functions. More specifically, each row of the dataset has the input vector to the model (x1, x2, ..., xn), which is a vector of coordinates depending on the black-box function dimensionality, as well as the black-box function evaluation Y at the specific input coordinates. Input coordinates are in the [0, 1] range and the black-box evaluation value Y is a scalar that can be positive or negative.
**format:** The data is stored as an .npy file (numpy array)
**size:** 

## 3. Collection Process

<!-- Describe how the data was collected -->

## 4. Preprocessing and Uses

<!-- Describe preprocessing steps and intended uses -->

## 5. Distribution and Maintenance

<!-- Describe distribution policies and maintenance procedures -->

