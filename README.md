# GLM: Negative Binomial Regression Engine from Scratch

This repository contains the official computational implementation for our Generalized Linear Models (GLM) project at **Université de Dschang**.

## Group Members
* Francois Wilchy
* Gouett Bitjocka
* Houssein Moustapha
* Issa Brahim Ahmat
* Jacquet John-Kervens

## Core Features
* **IRLS Optimization:** Iteratively Reweighted Least Squares engine coded entirely from scratch using standard matrix operations in `NumPy`.
* **Robust Inference:** Implementation of the empirical Huber-White Sandwich Covariance Estimator to secure valid standard errors under potential model misspecifications.
* **Boundary Safe:** Built-in numerical protection during initialization to eliminate mathematical singularities ($\log(0)$ errors).
