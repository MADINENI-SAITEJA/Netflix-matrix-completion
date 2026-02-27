# Netflix Matrix Completion Problem (ALS)

This project implements a Netflix-style recommender system using 
low-rank matrix factorization optimized via Alternating Least Squares (ALS).

The objective is to complete a sparse user–item rating matrix and predict missing ratings using regularized matrix factorization.

---

## Dataset

**MovieLens 100K**

- 943 users  
- 1682 movies  
- 100,000 ratings  
- Rating scale: 1–5  
- Matrix density ≈ 6.3% (≈ 93.7% sparse)

Official validation split used:
- `u1.base` (training)
- `u1.test` (testing)

Dataset source:  
https://grouplens.org/datasets/movielens/100k/

---

## Problem Formulation

Given a sparse rating matrix:

R ∈ ℝ^(m×n)

We assume a low-rank structure:

R ≈ U Vᵀ

Where:
- U ∈ ℝ^(m×k) → user latent factor matrix  
- V ∈ ℝ^(n×k) → item latent factor matrix  
- k << min(m, n)

We solve the regularized objective:

min_{U,V}  Σ_(i,j ∈ Ω) (R_ij − U_iᵀ V_j)²  
+ λ (||U||² + ||V||²)

Where:
- Ω denotes observed ratings  
- λ controls L2 regularization strength  

---

## Method

- Alternating Least Squares (ALS)
- Ridge-regularized normal equation updates
- Mean-centering for numerical stability
- Sparse masking of observed entries
- RMSE evaluation on validation split

At each iteration:
1. Fix V and solve for U
2. Fix U and solve for V
3. Repeat for a fixed number of iterations

---

## Hyperparameter Tuning

Performed grid search over:

- Latent dimension (k)
- Regularization strength (λ)
- Number of iterations

Best configuration found:

k = 20  
λ = 0.5  
Iterations = 15  

Best Test RMSE:
≈ 1.267

---

## Evaluation Metric

Root Mean Squared Error (RMSE):

RMSE = sqrt( (1/N) Σ (r_ui − r̂_ui)² )

Where:
- r_ui  → true rating  
- r̂_ui → predicted rating  

---

## Project Structure

netflix-matrix-completion/
│
├── als_matrix_completion.py   # ALS implementation  
├── load_data.py               # Dataset loader  
└── data/
    └── ml-100k/               # MovieLens dataset  

---

## Installation

pip install numpy pandas

---

## Run

python als_matrix_completion.py

---

## Key Contributions

- Implemented Alternating Least Squares (ALS) from scratch using NumPy  
- Solved regularized least-squares updates via normal equations  
- Performed systematic hyperparameter tuning  
- Evaluated generalization performance using RMSE on official validation splits  
- Analyzed sparsity characteristics of real-world rating data  

---

## Notes

This project reproduces the classical Netflix Prize-style matrix completion setup using the publicly available MovieLens 100K dataset.
