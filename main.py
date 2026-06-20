# -*- coding: utf-8 -*-
"""
Generalized Linear Models (GLM) - Custom Negative Binomial Regression Engine
Department of Mathematics - Université de Dschang
Multi-Member Project: Wilchy, Gouett, Houssein, Ahmat, John-Kervens
"""

import numpy as np

def irls_negbin(X, y, phi, max_iter=100, tol=1e-6):
    """
    Iteratively Reweighted Least Squares (IRLS) for Negative Binomial Regression.
    Implemented 100% from scratch using purely NumPy matrix operations.
    
    Complexity per iteration: O(n * p^2 + p^3)
    
    Parameters:
    -----------
    X : numpy.ndarray -> Design matrix of shape (n, p)
    y : numpy.ndarray -> Count response vector of shape (n,)
    phi : float       -> Overdispersion parameter (phi > 0)
    """
    n, p = X.shape
    
    # Step 1: Initialization (Boundary protection to avoid log(0) singularity)
    mu = y + 0.5
    eta = np.log(mu)
    beta = np.zeros(p)
    
    for t in range(max_iter):
        mu2 = mu ** 2
        
        # Step 2: Compute Quadratic Variance Vector Function V(mu)
        V = mu + phi * mu2       
        
        # Step 3: Analytical Weight Matrix Diagonal (W)
        W = mu2 / V              
        
        # Step 4: Adjusted Operational Working Dependent Target (z)
        z = eta + (y - mu) / mu  
        
        # Step 5: Solve Weighted Least Squares system via stable LU decomposition
        # Rather than manual inversion (inv(X.T @ W @ X)), we use np.linalg.solve
        XtW_X = (X.T * W) @ X
        XtW_z = (X.T * W) @ z
        beta_new = np.linalg.solve(XtW_X, XtW_z) 
        
        # Step 6: Evaluate L2 / Max Absolute Convergence Criterion
        if np.max(np.abs(beta_new - beta)) < tol:
            return {
                "beta": beta_new,
                "XtW_X": XtW_X,
                "V": V,
                "mu": mu,
                "iterations": t + 1,
                "converged": True
            }
            
        beta = beta_new
        eta = X @ beta
        mu = np.exp(eta)
        
    raise RuntimeError("IRLS algorithm failed to converge within maximum iterations.")


def compute_robust_sandwich_se(X, y, res):
    """
    Computes Huber-White Sandwich Standard Errors to secure model stability 
    against potential misspecifications: Sigma = I^-1 * J * I^-1
    """
    beta = res["beta"]
    XtW_X = res["XtW_X"]
    V = res["V"]
    mu = res["mu"]
    
    # Analytical Bread Matrix (Information Matrix I_hat)
    I_hat = XtW_X 
    
    # Empirical Meat Matrix (Score Outer Product J_hat)
    u_i = ((y - mu) / V)[:, np.newaxis] * mu[:, np.newaxis] * X
    J_hat = u_i.T @ u_i
    
    # Sandwich Multiplication
    I_inv = np.linalg.inv(I_hat)
    Sigma_robust = I_inv @ J_hat @ I_inv
    
    return np.sqrt(np.diagonal(Sigma_robust))


def run_bootstrap_validation(X, y, phi, B=1000):
    """
    Non-parametric Bootstrap framework to secure asymptotic validity 
    against finite small sample distributions (e.g., n=30 von Bortkiewicz).
    """
    n, p = X.shape
    bootstrap_betas = []
    
    print(f"Running non-parametric Bootstrap engine with B={B} resamples...")
    for b in range(B):
        # Resample indices with replacement
        indices = np.random.choice(n, size=n, replace=True)
        X_b, y_b = X[indices], y[indices]
        
        try:
            res_b = irls_negbin(X_b, y_b, phi=phi, max_iter=50, tol=1e-5)
            bootstrap_betas.append(res_b["beta"])
        except RuntimeError:
            continue # Skip rare non-converging degenerate resamples
            
    return np.array(bootstrap_betas)


# =====================================================================
# SIMULATION CASE STUDY: von Bortkiewicz Prussian Cavalry Replication
# =====================================================================
if __name__ == "__main__":
    # Generating synthetic baseline dataset matching von Bortkiewicz profile:
    # Sample Size n=30, Mean = 1.22, Variance = 1.42 (Overdispersion confirmed)
    np.random.seed(42)
    n_samples = 30
    
    # Design Matrix: Intercept + 1 Dummy Covariate (Cavalry Corps structure)
    X_data = np.ones((n_samples, 2))
    X_data[:, 1] = np.random.choice([0, 1], size=n_samples) # Corps Effect
    
    # Empirical count response exhibiting overdispersion (phi = 0.25)
    true_phi = 0.25
    true_beta = np.array([0.25, -0.03])
    
    true_mu = np.exp(X_data @ true_beta)
    # Negative Binomial simulation via Poisson-Gamma mixture simulation
    shape = 1 / true_phi
    gamma_rates = np.random.gamma(shape, true_mu / shape, size=n_samples)
    y_counts = np.random.poisson(gamma_rates)
    
    print("--- GLM DATA PROFILE INITIALIZED ---")
    print(f"Sample Size (n): {n_samples}")
    print(f"Empirical Mean: {np.mean(y_counts):.2f} | Empirical Variance: {np.var(y_counts):.2f}")
    print(f"Dispersion Ratio: {np.var(y_counts)/np.mean(y_counts):.2f}\n")
    
    # Running Custom NumPy IRLS Engine
    model_results = irls_negbin(X_data, y_counts, phi=true_phi)
    robust_errors = compute_robust_sandwich_se(X_data, y_counts, model_results)
    
    print("--- ESTIMATION RESULTS (CUSTOM IRLS) ---")
    print(f"Convergence status: Achieved in {model_results['iterations']} iterations.")
    for i in range(len(robust_errors)):
        z_stat = model_results['beta'][i] / robust_errors[i]
        print(f"Beta[{i}]: {model_results['beta'][i]:.4f} | Robust SE: {robust_errors[i]:.4f} | z-stat: {z_stat:.2f}")
        
    # Running Bootstrap Validation
    boot_distributions = run_bootstrap_validation(X_data, y_counts, phi=true_phi, B=1000)
    lower_bounds = np.percentile(boot_distributions, 2.5, axis=0)
    upper_bounds = np.percentile(boot_distributions, 97.5, axis=0)
    
    print("\n--- 95% BOOTSTRAP CONFIDENCE INTERVALS ---")
    for i in range(X_data.shape[1]):
        print(f"Beta[{i}] 95% Percentile Bound: [{lower_bounds[i]:.4f}, {upper_bounds[i]:.4f}]")
