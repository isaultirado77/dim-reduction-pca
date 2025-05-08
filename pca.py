"""PCA implementation from scratch."""

"""
TODO: 
    - Write documentation
"""

import numpy as np
from typing import Tuple


def eig(X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]: 
    """
    Compute the eigenvalues and egiengectors for a given matrix square X. 

    Returns: 
        tuple(np.ndarray, np.ndarray): An ordered pair (eigenvalues, eigenvectors) sorted in ascending order by eigenvalues.
        The eigenvectors are returned as columns in the corresponding order.  
    """
    eigvals, eigvecs = np.linalg.eig(X)
    idx = np.argsort(eigvals)[::-1]
    return (eigvals[idx], eigvecs[:, idx])

def projection_matrix(B: np.ndarray) -> np.ndarray: 
    """
    Compute the projection matrix onto the space spanned by the columns of B.
    Returns: 
        np.ndarray: The projection matrix P with shape (n, n). 
    """
    return (B @ np.linalg.inv(B.T @ B)) @ B.T

def mse(predict, actual):
    """Function for computing the mean squared error (MSE)"""
    return np.square(predict - actual).sum(axis=1).mean()    

class PCA: 
    def __init__(self, n_components: int):
        self.n_components = n_components
        self.components_ = None  # Set of n principal components
        self.mean_ = None
    
    def fit(self): 
        raise NotImplementedError
    
    def transform(self, X: np.ndarray): 
        return X @ self.components_

    def inverse_transform(self, X_proj: np.ndarray): 
        return X_proj @ self.components_.T

    def fit_transform(self, X: np.ndarray):
        self.fit(X) 
        return self.transform(X)

    def reconstruct(self, X: np.ndarray): 
        X_proj = self.transform(X)
        return self.inverse_transform(X_proj)

    def reconstruction_error(self, X: np.ndarray) -> float:
        X_reconst = self.reconstruct(X)
        return mse(X, X_reconst)

class PCASimple(PCA): 
    def fit(self, X: np.ndarray):
        # Compute the covariance matrix S
        cov_matrix = 1.0 / len(X) * (X.T @ X)

        # Find the eigenvalues and correspondig eigenvectors for S
        eig_vals, eig_vecs = eig(cov_matrix)

        # Take the n principal components
        eig_vals, eig_vecs = eig_vals[:self.n_components], eig_vecs[:, ::self.n_components]

        self.components_ = np.real(eig_vecs)  # Principal component matrix

class PCAHighDim(PCA): 
    def fit(self, X: np.ndarray):
        N = X.shape[0]
        M = X @ X.T / N
        eigvals, eigvecs = eig(M)
        U = X.T @ eigvecs
        self.components_ = U 
    
def get_pca_model(mode: str, n_components: int) -> PCA:
    if mode == 'high_dim':
        return PCAHighDim(n_components)
    else:
        return PCASimple(n_components)

def main(): 
    pass

if __name__ == '__main__': 
    main()