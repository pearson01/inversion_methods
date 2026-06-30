import numpy as np
from scipy.linalg import cholesky, solve_triangular


def woodbury(inv_A, U, C, V):
# def woodbury(A, U, C, V):

    """
    The Sherman-Morrison-Woodbury matrix identity.

    (A + UCV)^-1 = A^-1 - A^-1 U (C^-1 + V A^-1 U)^-1 V A^-1

    A is nxn
    C is kxk
    U is nxk
    V is kxn

    In our case, 
    inv_A = R^-1,
    U = H_hat,
    C = Pf,
    V = H_hat.T

    (R + H_hat @ Pf @ H_hat.T)^-1 = R^-1 - R^-1 @ H_hat (Pf^-1 + H_hat.T @ R^-1 @ H_hat)^-1 @ H_hat.T @ R ^-1
    """
    # inv_A = np.linalg.inv(A)
    inv_C = np.linalg.inv(C)
    step1 = np.linalg.inv(inv_C + V @ inv_A @ U)
    # step1 = np.linalg.pinv(inv_C + V @ inv_A @ U)
    # step1 = inv_C + V @ inv_A @ U

    # return inv_A - inv_A @ U @ np.linalg.solve(step1, V) @ inv_A
    return inv_A - inv_A @ U @ step1 @ V @ inv_A


def Ainv_B(A, B):

    """
    Takes 2 matrices A and B. Finds the solution X such that X = A^-1 B.

    Utilises chain of efficiency to optimise compute time.
    """

    
    try:
        LA = cholesky(A, lower=True)
        Y = solve_triangular(LA, B, lower=True)
        X = solve_triangular(LA.T, Y, lower=False)

    except np.linalg.LinAlgError:
        A = 0.5 * (A + A.T)
        eigvals, eigvecs = np.linalg.eigh(A)
        eigvals = np.maximum(eigvals, 1e-12)
        A = eigvecs @ np.diag(eigvals) @ eigvecs.T

        try:
            LA = cholesky(A, lower=True)
            Y = solve_triangular(LA, B, lower=True)
            X = solve_triangular(LA.T, Y, lower=False)

        except np.linalg.LinAlgError:
            X = np.linalg.solve(A + 1e-10 * np.eye(A.shape[0]), B)


    return X


def kalman_gain_woodbury(Pf_aug, H_aug, R_inv):

    """
    Calculates the Kalman gain K utilising an expansion of the gain calculation with the Woodbury identity in combination with Cholesky matrix decomposition.

    Avoids direct matrix inversions.
    
    """
    
    nz = Pf_aug.shape[0]

    try:

        L = cholesky(Pf_aug, lower=True)

    except np.linalg.LinAlgError:

        Pf_aug = 0.5 * (Pf_aug + Pf_aug.T)
        eigvals, eigvecs = np.linalg.eigh(Pf_aug)
        eigvals = np.maximum(eigvals, 1e-12)
        Pf_aug = eigvecs @ np.diag(eigvals) @ eigvecs.T

        L = cholesky(Pf_aug, lower=True)

    W = H_aug @ L

    RinvW = R_inv[:, None] * W

    S = np.eye(nz) + W.T @ RinvW

    A = H_aug.T * R_inv

    term1 = Pf_aug @ A 

    # (I + W^T R^{-1} W)^{-1}

    B = L.T @ A

    X = Ainv_B(S, B)

    term2 = term1 @ W @ X

    K = term1 - term2

    return K
