import numpy as np


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
