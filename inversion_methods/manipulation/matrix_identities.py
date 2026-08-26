import numpy as np
from scipy.linalg import cholesky, solve_triangular, cho_factor, cho_solve


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


# def kalman_gain_woodbury(Pf_aug, H_aug, R_inv):

#     """
#     Calculates the Kalman gain K utilising an expansion of the gain calculation with the Woodbury identity in combination with Cholesky matrix decomposition.

#     Avoids direct matrix inversions.
    
#     """
    
#     nz = Pf_aug.shape[0]

#     try:

#         L = cholesky(Pf_aug, lower=True)

#     except np.linalg.LinAlgError:

#         print(f"Trouble in paradise: Pf_aug is not positive definite. Is finite? {np.all(np.isfinite(Pf_aug))}, Cov condition: {np.linalg.cond(Pf_aug)}")
        
#         Pf_aug = _symmetrize_square(Pf_aug)
#         eigvals, eigvecs = np.linalg.eigh(Pf_aug)
#         eigvals = np.maximum(eigvals, 1e-12)
#         Pf_aug = eigvecs @ np.diag(eigvals) @ eigvecs.T

#         try:
#             L = cholesky(Pf_aug, lower=True)

#         except np.linalg.LinAlgError:
#             print(f"Trying iterative jitter...")
#             for jitter in (1e-12, 1e-10, 1e-8, 1e-6):    
#                 try:        
#                     L = cholesky(Pf_aug + jitter*np.eye(Pf_aug.shape[0]), lower=True)
#                     print(f"Success with jitter = {jitter}!!")        
#                     break    
#                 except np.linalg.LinAlgError:        
#                     pass
#             else:    
#                 raise np.linalg.LinAlgError("Unable to obtain positive-definite covariance in Kalman Gain calculation.")

        

#     W = H_aug @ L

#     RinvW = R_inv[:, None] * W

#     S = np.eye(nz) + W.T @ RinvW

#     A = H_aug.T * R_inv

#     term1 = Pf_aug @ A 

#     # (I + W^T R^{-1} W)^{-1}

#     B = L.T @ A

#     X = Ainv_B(S, B)

#     term2 = term1 @ W @ X

#     K = term1 - term2

#     return K


def kalman_gain_woodbury(
    Pf_aug,
    H_aug,
    R_inv,
):
    """
    Calculate the Kalman gain using the Woodbury identity.

    This implementation assumes that the observation-error covariance
    R is diagonal and that R_inv is supplied as a one-dimensional
    vector containing the diagonal of R^{-1}.

    Parameters
    ----------
    Pf_aug : ndarray, shape (nz, nz)
        Augmented forecast covariance matrix.

    H_aug : ndarray, shape (ny, nz)
        Observation operator.

    R_inv : ndarray, shape (ny,)
        Diagonal entries of the inverse observation-error covariance.

    validate_inputs : bool, default=True
        Check dimensions, finite values, and positivity of R_inv.

    Returns
    -------
    K : ndarray, shape (nz, ny)
        Kalman gain.

    Notes
    -----
    With

        Pf_aug = L @ L.T
        W = H_aug @ L

    the gain is calculated as

        K = L @ solve(
            I + W.T @ R_inv @ W,
            L.T @ H_aug.T @ R_inv
        ).

    No explicit matrix inverse is formed.
    """
    H_aug = np.asarray(H_aug)
    R_inv = np.asarray(R_inv)

    nz = Pf_aug.shape[0]

    L, _, _ = _cholesky_with_repair(Pf_aug)

    # W has shape (ny, nz).
    W = H_aug @ L

    # Apply diagonal R^{-1} by row scaling:
    #
    # R^{-1} W
    #
    # This avoids forming an ny-by-ny diagonal matrix.
    RinvW = R_inv[:, None] * W

    # Woodbury system:
    #
    # S = I + L.T H.T R^{-1} H L
    #   = I + W.T R^{-1} W
    #
    # S has shape (nz, nz).
    S = W.T @ RinvW
    S.flat[:: nz + 1] += 1.0

    # Floating-point matrix multiplication can introduce very small
    # asymmetries.
    S = _symmetrize_square(S)

    # Right-hand side:
    #
    # B = L.T H.T R^{-1}
    #   = W.T R^{-1}
    #
    # Since W = H L, using W.T directly avoids an additional
    # multiplication by L.T.
    B = RinvW.T

    # In exact arithmetic S is positive definite because:
    #
    # S = I + W.T R^{-1} W
    #
    # with R_inv > 0. A regular Cholesky factorization should
    # therefore normally succeed.
    try:
        S_factor = cho_factor(S, lower=True, overwrite_a=False, check_finite=False)

    except np.linalg.LinAlgError:
        # This should be rare and usually indicates severe numerical
        # scaling or invalid inputs.
        S_L, _, _ = _cholesky_with_repair(S)

        # Construct the tuple expected by cho_solve. Because S_L is
        # explicitly lower triangular, lower=True is appropriate.
        S_factor = (S_L, True)

    # X = S^{-1} B, without explicitly calculating S^{-1}.
    X = cho_solve(S_factor, B, overwrite_b=False, check_finite=False)

    # Simplified Woodbury expression.
    K = L @ X

    return K


def kalman_gain_woodbury_from_cholesky(
    L,
    H_aug,
    R_inv,
):
    """
    Compute the Kalman gain using the Woodbury identity and a
    Cholesky factor of the forecast covariance.

    This implementation is intended for problems in which the number
    of observations is substantially greater than the state dimension,
    ``ny >> nz``. It avoids constructing or factorizing the
    ``ny x ny`` innovation covariance matrix by performing the update
    in state space.

    The gain is calculated using

        K = A^{-1} H_hat.T R^{-1},

    where

        A = Pf^{-1} + H_hat.T R^{-1} H_hat,

    ``Pf`` is the forecast-error covariance, ``R`` is the
    observation-error covariance, and ``Pf_chol`` is a Cholesky factor
    of ``Pf``.

    The observation-error covariance is assumed to be diagonal, with
    ``r_inv`` containing the diagonal elements of ``R^{-1}``. The
    function does not explicitly construct ``Pf^{-1}``, ``R``, or the
    observation-space innovation covariance.

    Parameters
    ----------
    Pf_chol : ndarray, shape (nz, nz)
        Lower- or upper-triangular Cholesky factor of the forecast-error
        covariance matrix ``Pf``. The interpretation of the triangular
        factor must match that used internally by the implementation.

    H_hat : ndarray, shape (ny, nz)
        Linearized observation operator mapping the state vector into
        observation space.

    r_inv : ndarray, shape (ny,)
        Diagonal elements of the inverse observation-error covariance
        matrix, ``R^{-1}``. All elements should be finite and strictly
        positive.

    Returns
    -------
    K : ndarray, shape (nz, ny)
        Kalman gain matrix.

    S_factor : tuple of (ndarray, bool)
        Cholesky factorization of the state-space posterior precision matrix

    """

    H_aug = np.asarray(H_aug)
    R_inv = np.asarray(R_inv)

    nz = L.shape[0]

    # W has shape (ny, nz).
    W = H_aug @ L

    # Apply diagonal R^{-1} by row scaling:
    #
    # R^{-1} W
    #
    # This avoids forming an ny-by-ny diagonal matrix.
    RinvW = R_inv[:, None] * W

    # Woodbury system:
    #
    # S = I + L.T H.T R^{-1} H L
    #   = I + W.T R^{-1} W
    #
    # S has shape (nz, nz).
    S = W.T @ RinvW
    S.flat[:: nz + 1] += 1.0

    # Floating-point matrix multiplication can introduce very small
    # asymmetries.
    S = _symmetrize_square(S)

    # Right-hand side:
    #
    # B = L.T H.T R^{-1}
    #   = W.T R^{-1}
    #
    # Since W = H L, using W.T directly avoids an additional
    # multiplication by L.T.
    B = RinvW.T

    # In exact arithmetic S is positive definite because:
    #
    # S = I + W.T R^{-1} W
    #
    # with R_inv > 0. A regular Cholesky factorization should
    # therefore normally succeed.
    try:
        S_factor = cho_factor(S, lower=True, overwrite_a=False, check_finite=False)

    except np.linalg.LinAlgError:
        # This should be rare and usually indicates severe numerical
        # scaling or invalid inputs.
        S_L, _, _ = _cholesky_with_repair(S)

        # Construct the tuple expected by cho_solve. Because S_L is
        # explicitly lower triangular, lower=True is appropriate.
        S_factor = (S_L, True)

    # X = S^{-1} B, without explicitly calculating S^{-1}.
    X = cho_solve(S_factor, B, overwrite_b=False, check_finite=False)

    # Simplified Woodbury expression.
    K = L @ X

    return K, S_factor


def covariance_from_woodbury_factor(L, S_factor):
    Sinv_LT = cho_solve(
        S_factor,
        L.T,
        overwrite_b=False,
        check_finite=False,
    )

    Pa = L @ Sinv_LT
    Pa = 0.5 * (Pa + Pa.T)

    return Pa


def _symmetrize_square(matrix):
    """Remove small numerical asymmetry from a square matrix."""
    return 0.5 * (matrix + matrix.T)



def _matrix_scale(matrix):
    """
    Return a scale used to determine an appropriate diagonal jitter.

    The minimum value of 1.0 prevents extremely small absolute jitter
    when all covariance entries are close to zero.
    """
    return max(
        1.0,
        float(np.max(np.abs(np.diag(matrix)))),
    )


def _cholesky_with_repair(
    matrix,
    relative_jitters=(0.0, 1e-12, 1e-10, 1e-8, 1e-6),
):
    """
    Return a lower triangular Cholesky factorised matrix.

    The matrix is first symmetrized. If its Cholesky factorization
    fails, progressively larger scale-dependent diagonal jitter is
    added.

    Returns
    -------
    factor : matrix
        Output from scipy.linalg.cholesky.
    repaired_matrix : ndarray
        The matrix that was actually factorized.
    jitter : float
        Absolute diagonal jitter applied.
    """
    matrix = np.asarray(matrix, dtype=float)
    matrix = _symmetrize_square(matrix)

    if not np.all(np.isfinite(matrix)):
        raise ValueError(
            "Cannot factorize a matrix containing NaN or infinite values."
        )

    nstate = matrix.shape[0]

    if matrix.shape != (nstate, nstate):
        raise ValueError("Input matrix must be square.")

    scale = _matrix_scale(matrix)

    for relative_jitter in relative_jitters:
        absolute_jitter = relative_jitter * scale

        if relative_jitter == 0.0:
            candidate = matrix
        else:
            candidate = matrix.copy()
            candidate.flat[:: nstate + 1] += absolute_jitter

        try:
            factor = cholesky(candidate, lower=True, overwrite_a=False, check_finite=False)

            return factor, candidate, absolute_jitter

        except np.linalg.LinAlgError:
            continue

    # Final, more expensive eigenvalue repair.
    eigenvalues, eigenvectors = np.linalg.eigh(matrix)

    eigenvalue_floor = max(1e-12 * scale, np.finfo(matrix.dtype).eps * scale)

    eigenvalues = np.maximum(eigenvalues, eigenvalue_floor)

    # This is equivalent to:
    #
    # eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T
    #
    # but avoids constructing the diagonal matrix explicitly.
    repaired_matrix = (eigenvectors * eigenvalues[None, :]) @ eigenvectors.T

    repaired_matrix = _symmetrize_square(repaired_matrix)

    factor = cholesky(repaired_matrix, lower=True, overwrite_a=False, check_finite=False)

    return factor, repaired_matrix, eigenvalue_floor


def _sample_gaussian_cholesky(
    mean,
    covariance,
    rng,
    relative_jitters=(0.0, 1e-12, 1e-10, 1e-8, 1e-6),
):
    """
    Draw a sample from N(mean, covariance) using a Cholesky factor.

    If covariance is not numerically positive definite, diagonal
    jitter and then eigenvalue clipping are attempted.

    Parameters
    ----------
    mean : ndarray, shape (nstate,)
        Gaussian mean.
    covariance : ndarray, shape (nstate, nstate)
        Gaussian covariance.
    rng : numpy.random.Generator
        Random-number generator.
    relative_jitters : tuple of float
        Relative diagonal jitters to attempt.

    Returns
    -------
    sample : ndarray, shape (nstate,)
        Gaussian random sample.
    """
    mean = np.asarray(mean)
    covariance = np.asarray(covariance)
    covariance = _symmetrize_square(covariance)

    if not np.all(np.isfinite(mean)):
        raise ValueError(
            "Cannot sample from a Gaussian with a non-finite mean."
        )

    if not np.all(np.isfinite(covariance)):
        raise ValueError(
            "Cannot sample from a Gaussian with a non-finite covariance."
        )

    nstate = mean.size

    if covariance.shape != (nstate, nstate):
        raise ValueError(
            "Mean and covariance dimensions are inconsistent. "
            f"Mean length is {nstate}, while covariance has shape "
            f"{covariance.shape}."
        )

    factor, _, _ = _cholesky_with_repair(covariance, relative_jitters=relative_jitters)

    standard_normal = rng.standard_normal(nstate)

    return mean + factor @ standard_normal

