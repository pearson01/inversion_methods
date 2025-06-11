import numpy as np
from scipy.optimize import fsolve



def lognormal_mean_stdev(mean_lognormal: float, stdev_lognormal: float) -> tuple[float, float]:
    """Return the pymc `mu` and `sigma` parameters that give a log normal distribution
    with the given mean and stdev.

    Args:
        mean: desired mean of log normal
        stdev: desired standard deviation of log normal

    Returns:
        tuple (mu, sigma), where `pymc.LogNormal(mu, sigma)` has the given mean and stdev.

    Formulas for log normal mean and variance:

    mean = exp(mu + 0.5 * sigma ** 2)
    stdev ** 2 = var = exp(2*mu + sigma ** 2) * (exp(sigma ** 2) - 1)

    This gives linear equations for `mu` and `sigma ** 2`:

    mu + 0.5 * sigma ** 2 = log(mean)
    sigma ** 2 = log(1 + (stdev / mean)**2)

    So

    mu = log(mean) - 0.5 * log(1 + (stdev/mean)**2)
    sigma = sqrt(log(1 + (stdev / mean)**2))
    """
    var = np.log(1 + (stdev_lognormal / mean_lognormal) ** 2)
    mu = np.log(mean_lognormal) - 0.5 * var
    sigma = np.sqrt(var)
    return mu, sigma


def lognormal_mode_stdev(mode_lognormal: float,
                        stdev_lognormal: float,
                        initial_var: float = 1.0,
                        ) -> tuple[float, float]:

    """Return the pymc `mu` and `sigma` parameters that give a log normal distribution
    with the given mode and stdev.

    Args:
        mode: desired mode of log normal
        stdev: desired standard deviation of log normal

    Returns:
        tuple (mu, sigma), where `pymc.LogNormal(mu, sigma)` has the given mean and stdev.

    Formulas for mu and var:

    mean = ln(mode_y) + sigma ** 2
    stdev ** 2 = exp(2*mu + sigma ** 2) * (exp(sigma ** 2) - 1)

    As sigma ** 2 is required for the mean, we must solve these as a system of non-linear equations.
    
    """

    def equations(vars): 
        """
        For brevity, the lognormal distribution represented by y and the underlying normal represented by x
        """
        var_y = stdev_lognormal**2
        mode_y = mode_lognormal
        var_x = vars[0]
        mu_x = np.log(mode_y) + var_x

        return np.exp(2*mu_x + 2*var_x) - np.exp(2*mu_x + var_x) - var_y
    
    initial_guess = [initial_var]
    
    variance_normal = fsolve(equations, initial_guess)[0]
    sigma = np.sqrt(variance_normal)
    mu = variance_normal + np.log(mode_lognormal)

    return mu, sigma


def lognormal_median_stdev(median_lognormal: float,
                        stdev_lognormal: float,
                        ) -> tuple[float, float]:

    """Return the pymc `mu` and `sigma` parameters that give a log normal distribution
    with the given median and stdev.

    Args:
        median: desired mode of log normal
        stdev: desired standard deviation of log normal

    Returns:
        tuple (mu, sigma), where `pymc.LogNormal(mu, sigma)` has the given mean and stdev.

    Formulas for mu and var:

    mu = ln(median_y)
    stdev ** 2 = ln((1 + sqrt(1 + 4 * stdev_y^2 / median_y)) / 2)
    
    """

    mu = np.log(median_lognormal)

    sigma = 0.5 * np.log((1 + (1 + 4 * stdev_lognormal**2 / median_lognormal)**0.5) / 2)

    return mu, sigma


def covariance_lognormal_transform(covariance_lognormal: np.ndarray,
                                   mean_normal: float,
                                   stdev_normal: float,
                                   ) -> tuple[np.ndarray, np.ndarray]:

    """
    Takes the covariance matrix of a lognormal distribution, the mean and stdev of the underlying normal distribution 
    and outputs the covariance and precision matrices of the underlying normal distribution.

    Equations:

    mean_lognormal = exp( mean_normal + 0.5 * (stdev_normal ** 2))
    covariance_normal[i,j] = covariance_normal[j,i] = ln( 1 + covariance_lognormal[i,j] / (mean_lognormal ** 2))

    """
    dim = len(covariance_lognormal)

    mean_lognormal = np.exp(mean_normal + 0.5*(stdev_normal**2))
    covariance_normal = np.diag(np.ones(dim)*stdev_normal**2)

    for i in range(dim):
        for j in range(i+1, dim):
            covariance_normal[i, j] = covariance_normal[j, i] = np.log(1 + covariance_lognormal[i,j] / (mean_lognormal**2))

    precision_normal = np.linalg.inv(covariance_normal)
    
    return covariance_normal, precision_normal


def update_log_normal_prior(prior):
    if prior["pdf"].lower() == "lognormal":
        if "stdev" in prior:
            stdev = float(prior["stdev"])
            if "mean" in prior:
                mean = float(prior["mean"])
                mu, sigma = lognormal_mean_stdev(mean, stdev)
                del prior["mean"]
            elif "mode" in prior:
                mode = float(prior["mode"])
                mu, sigma = lognormal_mode_stdev(mode, stdev)
                del prior["mode"]
            else:
                raise ValueError("prior['stdev'] must be coupled with prior['mean'] or prior['mode']")
            del prior["stdev"]
            prior["mu"] = mu
            prior["sigma"] = sigma
        elif "mu" and "sigma" in prior: 
            pass
        else:
            raise ValueError("Incompatible combination of prior parameters.") 