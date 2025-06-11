import numpy as np
from inversion_methods.manipulation import woodbury
from pandas import date_range, to_datetime
from dataclasses import dataclass


@dataclass
class InversionInput:
    start_date: str
    end_date: str
    Hx: np.ndarray
    Y: np.ndarray
    Ytime: np.ndarray
    error: np.ndarray
    siteindicator: np.ndarray
    use_bc: bool = False
    Hbc: np.ndarray | None = None
    nbasis: int
    xprior: dict
    bcprior: dict
    x_covariance: np.ndarray
    bc_covariance: np.ndarray | None = None


@dataclass
class InversionIntermediate:
    Y_dic: dict
    Yerr_dic: dict
    H_dic: dict
    Hbc_dic: dict | None = None
    nbasis: dict
    nperiod: dict
    use_bc: bool = False
    xprior: dict
    bcprior: dict
    x_covariance: np.ndarray
    bc_covariance: np.ndarray | None = None


def mxkf_monthly_dictionaries(config: InversionInput):

    """
    Manipulates inversion data into dictionaries separated by month.
    """

    Y_dic = {}
    Yerr_dic = {}
    Ytime_dic = {}
    H_dic = {}
    siteindicator_dic = {}
    allmonth = date_range(config.start_date, config.end_date, freq="MS")[:-1]
    Ymonth = to_datetime(config.Ytime).to_period("M")
    nperiod = len(allmonth)

    if config.use_bc is True:
        Hbc_dic = {}

    else:
        Hbc_dic = None

    bc_count = 0
    for period in range(nperiod):
        mnth = allmonth[period].month
        yr = allmonth[period].year
        mnthloc = np.where(np.logical_and(Ymonth.month == mnth, Ymonth.year == yr))[0]
        Y_dic[period] = config.Y[mnthloc]
        Yerr_dic[period] = config.error[mnthloc]
        Ytime_dic[period] = config.Ytime[mnthloc]
        H_dic[period] = config.Hx.T[mnthloc, :]
        siteindicator_dic[period] = config.siteindicator[mnthloc]
        if config.use_bc:
            Hbc_dic[period] = config.Hbc.T[mnthloc, bc_count:bc_count+4]
            bc_count += 4

    return InversionIntermediate(Y_dic, 
                                 Yerr_dic, 
                                 H_dic, 
                                 Hbc_dic, 
                                 config.nbasis, 
                                 nperiod, 
                                 config.use_bc,
                                 config.xprior,
                                 config.bcprior,
                                 config.x_covariance,
                                 config.bc_covariance,
                                 )




def MX_Kalman_Filter(config: InversionIntermediate):

    if config.use_bc is True:
        nparam = config.nbasis + 4
        P_prior = np.zeros((nparam, nparam))
        P_prior[:config.nbasis, :config.basis] = config.x_covariance
        P_prior[config.nbasis:, config.nbasis:] = config.bc_covariance

    else:
        nparam = config.nbasis
        P_prior = config.x_covariance

    Ymod_dic = {}

    xprior_mu = config.xprior["mu"]
    xprior_ln_median = np.exp(xprior_mu)

    bcprior_mu = config.bcprior["mu"]
    bcprior_ln_median = np.exp(bcprior_mu)

    xb = np.ones(config.nbasis) * xprior_ln_median
    bcb = np.ones(4) * bcprior_ln_median

    periods = np.arange(config.nperiod)
    Q = np.eye(nparam) * 0

    xouts_median = np.zeros((nparam, config.nperiod))
    xouts_stdev = np.zeros((nparam, config.nperiod))
    xouts_mu = np.zeros((nparam, config.nperiod))
    xouts_sigma = np.zeros((nparam, config.nperiod))
    xouts_mean_kalman = np.zeros((nparam, config.nperiod))
    xouts_mode_kalman = np.zeros((nparam, config.nperiod))
    xouts_68 = np.zeros((nparam, 2, config.nperiod))
    xouts_95 = np.zeros((nparam, 2, config.nperiod))

    for t in periods:
        if config.use_bc:
            H = np.hstack((config.H_dic[t], config.Hbc_dic[t]))
            if t == 0:
                xb = np.append(xb, bcb)
        else:
            H = config.H_dic[t]

        Y = config.Y_dic[t]
        Yerr = config.Yerr_dic[t]
        nm = len(Yerr)
        R = np.diag(Yerr**2)
        R_inv = np.diag(1/(Yerr**2))

        # Wb = np.diag(1/xb)
        Wb = np.diag(xb)
        Wo_inv = np.eye(nm)
        H_hat = Wo_inv @ H @ Wb
        

        if t == 0:
            Pf = P_prior + Q

        K = Pf @ H_hat.T @ woodbury(R_inv, H_hat, Pf, H_hat.T)
        # K = Pf @ H_hat.T @ np.linalg.inv(H_hat @ Pf @ H_hat.T + R)
        xa_mu = np.log(xb) + K @ (Y - H @ xb)
        xa = np.exp(xa_mu)

        Pa = (np.eye(nparam) - K @ H_hat) @ Pf

        print(f"{t} - Condition of R_inv: {round(np.linalg.cond(R_inv),2)}, H_hat: {round(np.linalg.cond(H_hat),2)}, Pf: {round(np.linalg.cond(Pa),2)}, Pf_inv + HR_invH.T: {round(np.linalg.cond(np.linalg.inv(Pf) +H_hat.T@R_inv@H_hat),2)}")

        # Pf = Pa + Q
        # xb = xa
    
        Pf = P_prior + Q

        if config.use_bc:
            xb = np.append(np.ones(config.nbasis)*xprior_ln_median, np.ones(4)*bcprior_ln_median)
        else:
            xb = np.ones(config.nbasis)*xprior_ln_median
    
        xouts_median[:,t] = xa
        xouts_sigma[:,t] = np.diag(Pa)**0.5
        xouts_mu[:,t] = xa_mu
        xouts_stdev[:,t] = ((np.exp(xouts_sigma[:,t]**2) - 1) * np.exp(2*xouts_mu[:,t] + xouts_sigma[:,t]**2))**0.5
        xouts_mean_kalman[:,t] = np.exp(xouts_mu[:,t] + 0.5*xouts_sigma[:,t]**2)
        # for basis in np.arange(nparam):
            # xouts_68[basis, 0, t] = lognorm.ppf(0.16, xouts_sigma[basis, t], scale=np.exp(xouts_mu[basis, t]))
            # xouts_68[basis, 1, t] = lognorm.ppf(0.34, xouts_sigma[basis, t], scale=np.exp(xouts_mu[basis, t]))
        
        
        xouts_68[:, 0, t] = np.exp(xouts_mu[:,t] - xouts_sigma[:,t])
        xouts_68[:, 1, t] = np.exp(xouts_mu[:,t] + xouts_sigma[:,t])

        xouts_95[:, 0, t] = np.exp(xouts_mu[:,t] - 2*xouts_sigma[:,t])
        xouts_95[:, 1, t] = np.exp(xouts_mu[:,t] + 2*xouts_sigma[:,t])
        

        xouts_mode_kalman[:,t] = np.exp(xouts_mu[:,t] - xouts_sigma[:,t]**2)

        Ymod_dic[t] = H @ xouts_mean_kalman[:,t]

    return xouts_mean_kalman, xouts_68, xouts_95, Ymod_dic, nparam