import numpy as np
import scipy.stats as scs
from numba import jit

def get_Xt(X0, alpha, mu, sigma, T=100):
    Xt_mu = np.zeros(T)[:,None]
    Xt_mu[0] = X0
    for i in range(1,T):
        Xt_mu[i] = mu + alpha * (Xt_mu[i-1] - mu) + scs.norm(0, sigma).rvs()
    return Xt_mu

def get_Xt_7(X0, alpha, epsilon_dist, lst_eps_param, T=500):
    Xt = np.zeros(T)[:,None]
    Xt[0] = X0
    for i in range(1,T):
        Xt[i] = alpha * Xt[i-1] + epsilon_dist(*lst_eps_param).rvs()
    return Xt

@jit
def ols(
    y: np.array,
    X: np.array,
) -> (np.array, float):
    n, k = X.shape
    beta = np.linalg.inv(X.T @ X) @ X.T @ y
    r_hat = y - X @ beta
    sigma = np.sqrt((r_hat.T @ r_hat) / (n - k))
    return beta, sigma


def get_monte_carlo_5(dict_arguments):
    def wrapper(X0, alpha, mu, sigma, T):
        return get_Xt(X0, alpha, mu=mu, sigma=sigma, T=T).ravel(), mu
    return wrapper(**dict_arguments)


def get_monte_carlo_7(dict_arguments):
    def wrapper(X0, alpha, epsilon_dist, lst_eps_param):
        dct = {}
        Xt = get_Xt_7(X0, alpha, epsilon_dist, lst_eps_param, T=1_000).ravel()
        # return Xt
        for T in [10, 100, 1_000]:
            _y, _X = Xt[1:T][:,None], np.concat([np.ones(T-1)[:,None], Xt[:T-1][:,None]], axis=1)
            dct[f'ols{T}'] = ols(_y, _X)

        dct['Xt'] = Xt
        return dct            
    return wrapper(**dict_arguments)