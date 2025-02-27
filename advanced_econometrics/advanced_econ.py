import numpy as np
import scipy.stats as scs

def get_Xt(X0, alpha, mu, sigma, T=100):
    Xt_mu = np.zeros(T)[:,None]
    Xt_mu[0] = X0
    for i in range(1,T):
        Xt_mu[i] = mu + alpha * (Xt_mu[i-1] - mu) + scs.norm(0, sigma).rvs()
    return Xt_mu

def get_Xt_7(X0, alpha, mu_epsilon, sigma_epsilon, T=100):
    Xt = np.zeros(T)[:,None]
    Xt[0] = X0
    for i in range(1,T):
        Xt[i] = alpha * Xt[i-1] + scs.norm(0, sigma).rvs()
    return Xt

def get_monte_carlo_5(dict_arguments):
    def wrapper(X0, alpha, mu, sigma, T):
        return get_Xt(X0, alpha, mu=mu, sigma=sigma, T=T).ravel(), mu
    return wrapper(**dict_arguments)