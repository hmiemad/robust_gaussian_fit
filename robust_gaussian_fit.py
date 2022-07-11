import numpy as np
from scipy.stats import norm
from scipy.special import erf

#Integral of a normal distribution from -x to x
weights = lambda x:erf(x/np.sqrt(2))
#Standard deviation of a truncated normal distribution from -x to x
sigmas = lambda x:np.sqrt(1-2*x*norm.pdf(x)/weights(x))

def robust_gaussian_fit(X, mu = None, sigma = None, bandwidth = 1.0, eps = 1.0e-5):
    """
    Fits a single principal gaussian component around a starting guess point
    in a 1-dimensional gaussian mixture of unknown components with EM algorithm

    Args:
        X (np.array): A sample of 1-dimensional mixture of gaussian random variables
        mu (float, optional): Expectation. Defaults to None.
        sigma (float, optional): Standard deviation. Defaults to None.
        bandwidth (float, optional): Hyperparameter of truncation. Defaults to 1.
        eps (float, optional): Convergence tolerance. Defaults to 1.0e-5.

    Returns:
        w,mu,sigma: weight, mean and stdev of the gaussian component
    """

    w,w0=0,2
    
    if mu is None:
        #median is an approach as robust and naïve as possible to Expectation
        mu = np.median(X)
    
    if sigma is None:
        #rule of thumb
        sigma = np.std(X)/3
        
    bandwidth_truncated_normal_weight = weights(bandwidth)
    bandwidth_truncated_normal_sigma = sigmas(bandwidth)
    
    while abs(w - w0) > eps:
        #loop until tolerence is reached
        try:
            """
            create a window on X around mu of width 2*bandwidth*sigma
            find the mean of that window to shift the window to most expected local value
            measure the standard deviation of the window and divide by the standard deviation of a truncated gaussian distribution
            measure the proportion of points inside the window, divide by the weight of a truncated gaussian distribution
            """
            W = np.where(np.logical_and(X - mu - bandwidth * sigma < 0 , X - mu + bandwidth * sigma > 0), 1, 0)
            mu = np.mean(X[W == 1])
            sigma = np.std(X[W == 1])/bandwidth_truncated_normal_sigma
            w0 = w
            w = np.mean(W)/bandwidth_truncated_normal_weight
        
        except:
            break
    
    return w,mu,sigma