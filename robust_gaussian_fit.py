import numpy as np
from scipy.stats import norm
from scipy.special import erf

#Integral of a normal distribution from -x to x
truncated_integral = lambda x:erf(x/np.sqrt(2))
#Standard deviation of a truncated normal distribution from -x to x
truncated_sigma = lambda x:np.sqrt(1-2*x*norm.pdf(x)/truncated_integral(x))

def robust_gaussian_fit(X, mu = None, sigma = None, bandwidth = 1.0, eps = 1.0e-5, weights = None):

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
    if weights is None:
        weights = np.ones(X.shape)
    else :
        if weights.shape != X.shape :
            raise "weights and values must have the same shape"
    if mu is None:
        #median is an approach as robust and naïve as possible to Expectation
        mu = np.median(X)
    
    if sigma is None:
        #rule of thumb
        sigma = np.std(X)/3
        
    bandwidth_truncated_normal_weight = truncated_integral(bandwidth)
    bandwidth_truncated_normal_sigma = truncated_sigma(bandwidth)
    
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
            mu = np.average(X[W == 1], weights = weights[W == 1])
            var = np.average(np.square(X[W == 1]), weights = weights[W == 1]) - mu**2
            sigma = np.sqrt(var)/bandwidth_truncated_normal_sigma
            w0 = w
            w = np.average(W, weights = weights)/bandwidth_truncated_normal_weight
        
        except:
            break
    
    return w,mu,sigma
