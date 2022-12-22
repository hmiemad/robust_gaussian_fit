import numpy as np

def normal_erf(x, depth = 50):
    ele = 1.0
    normal = 1.0
    erf = x
    for i in range(1,depth):
        ele = - ele * x * x/2.0/i
        normal = normal + ele
        erf = erf + ele * x / (2.0 * i + 1)

    return normal/np.sqrt(2.0*np.pi) , erf/np.sqrt(2.0*np.pi)

def truncated_intergral_and_sigma(x):
    n,e = normal_erf(x)
    return 2*e, np.sqrt(1-n*x/e)


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
        
    bandwidth_truncated_normal_weight, bandwidth_truncated_normal_sigma = truncated_intergral_and_sigma(bandwidth)

    
    while abs(w - w0) > eps:
        #loop until tolerence is reached

        """
        create a uniform window on X around mu of width 2*bandwidth*sigma
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

    return w,mu,sigma

