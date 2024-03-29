import numpy as np

def normal_erf(x, mu = 0, sigma = 1,  depth = 50):
    ele = 1.0
    normal = 1.0
    x = (x - mu)/sigma
    erf = x
    for i in range(1,depth):
        ele = - ele * x * x/2.0/i
        normal = normal + ele
        erf = erf + ele * x / (2.0 * i + 1)

    return np.clip(normal/np.sqrt(2.0*np.pi)/sigma,0,None) , np.clip(erf/np.sqrt(2.0*np.pi)/sigma,-0.5,0.5)

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
            
    if mu is None:
        #median is an approach as robust and naïve as possible to Expectation
        mu = np.median(X)
    mu_0 = mu + 1
    
    if sigma is None:
        #rule of thumb
        sigma = np.std(X)/3
    sigma_0 = sigma + 1
    
    bandwidth_truncated_normal_weight, bandwidth_truncated_normal_sigma = truncated_intergral_and_sigma(bandwidth)

    
    while abs(mu - mu_0) + abs(sigma - sigma_0) > eps:
        #loop until tolerence is reached

        """
        create a uniform window on X around mu of width 2*bandwidth*sigma
        find the mean of that window to shift the window to most expected local value
        measure the standard deviation of the window and divide by the standard deviation of a truncated gaussian distribution
        measure the proportion of points inside the window, divide by the weight of a truncated gaussian distribution
        """
        Window = np.logical_and(X - mu - bandwidth * sigma < 0 , X - mu + bandwidth * sigma > 0)
        if weights is None : 
            Window_weights = None
        else :
            Window_weights = weights[Window]
        mu_0, mu = mu, np.average(X[Window], weights = Window_weights)
        var = np.average(np.square(X[Window]), weights = Window_weights) - mu**2
        sigma_0 , sigma = sigma, np.sqrt(var)/bandwidth_truncated_normal_sigma

    w = np.average(Window, weights = weights)/bandwidth_truncated_normal_weight

    return w,mu,sigma
