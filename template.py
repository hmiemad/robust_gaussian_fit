from matplotlib import pyplot as plt
import numpy as np

from robust_gaussian_fit import robust_gaussian_fit, normal_erf

normal_1 = np.random.normal(0,1,10000)
normal_2 = np.random.normal(5,2,20000)
normal_3 = np.random.normal(-10,2,20000)
trimodal = np.concatenate([normal_1,normal_2,normal_3])
print(f"Trimodal data, mean = {trimodal.mean():5.1f}, std = {trimodal.std():5.1f}")
fig, ax = plt.subplots(figsize = (15,10))

_, bins, _ = ax.hist(trimodal, 100, density=True)

for mu_0 in np.linspace(-12,8,21):
    w, mu, sigma = robust_gaussian_fit(trimodal, mu = mu_0, bandwidth=1)
    y = w*normal_erf(bins,mu,sigma)[0]
    ax.plot(bins, y, '--')

    print(f"{mu_0:5.1f} : {w:5.2f} x Normal({mu:5.1f} ,{sigma:4.1f})")

fig.show()
