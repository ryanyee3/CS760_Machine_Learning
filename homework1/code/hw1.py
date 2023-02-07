import numpy as np
import matplotlib.pyplot as plt

# Multivariate Normal
mvn_samples = np.random.multivariate_normal(mean=[1, -1], cov=[[1, 0], [0, 1]], size=100)
mvn_x, mvn_y = mvn_samples.T
plt.scatter(mvn_x, mvn_y)
plt.title("100 Samples Generated from a MVN Distribution")
plt.savefig("plots/mvn_samples.png")

# Gaussian Mixture
unif_samples = np.random.uniform(size=100)
gm_samples = np.empty(shape=(100, 2))
for i in range(0, 100):
    if unif_samples[i] <= .3:
        gm_samples[i] = (np.random.multivariate_normal(mean=[5, 0], cov=[[1, 0.25], [0.25, 1]], size=1))
    else:
        gm_samples[i] = (np.random.multivariate_normal(mean=[-5, 0], cov=[[1, -0.25], [-0.25, 1]], size=1))

gm_x, gm_y = gm_samples.T
plt.clf()
plt.scatter(gm_x, gm_y)
plt.title("100 Samples Generated from a Gaussian Mixture")
plt.savefig("plots/gm_samples.png")

if __name__ == "__main__":
    pass
