import matplotlib.pyplot as plt
import numpy as np

# Parameters
epsilon = 0.1  # Step size
iterations = 20
dim = 2  # Dimension of the distribution


# Potential function U(x) and its gradient
def U(x):
    return np.array([0.5, 10.0]) * np.dot(x, x)


def grad_U(x):
    return np.array([1.0, 20.0]) * x


# Langevin MCMC sampling
x = np.array([2.0, 2.0])  # Initial state
samples = []
for _ in range(1000):
    for _ in range(iterations):
        noise = np.random.normal(0, 1, dim)
        x = x - (epsilon / 2) * grad_U(x) + np.sqrt(epsilon) * noise
    samples.append(x)

samples = np.array(samples)

# Plot the samples
plt.figure(figsize=(6, 6))
plt.plot(samples[:, 0], samples[:, 1], 'o', alpha=0.5)
plt.xlabel('x1')
plt.ylabel('x2')
plt.title('Langevin MCMC Samples from 2D Gaussian')
plt.grid()
plt.show()
