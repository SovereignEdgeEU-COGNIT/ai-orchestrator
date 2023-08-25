import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import beta

# Parameters for the beta distribution
a, b = 2, 5

# Number of samples
n_samples = 1000

# Generate random samples from the beta distribution
samples = beta.rvs(a, b, size=n_samples)

# Plot histogram of the samples
plt.figure(figsize=(8, 6))
plt.hist(samples, bins=50, density=True, alpha=0.6, label='Samples')

# To compare, let's plot the true PDF of the beta distribution
x = np.linspace(0, 1, 1000)
y = beta.pdf(x, a, b)
plt.plot(x, y, 'r-', label=f'Beta({a}, {b})')

plt.title('Sampling from Beta Distribution')
plt.xlabel('Value')
plt.ylabel('Density')
plt.legend()
plt.grid(True)
plt.show()
