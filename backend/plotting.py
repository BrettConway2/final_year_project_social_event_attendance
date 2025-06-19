import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# Parameters for the distributions
mean1, std1 = 0.71225, 0.5425
mean2, std2 = 0.84942, 0.128

# Create a range of x values that cover both distributions well
x = np.linspace(min(mean1 - 4*std1, mean2 - 4*std2), max(mean1 + 4*std1, mean2 + 4*std2), 1000)

# Compute the probability density functions
pdf1 = norm.pdf(x, mean1, std1)
pdf2 = norm.pdf(x, mean2, std2)

# Plot the distributions
plt.figure(figsize=(10, 6))
plt.plot(x, pdf1, label="Embedding distance of same people")
plt.plot(x, pdf2, label="Embedding distance of different people")
plt.title('Comparison of distributions for RetinaFace embedding distances for figure matching')
plt.xlabel('Value')
plt.ylabel('Probability Density')
plt.grid(True)

# Move legend outside the plot
plt.legend(loc='upper left', bbox_to_anchor=(1.05, 1), borderaxespad=0.)

plt.tight_layout()  # Adjust layout to make room for the legend
plt.show()
