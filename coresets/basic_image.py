import matplotlib.pyplot as plt
import numpy as np

center = (0.5, 0.5)
radius = 0.3

data = np.random.normal(0.5, 0.09, size=(100, 2))

# for infill
circle1 = plt.Circle(center, radius, color='cyan', fill=True, ls='--', alpha=0.1)
# for border
circle2 = plt.Circle(center, radius, color='black', fill=False, ls='--')

fig, ax = plt.subplots() 
ax.add_patch(circle1)
ax.add_patch(circle2)

# plot center
plt.scatter(center[0], center[1], marker='*', color='red', label='Center', s=100)
# plot data
plt.scatter(data[:, 0], data[:, 1], alpha=0.8, color='green', label='Datapoint')
# ensure border points
plt.scatter([0.5, 0.2], [0.8, 0.5], alpha=0.8, color='green')

plt.legend()
plt.xlim(0, 1)
plt.ylim(0, 1)
plt.xlabel('Dimension 0')
plt.ylabel('Dimension 1')

ax.set_rasterized(True)
plt.savefig('minimum_enc_ball.pdf', dpi=300)