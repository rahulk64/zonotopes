import numpy as np
import matplotlib.pyplot as plt

e1 = np.array([4, 2, 2, 3, 1, -2, -1])
e2 = np.array([2, 1, -3, -1, 1, 4, 1])

x = []
y = []

size = 5

for a in np.linspace(-1, 1, size):
	for b in np.linspace(-1, 1, size):
		for c in np.linspace(-1, 1, size):
			for d in np.linspace(-1, 1, size):
				for e in np.linspace(-1, 1, size):
					for f in np.linspace(-1, 1, size):
						for g in np.linspace(-1, 1, size):
							for h in np.linspace(-1, 1, size):
								arr = np.array([a, b, c, d, e, f, g])
								x.append(np.dot(e1, arr))
								y.append(np.dot(e2, arr))

colors = (0,0,0)

plt.scatter(x, y, c=colors)
plt.show()
