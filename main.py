import numpy as np 
import matplotlib.pyplot as plt
from matplotlib.patches import RegularPolygon


n, m = 10, 10
radius = 1
d = np.sqrt(3)*radius/2
margin = 0.5
width = 2*margin + d*(2*n + 1)
height = 2*margin + (m + m//2 + m%2)*2*d/np.sqrt(3)

fig = plt.figure()
  
ax = fig.add_subplot(111)
ax.set_aspect('equal', adjustable='box')
ax.set_xlim([0, width])
ax.set_ylim([0, height])

for i in range(n):
    x = margin + d*(2*i + 1)
    y = y = height - margin - np.sqrt(3)*d/2
    for j in range(m):
        ax.add_patch(RegularPolygon((x, y), 6, radius=radius, lw=1, edgecolor="black"))

        if j % 2 == 0:
            x += d
        else:
            x -= d
        y -= np.sqrt(3)*d

# x = margin
# if n % 2 == 0:
#     x += 2*d
# else:
#     x += d

# y = height - margin - np.sqrt(3)*d/2

plt.show()