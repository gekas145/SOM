import numpy as np 
import matplotlib.pyplot as plt
from matplotlib.patches import RegularPolygon

col, row = 2, 3

n, m = 5, 5
radius = 1
d = np.sqrt(3)*radius/2
margin = 0.5
width = 2*margin + d*(2*n + 1)
height = 2*margin + (m + m//2 + m%2)*2*d/np.sqrt(3)

# axial coords
axial = np.array([[[i, j] for j in range(n)] for i in range(m)])
for i in range(axial.shape[0]):
    if i % 2 == 0:
        axial[i, :, 1] *=2
    else:
        axial[i, :, 1] += np.arange(1, n + 1) 

dcol = np.abs(axial[:, :, 1] - axial[row, col, 1])
drow = np.abs(axial[:, :, 0] - axial[row, col, 0])
distances = drow + np.maximum((dcol - drow)/2, 0)

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
        ax.annotate(str(round(distances[j, i])), (x, y), weight='bold', fontsize=10*d, ha='center', va='center')

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