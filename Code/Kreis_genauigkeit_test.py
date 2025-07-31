#dieses programm testet die genauigkeit der messung des vicon

import RaTGen as rt
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Initialisierung
rat = rt.RaTGen()
rat.dt = .01
rat.set_mean(0)
rat.set_std_dev(.1)
# Punkte generieren
x = rat.generate_sin(1, 1, 0, 0, 2 * np.pi)
y = rat.generate_cos(1, 1, 0, 0, 2 * np.pi)
z = rat.generate_noise(0, 2 * np.pi)

# Rotation (angenommen: rot ist eine Liste von 3x3-Rotationsmatrizen)
rot = rat.generate_rot_Z_range(0, 2*np.pi)

# Punktwolke p
p = np.array([x, y, z])

# Plot
# Plot
fig = plt.figure()
ax = fig.add_subplot(projection='3d')

# Linie zeichnen
ax.plot(p[0], p[1], p[2], color='blue', label='Kurve p')

# Achsenlabel
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('3D-Kurve und Rotationsvektoren')
ax.legend()

# --- GLEICHE ACHSENVERHÄLTNISSE ---
# Bereichsgrenzen bestimmen
max_range = np.array([p[0].max()-p[0].min(), p[1].max()-p[1].min(), p[2].max()-p[2].min()]).max() / 2.0

mid_x = (p[0].max()+p[0].min()) * 0.5
mid_y = (p[1].max()+p[1].min()) * 0.5
mid_z = (p[2].max()+p[2].min()) * 0.5

ax.set_xlim(mid_x - max_range, mid_x + max_range)
ax.set_ylim(mid_y - max_range, mid_y + max_range)
ax.set_zlim(mid_z - max_range, mid_z + max_range)
# ----------------------------------

plt.show()

#safe data to csv
data = np.zeros([len(p[0]), 16])
for i in range(len(p)):
    data[i, 0:3] = rot[i, 0, :]    # First row of rotation
    data[i, 3] = p[i, 0]           # x position
    data[i, 4:7] = rot[i, 1, :]    # Second row of rotation
    data[i, 7] = p[i, 1]           # y position
    data[i, 8:11] = rot[i, 2, :]   # Third row of rotation
    data[i, 11] = p[i, 2]          # z position
    data[i, 12:16] = [0, 0, 0, 0]  # Bottom row, although normally [0,0,0,1]


#safer data
import pandas as pd
df = pd.DataFrame(data)
df.to_csv('data.csv', index=False, header=False)
print(data)