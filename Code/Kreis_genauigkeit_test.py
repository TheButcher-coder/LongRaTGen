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

