import RaTGen as rt
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Initialisierung
rat = rt.RaTGen()
rat.dt = .01
# Punkte generieren
x = rat.generate_sin(1, 1, 0, 0, 2 * np.pi)
y = rat.generate_cos(1, 1, 0, 0, 2 * np.pi)
z = np.zeros(len(x))

# Rotation (angenommen: rot ist eine Liste von 3x3-Rotationsmatrizen)
rot = rat.generate_rot_Z_range(0, 2*np.pi)

rx = z
ry = z
rz =
# Punktwolke p
p = np.array([x, y, z])

# Plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Linie zeichnen
ax.plot(p[0], p[1], p[2], color='blue', label='Kurve p')


# Achsenlabel
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('3D-Kurve und Rotationsvektoren')
ax.legend()

plt.show()
