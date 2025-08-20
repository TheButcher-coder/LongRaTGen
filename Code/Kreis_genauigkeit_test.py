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

#home position
rat.set_nullpos(np.array(([250, 250, 500, 0, 0, 0])))

x = rat.generate_sin(.5, 1, 0, 0, 1/2*np.pi)*250
y = rat.generate_cos(.5, 1, 0, 0, 1/2*np.pi)*250
z = np.ones(len(x))*600

# Rotation (angenommen: rot ist eine Liste von 3x3-Rotationsmatrizen)
rot = np.zeros(len(x))
# da val3 keine transformationsmatrix entgegennimmt, sondern 3 Achswinkel muss nicht transformiert werden
rx = np.zeros(len(x))
ry = np.zeros(len(x))
rz = np.zeros(len(rx))

# Punktwolke p
#rot = np.array([rx, ry, rz])
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

#plt.show()

#safe data to csv
data = np.zeros([len(x), 6])
data[:, 0] = x
data[:, 1] = y
data[:, 2] = z
data[:, 3] = rx
data[:, 4] = ry
data[:, 5] = rz



#safer data
import pandas as pd
df = pd.DataFrame(rat.get_nullpos())
df = pd.concat([df, pd.DataFrame(data)])
df = pd.concat([df, pd.DataFrame(rat.get_nullpos())])
df.to_csv('data.csv', index=False, header=False)
print(data)