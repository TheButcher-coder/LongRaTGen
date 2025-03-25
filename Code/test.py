from RaTGen import *
from mpl_toolkits import mplot3d

import numpy as np
import matplotlib.pyplot as plt


fig = plt.figure()
ax = plt.axes(projection='3d')

rt = RaTGen()
rt.set_dt(.1)

fun = lambda x: x**(2)/4

x = rt.generate_sin(1, 1)
y = rt.generate_sin(1,  1)
z = rt.generate_custom(fun, 0, 2*np.pi)

ax.plot3D(x, y, z)

#plt.show()

R = rt.generate_rot_X(rt.generate_sin(180, 1))
print(R)
p = np.array([1, 2, 3])

rt.make_traj(p, R)
print(rt.get_traj())
