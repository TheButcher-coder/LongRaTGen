from RaTGen import *
from mpl_toolkits import mplot3d
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3.common.noise import OrnsteinUhlenbeckActionNoise  # Noise gen

fig = plt.figure()
ax = plt.axes(projection='3d')

rt = RaTGen()
rt.set_dt(.01)
rt.set_mean(10)
rt.set_std_dev(1)
rt.set_max_vel(100)  # m/s
rt.set_max_accel(10000)  # m/s^2

fun = lambda x: 0.5 * x

x = rt.generate_sin(1, 1)
y = rt.generate_cos(1, 1)
z = rt.generate_sin(.5, 10)

# Plot original trajectory
ax.plot3D(x, y, z, label='Original')

z = rt.smooth_add_noise2(z)
z.resize(len(x))
ax.plot3D(x, y, z, label='Smoothed')

ax.legend()
plt.show()
