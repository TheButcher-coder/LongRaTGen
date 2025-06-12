from RaTGen import *
from mpl_toolkits import mplot3d
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3.common.noise import OrnsteinUhlenbeckActionNoise  # Noise gen

fig = plt.figure()
ax = plt.axes(projection='3d')

rt = RaTGen()
rt.set_dt(.01)
rt.set_mean(0)
rt.set_std_dev(.5)
rt.set_max_vel(5)  # m/s
rt.set_max_accel(160)  # m/s^2

t0 = 0
tmax = 4*np.pi

fun = lambda x: 0.5 * x

x = rt.generate_sin(1, .5, 0, t0, tmax)
y = rt.generate_cos(1, .5, 0, t0, tmax)
z = rt.generate_sin(.5, 10, 0, t0, tmax) + rt.generate_noise(t0, tmax)

# Plot original trajectory
ax.plot3D(x, y, z, label='Original')

z = rt.smooth(z)
z.resize(len(x))
ax.plot3D(x, y, z, label='Smoothed')

ax.legend()
plt.show()

