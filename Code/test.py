from RaTGen import *
from mpl_toolkits import mplot3d

import numpy as np
import matplotlib.pyplot as plt


fig = plt.figure()
ax = plt.axes(projection='3d')

rt = RaTGen()
rt.set_dt(.01)

fun = lambda x: 0.1*x

x = rt.generate_sin(1, 1)
y = rt.generate_cos(1,  1)
z = rt.generate_custom(fun, 0, 2*np.pi) + rt.generate_noise(.01, 0, 2*np.pi)
ax.plot3D(x, y, z)

#plt.show()


## Robot test
import roboticstoolbox as rtb

bot = rtb.ERobot.URDF(
    '/Users/jakubadmin/Documents/Uni/Bach/python/LongRatGen/urdf_files_dataset/urdf_files_dataset/urdf_files/ros-industrial/xacro_generated/staubli/staubli_tx2_90_support/urdf/tx2_90l.urdf')  # Change to relative path

bot.plot([0, 0, 0, 0, 0, 0])