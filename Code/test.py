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
import exudyn as exu
from exudyn.robotics import Robot
import os

# SystemContainer und MultibodySystem erstellen
SC = exu.SystemContainer()
mbs = SC.AddSystem()

# Pfad zur URDF-Datei
urdf_path = os.path.join('/Users/jakubadmin/Documents/Uni/Bach/python/LongRatGen/Code/data/tx2-90L/urdf/tx2_90l.urdf')

# Roboter aus URDF erstellen
robot = Robot(urdf_path, mbs)

# Visualisierungseinstellungen
SC.visualizationSettings.nodes.defaultSize = 0.01
SC.visualizationSettings.openGL.multiSampling = 4
SC.visualizationSettings.openGL.shadow = 0.5
SC.visualizationSettings.window.renderWindowSize = [1200, 900]

# Simulation starten
mbs.Assemble()
exu.StartRenderer()
mbs.SolveDynamic()
exu.WaitForUserToContinue()
exu.StopRenderer()

