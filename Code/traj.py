import RaTGen as rt
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Initialisierung
rat = rt.RaTGen()
rat.dt = .01
rat.set_mean(0)
rat.set_std_dev(.1)
# Punkte generieren

#home position
rat.set_nullpos(np.array(([250, 250, 750, 0, 0, 0])))

#Programm zur generierung mehrerer verschiedener randomisierter Trajektorien
def half_circ():
    x = rat.generate_sin(250, 1, 0, 0, np.pi)
    y = rat.generate_cos(250, 1, 0, 0, np.pi)
    z = np.ones(len(x))*750


    rx = np.zeros(len(x))
    ry = np.zeros(len(x))
    rz = np.zeros(len(rx))
    return np.array([x, y, z, rx, ry, rz])

def sin_z():
    z = rat.generate_sin(100, 1, 0, 0, 2*np.pi)
    x = np.ones(len(z))*300
    y = np.ones(len(z))*300#

    rx = np.zeros(len(z))
    ry = np.zeros(len(z))
    rz = np.zeros(len(rx))
    return np.array([x, y, z, rx, ry, rz])

def rot_sin():
    rx = rat.generate_sin(45, 0.75, 0, 0, 2*np.pi)
    ry = np.zeros(len(rx))
    rz = np.zeros(len(rx))

    x = np.ones(len(rx))*300
    y = np.ones(len(rx))*300
    z = np.ones(len(rx))*750
    return np.array([x, y, z, rx, ry, rz])


def half_circ_noise():
    traj = half_circ()
    traj[2, :] += rat.generate_noise(0, np.pi) #fix
    return traj
def sin_z_noise():
    traj = sin_z()
    traj[2, :] += rat.generate_noise(0, 2*np.pi)
    return traj
def rot_sin_noise():
    traj = rot_sin()
    traj[3, :] += rat.generate_noise(0, 2*np.pi)
    return traj


#generierung der n Trajektorien mit #
n = 1   # Anzahl Wiederholungen

# alle Trajektorien in eine Liste sammeln
traj_blocks = []

home = pd.DataFrame(rat.get_nullpos().reshape(-1, 1)) # (1,6)

data = pd.DataFrame()
for i in range(n):
    data = pd.concat([data, home.T])
    data = pd.concat([data, pd.DataFrame(half_circ()).T])
    data = pd.concat([data, home.T])
    data = pd.concat([data, pd.DataFrame(sin_z()).T])
    data = pd.concat([data, home.T])
    data = pd.concat([data, pd.DataFrame(rot_sin()).T])
    data = pd.concat([data, home.T])
    data = pd.concat([data, pd.DataFrame(half_circ_noise()).T])
    data = pd.concat([data, home.T])
    data = pd.concat([data, pd.DataFrame(sin_z_noise()).T])
    data = pd.concat([data, home.T])
    data = pd.concat([data, pd.DataFrame(rot_sin_noise()).T])
    data = pd.concat([data, home.T])


data.to_csv("traj.csv", index=False, header=False)

print("Shape von data:", data.shape)
print(data.head())