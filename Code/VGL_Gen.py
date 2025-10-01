#Programm zum testen der genauigkeit
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from scipy.interpolate import interp1d

import logging
import plotly.io as pio
logging.getLogger('plotly').setLevel(logging.WARNING)
pio.renderers.default = "browser"   # Öffnet den Plot im Browser


def get_rx(a):
    c = np.cos(a)
    s = np.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
def get_ry(a):
    c = np.cos(a)
    s = np.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
def get_rz(a):
    c = np.cos(a)
    s = np.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])

def getR(a, b, c):
    return get_rz(c) @ get_ry(b) @ get_rx(a)
def getRhomo(a, b, c):
    r = getR(a, b, c)
    homo = np.eye(4)
    homo[:3, :3] = r
    return homo
#lesen der Daten
ref = pd.read_csv('data.csv', header=None).to_numpy()
mess = pd.read_csv('TestTwoHalfCircles 1.csv').to_numpy()

mess_rx = mess[:, 2]
mess_ry = mess[:, 3]
mess_rz = mess[:, 4]

rx = np.mean(mess_rx)*180/np.pi
ry = np.mean(mess_ry)*180/np.pi
rz = np.mean(mess_rz)*180/np.pi
T_ref = np.array([[1, 0, 0, ref[0, 0]], [0, 1, 0, ref[0, 1]], [0, 0, 1, ref[0, 2]], [0, 0, 0, 1]])
T_mess = np.array([[1, 0, 0, mess[0, 0]], [0, 1, 0, mess[0, 1]], [0, 0, 1, mess[0, 2]], [0, 0, 0, 1]])
#T_mess[:3, :3] = getR(rx, ry, rz)
#T_mess = T_mess@getRhomo(rx, ry, rz)
#rotate tmess by -90
T_mess = T_mess@getRhomo(0, 0, np.pi/2)
#find optimal rotation:


#Transform both to the same coordinate system by subtracting the first point from all points
#ref_new = ref[:] - ref[0]
#
#mess_new = (mess[:] - mess[0])
ref_new = np.zeros([len(ref[:]), 6])
mess_new = np.zeros([len(mess[:]), 8])



for i in range(len(ref[:, :3])):
    ref_new[i, :3] = (np.linalg.inv(T_ref) @ np.array([ref[i, 0], ref[i, 1], ref[i, 2], 1]))[:3]

for i in range(len(mess[:, :3])):
    mess_new[i, :3] = (np.linalg.inv(T_mess) @ np.array([mess[i, 5], mess[i, 6], mess[i, 7], 1]))[:3]

mess_new = mess_new[:] - mess_new[0]
mess_new[:, 2] = 0


#plot both in 3d
import plotly.graph_objects as go

# Erstelle die 3D-Figur
fig = go.Figure()

# Füge die Traces (Datenreihen) hinzu
fig.add_trace(go.Scatter3d(
    x=mess_new[:, 0],
    y=mess_new[:, 1],
    z=mess_new[:, 2],
    mode='lines',
    name='Measurement',
    line=dict(color='blue', width=4)
))

# Optional: Falls du ref_new einbeziehen möchtest
fig.add_trace(go.Scatter3d(
    x=ref_new[:, 0],
    y=ref_new[:, 1],
    z=ref_new[:, 2],
    mode='lines',
    name='Reference',
    line=dict(color='green', width=4)
))

# Achsenbeschriftungen
fig.update_layout(
    scene=dict(
        xaxis_title='X',
        yaxis_title='Y',
        zaxis_title='Z'
    ),
    title='Interaktiver 3D-Plot',
    margin=dict(l=0, r=0, b=0, t=30)
)

# Zeige den Plot
#fig.show()

#Sync by home
#Aufgrund der formatierung istdas home immer 0, 0, 0, 0, 0, 0


import numpy as np

# --- Beispiel-Daten generieren ---
# Referenzdaten (kleinere Auflösung)
ref = ref_new[:, :3]

# Messdaten (höhere Auflösung, mit Verzögerung am Anfang)
mess = mess_new[:, :3]

home = [0, 0, 0]
#get indizes where arrays are [0, 0, 0]
home_ref_i = np.where((ref == home).all(axis=1))[0]

# --- Funktion zum Aufteilen ---
def split_mess_into_segments(mess):
    """
    Teilt die Messdaten in Bewegungs- und Home-Segmente auf.

    Args:
        mess (np.ndarray): Messdaten (N x 3).

    Returns:
        list of np.ndarray: Liste der Segmente (abwechselnd Bewegung und Home).
    """
    segments = []
    current_segment = []

    for i, point in enumerate(mess):
        if np.all(point == [0, 0, 0]):
            # Wenn das aktuelle Segment nicht leer ist, füge es hinzu (Bewegung)
            if current_segment:
                segments.append(np.array(current_segment))
                current_segment = []
            # Starte ein neues Home-Segment
            current_segment.append(point)
        else:
            # Wenn das aktuelle Segment ein Home-Segment ist, füge es hinzu
            if current_segment and np.all(current_segment[0] == [0, 0, 0]):
                segments.append(np.array(current_segment))
                current_segment = []
            # Füge den Punkt zum Bewegungs-Segment hinzu
            current_segment.append(point)

    # Füge das letzte Segment hinzu
    if current_segment:
        segments.append(np.array(current_segment))

    return segments

# --- Aufteilen ---
mess_segments = split_mess_into_segments(mess)

print("Indizes der [0, 0, 0]-Zeilen:", home_ref_i)
#print("Indizes der [0, 0, 0]-Zeilen:", home_mess_i)

#unterteilung der mess indizes
n_home = len(home_ref_i)
#unterteile ref in segmente
ref_segments = []
for i in range(n_home - 1):
    ref_segments.append(ref[home_ref_i[i]:home_ref_i[i + 1] + 1])
    print("Segment", i, ":", ref_segments[i])

print("mess", mess_segments)
print("ref", ref_segments)
temp = np.array([])
n = 0
for i, p in enumerate(mess_segments):
    if not (p==home).all():
        n += 1
        print("not hoem", n)
print(n)
temp = np.zeros(n+1, dtype=object)
t = 0
for i, p in enumerate(mess_segments):
    if not (p==home).all():
        temp[t] = p
        t += 1

print(temp)
mess_segments = temp

