import socket
import pickle
import numpy as np

HOST = 'localhost'  # IP-Adresse des Servers
PORT = 6969        # Port des Servers

# Socket erstellen
with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    try:
        # Verbindung zum Server aufbauen
        s.connect((HOST, PORT))
        print(f"Verbunden mit {HOST}:{PORT}")

        points = [(0, 0, 0, 0, 0, 0), (np.pi/2, 0, 0, 0, 0, 0), (np.pi, 0, 0, 0, 0, 0)]

        s.sendall(pickle.dumps(points))
        print(f"Gesendet: {points}")

    except Exception as e:
        print(f"Fehler: {e}")
