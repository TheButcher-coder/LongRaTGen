import socket
import pickle
import numpy as np
import pandas as pd
from time import sleep


HOST = 'localhost'  # IP-Adresse des Servers
PORT = 6969        # Port des Servers

data = pd.read_csv('random_100x6.csv', header=None).to_numpy()


# Socket erstellen
with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    try:
        # Verbindung zum Server aufbauen
        s.connect((HOST, PORT))
        print(f"Verbunden mit {HOST}:{PORT}")

        for p in data:
            # Warte bis bot ready schickt
            ready = 0
            while True:
                ready = s.recv(1024).decode('utf-8')
                print(f"Received: {ready}")
                if ready == "1":
                    break
                sleep(0.1)

            s.sendall(pickle.dumps(p))
            print(f"Gesendet: {p}")

    except Exception as e:
        print(f"Fehler: {e}")
