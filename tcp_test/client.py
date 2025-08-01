import socket
import pickle
import struct
import numpy as np
from numpy import pi
import pandas as pd
from time import sleep

#oajnwdjabndo
#HOST = '192.168.0.254'  # IP-Adresse des Servers
HOST = 'localhost'  # IP-Adresse des Servers
PORT = 6969        # Port des Servers

data = pd.read_csv('../Code/data.csv', delimiter=',', header=None).to_numpy()

for p in data:
    print(f"p: {p}")


# Socket erstellen
with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    try:
        # Verbindung zum Server aufbauen
        s.connect((HOST, PORT))
        print(f"Verbunden mit {HOST}:{PORT}")
        #begüßen
        print("Greeting with 42")
        s.sendall(struct.pack('f', 42))
        for p in data:
            #print(p)
            # Warte bis bot ready schickt
            ready = 0
            while True:
                #ready = s.recv(1024).decode('utf-8')
                ready = list(s.recv(1024))[0]
                print(f"Received: {ready}")
                if ready == 1:
                    break
                sleep(0.1)
            #wait 5s before sending data for more robustness
            #print("DEBUG: Waiting 5s before starting to send Data")
            sleep(.1)
            #Send data to bot
            #for c in p:
            print(f"DEBUG: Sending Data: {p} \n as\n {struct.pack('ffffff', p)}")
            s.sendall(struct.pack('ffffff', p))
            #print(f"Gesendet: {p}")

    except Exception as e:
        print(f"Fehler: {e}")