import socket
import pickle
import struct
import numpy as np
from numpy import pi
import pandas as pd
from time import sleep

#oajnwdjabndo
#HOST = '192.168.0.254'  # IP-Adresse des Servers
HOST = '192.168.0.254'  # IP-Adresse des Servers
PORT = 6969        # Port des Servers

data = pd.read_csv('data.csv', delimiter=',', header=None).to_numpy()
homepos = data[0]
print("DEBUG: HOMEPOS IS:", homepos)
# Socket erstellen
flag_skip = False
i = 0
with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    try:
        # Verbindung zum Server aufbauen
        s.connect((HOST, PORT))
        print(f"Verbunden mit {HOST}:{PORT}")
        #begüßen
        print("Greeting with 42")
        s.sendall(struct.pack('f', 42))
        for p in data:
            if (p == homepos).all():
                flag_skip = False
            if flag_skip:
                #TODO
                data[i-1] = homepos
                continue
            
            #print(p)
            # Warte bis bot ready schickt
            ready = 0
            while True:
                #ready = s.recv(1024).decode('utf-8')
                ready = list(s.recv(1024))[0]
                print(f"Received: {ready}")
                if ready == 1:
                    break
                elif ready == 123:
                    print("DEBUG: IMPOSSIBLE MOVE SKIPPING MOTION AND RESUMING FROM HOME!")
                    flag_skip = True
                    s.sendall(struct.pack('ffffff', *homepos))
                sleep(0.1)
            #wait 5s before sending data for more robustness
            #print("DEBUG: Waiting 5s before starting to send Data")
            sleep(.1)
            #Send data to bot
            #for c in p:
            print(f"DEBUG: Sending Data: {p} \n as\n {struct.pack('ffffff', *p)}")
            s.sendall(struct.pack('ffffff', *p))
            #if is hompos send 123 to bot to signal him to stop
            if (p == homepos).all():
                sleep(2.5)
                print("DEBUG: WAITING AT HOMEPOS")
                n = 123.0
                s.sendall(struct.pack('f', n))
            i += 1
            #print(f"Gesendet: {p}")
        s.sendall(struct.pack("f", 0.0))
    except Exception as e:
        print(f"Fehler: {e}")
        
        
temp = pd.DataFrame(data)
temp.to_csv("data_executed.csv", header=None)
