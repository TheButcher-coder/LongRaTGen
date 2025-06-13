import socket
import pickle
from time import sleep

tcp_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
tcp_socket.bind(('localhost', 6969))
tcp_socket.listen(1)

while True:
    print("Waiting for connection")
    connection, client = tcp_socket.accept()
    print("Connected to client:", client)
    done = 0
    while done == 0:
        try:


            # Send ready
            response = "1"
            connection.sendall(response.encode('utf-8'))
            print("Sent ready=1")

            #read all data
            #data = b""
            data = connection.recv(8*4096)

            # Debug: Print raw data
            #print("Raw data received:", data)

             # Unpickle full message
            try:
                obj = pickle.loads(data)
                print("Received:", obj)
            except Exception as e:
                print("Error during unpickling:", e)

            response = "0"
            print("Sent ready=0")
            connection.sendall(response.encode('utf-8'))
            # Fake wait time
            print("Waiting for 5 seconds before sending response")
            sleep(5)
            # Send a response back to the client
            response = "1"
            print("sent ready=1")
            connection.sendall(response.encode('utf-8'))

        except Exception as e:
            print("Error:", e)

    connection.close()