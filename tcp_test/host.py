import socket
import pickle

tcp_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
tcp_socket.bind(('localhost', 6969))
tcp_socket.listen(1)

while True:
    print("Waiting for connection")
    connection, client = tcp_socket.accept()

    try:
        print("Connected to client:", client)

        # Read all data until the client closes the connection
        data = b""
        while True:
            packet = connection.recv(4096)
            if not packet:
                break
            data += packet

        # Unpickle full message
        obj = pickle.loads(data)
        print("Received:", obj)

    except Exception as e:
        print("Error:", e)

    finally:
        connection.close()
