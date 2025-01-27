import socket
import time
import numpy as np

TCP_IP = '127.0.0.1'
TCP_PORT = 5005

def create_magnitude(initial_size=400, jammer_size=40, total_symbols=1024):
    # Ensure initial_size is within valid bounds
    if initial_size + jammer_size > total_symbols:
        initial_size = total_symbols - jammer_size

    left_size = total_symbols - initial_size - jammer_size
    floor_1 = list(np.round(np.random.uniform(8, 15, size=initial_size), 2))
    jammer = list(np.round(np.random.uniform(60, 80, size=jammer_size), 2))
    floor_2 = list(np.round(np.random.uniform(8, 15, size=left_size), 2))
    
    return floor_1 + jammer + floor_2, initial_size


if __name__ == '__main__':
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client_socket.connect((TCP_IP, TCP_PORT))
    initial_size = 400
    total_symbols = 1024
    try:
        for i in range(6000):
            # Generate random IQ data
            if i % 1000 == 0:
                initial_size += 10
                print(f'Shift')
            magnitude, initial_size = create_magnitude(initial_size)
            magnitude = list(map(float, magnitude))

            message = f"magnitude,{magnitude}<END>"
            # print(len(message))
            client_socket.send(message.encode())

            # Generate random PRB list (128 integers)
            prb_list = np.random.randint(0, 105, size=5).tolist()
            message = f"prb_list,{prb_list}<END>"
            # print(len(message))
            client_socket.send(message.encode())
            time.sleep(1.05)

    finally:
        client_socket.close()
