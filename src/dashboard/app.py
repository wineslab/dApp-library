import ast
import socket
import threading
from flask import Flask, render_template
from flask_socketio import SocketIO
import numpy as np

app = Flask(__name__, template_folder='../dapp/templates')
app.config['TEMPLATES_AUTO_RELOAD'] = True
socketio = SocketIO(app)

TCP_IP = "127.0.0.1"
TCP_PORT = 5005
BUFFER_SIZE = 9812
ROLLING_SIZE = 200
FFT_SIZE = 1024
MESSAGE_TERMINATOR = "<END>"


@app.route("/")
def index():
    return render_template("index.html")


@socketio.on("connect")
def handle_inital_connection():
    num_prbs = 106
    data_buffer = {
        "magnitude": np.zeros((ROLLING_SIZE, FFT_SIZE)).tolist(),
        "num_prbs": num_prbs,
        # "ROLLING_SIZE": ROLLING_SIZE
    }
    socketio.emit("initialize_plot", data_buffer)


def process_message(message):
    """Process a single message by extracting the plot and payload."""
    plot, payload = message.split(",", 1)
    if plot == "magnitude":
        magnitude = list(map(float, ast.literal_eval(payload)))
        socketio.emit("update_plot", {"magnitude": magnitude})
    elif plot == "prb_list":
        prbs = list(map(int, ast.literal_eval(payload)))
        socketio.emit("update_plot", {"prb_list": prbs})
    else:
        raise ValueError(f"Unknown plot name {plot}")


def receive_data(conn):
    buffer = ""
    while True:
        data = conn.recv(BUFFER_SIZE)
        if data:
            buffer += data.decode()
            while MESSAGE_TERMINATOR in buffer:
                # Split the buffer by the terminator to process complete messages
                message, buffer = buffer.split(MESSAGE_TERMINATOR, 1)
                process_message(message)
        else:
            print("No data received, closing connection")
            break
    conn.close()


def tcp_server(stop: threading.Event):
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.bind((TCP_IP, TCP_PORT))
    server_socket.listen(5)
    server_socket.settimeout(1)

    print("TCP server listening on port", TCP_PORT)

    while not stop.is_set():
        try:
            conn, addr = server_socket.accept()
            print("Connection address:", addr)
            threading.Thread(target=receive_data, args=(conn,)).start()
        except TimeoutError:
            pass
    print("Server out")


if __name__ == "__main__":
    stop = threading.Event()
    tcp_thread = threading.Thread(target=tcp_server, args=(stop,))
    tcp_thread.start()
    socketio.run(app, host="0.0.0.0", port=7779)
    print("Close TCP server")
    stop.set()
    tcp_thread.join()