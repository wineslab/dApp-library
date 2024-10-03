import socket
import threading
import logging

# Configure logging for E3Interface
e3_logger = logging.getLogger("e3_logger")
e3_logger.setLevel(logging.DEBUG)
e3_handler = logging.FileHandler("./logs/e3.log")
e3_handler.setLevel(logging.DEBUG)
e3_formatter = logging.Formatter("[E3] [%(created)f] %(levelname)s - %(message)s")
e3_handler.setFormatter(e3_formatter)
e3_logger.addHandler(e3_handler)


class E3Interface:
    _instance = None
    _lock = threading.Lock()

    def __new__(cls, server_ip, port, *args, **kwargs):
        with cls._lock:
            if not cls._instance:
                cls._instance = super(E3Interface, cls).__new__(cls, *args, **kwargs)
        return cls._instance

    def __init__(self, server_ip, port, *args, **kwargs):
        if not hasattr(self, "initialized"):
            self.server_ip = server_ip
            self.port = port
            self.callbacks = []
            self.stop_event = threading.Event()
            self.initialized = True
            self.server_thread = threading.Thread(target=self._tcp_server)
            self._start_tcp_server()

    def __del__(self):
        self.stop_server()

    def _start_tcp_server(self):
        self.server_thread.daemon = True
        self.server_thread.start()

    def _tcp_server(self):
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.bind((self.server_ip, self.port))
        sock.listen(5)
        e3_logger.info(f"TCP server listening on {self.server_ip}:{self.port}")

        try:
            while not self.stop_event.is_set():
                sock.settimeout(5.0)  # Set timeout to periodically check stop_event
                try:
                    conn, addr = sock.accept()
                    threading.Thread(target=self._handle_client, args=(conn,)).start()
                except socket.timeout:
                    continue
        finally:
            sock.close()
            e3_logger.info("TCP server stopped.")

    # This should be the actual E3 data handling, for this demo we use the one with Rajeev format
    # def _handle_client(self, conn):
    #     with conn:
    #         while not self.stop_event.is_set():
    #             try:
    #                 conn.settimeout(1.0)  # Set timeout to periodically check stop_event
    #                 data = conn.recv(1024)
    #                 if not data:
    #                     break
    #                 self._handle_incoming_data(data)
    #             except socket.timeout:
    #                 continue

    # message format: length of the buffer (4 bytes) + buffer
    def _handle_client(self, conn):
        try:
            while True:
                sense_symbol = b""
                num_bytes = conn.recv(4)
                if not num_bytes:
                    break
                buf_size = int.from_bytes(num_bytes, "big")
                e3_logger.debug(f"I expect {buf_size}")
                while buf_size > 0:
                    iq_buf = conn.recv(buf_size)
                    if not iq_buf:
                        break
                    sense_symbol += iq_buf
                    buf_size -= len(iq_buf)
                self._handle_incoming_data(sense_symbol)
        except Exception as e:
            e3_logger.error(f"Error handling client: {e}")
        finally:
            e3_logger.debug("Close connection")
            conn.close()

    def _handle_incoming_data(self, data):
        for callback in self.callbacks:
            e3_logger.debug("Launch callback")
            callback(data)

    def add_callback(self, callback):
        if callback not in self.callbacks:
            e3_logger.debug("Add callback")
            self.callbacks.append(callback)

    def remove_callback(self, callback):
        if callback in self.callbacks:
            e3_logger.debug("Remove callback")
            self.callbacks.remove(callback)

    def stop_server(self):
        e3_logger.debug("Stop event")
        self.stop_event.set()
        self.server_thread.join()


if __name__ == "__main__":
    # Usage Example
    def sample_callback(data):
        print(f"Callback called with data: {data.decode()}")

    # Initialize the singleton instance
    e3_interface = E3Interface("127.0.0.1", 9999)
    e3_interface.add_callback(sample_callback)

    # Remove a callback
    e3_interface.remove_callback(sample_callback)

    # Stop the server (for cleanup or at program exit)
    e3_interface.stop_server()
