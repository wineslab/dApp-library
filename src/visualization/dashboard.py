import threading
try:
    from flask import Flask, render_template
    from flask_socketio import SocketIO
except ModuleNotFoundError:
                    print(
                        "Optional dependencies for GUI not installed.\n"
                        "Fix this by running:\n\n"
                        "    pip install 'dApps[gui]'  # OR\n"
                        "    pip install 'dApps[all]'\n",
                        exc_info=True
                    )
                    exit(-1)
import numpy as np

class Dashboard:
    # The default values of the dashboard are based on the default configuration
    def __init__(self, buffer_size: int = 100, ofdm_symbol_size: int = 1272, bw: float = 38.16e6, center_freq: float = 3.6192e9, 
                 num_prbs: int = 106, first_carrier_offset: int = 900, classifier = None):
        # Flask app setup
        self.app = Flask(__name__)
        self.app.config['TEMPLATES_AUTO_RELOAD'] = True
        self.socketio = SocketIO(self.app)

        # Parameters
        self.buffer_size = buffer_size
        self.ofdm_symbol_size = ofdm_symbol_size
        self.bw = bw
        self.center_freq = center_freq
        self.num_prbs = num_prbs
        self.first_carrier_offset = first_carrier_offset
        
        # Route setup
        self.app.add_url_rule("/", view_func=self.index)

        self.classifier = classifier if classifier is not None else None

        # SocketIO event handlers
        self.socketio.on_event("connect", self.handle_initial_connection)
        self._initialize_plot()

        # We do not show every iqs otherwise the GUI is too slow
        self.sampling_counter = 0
        self.sampling_threshold = 5

    def index(self):
        return render_template("index.html")

    def run(self):
        self.socketio.run(self.app, host="0.0.0.0", port=7778)

    def _initialize_plot(self):
        self.run_thread = threading.Thread(target=self.run)
        self.run_thread.start()

    def handle_initial_connection(self):
        data_buffer = {
            "magnitude": np.zeros((self.buffer_size, self.ofdm_symbol_size)).tolist(),
            "center_freq": self.center_freq,
            "bw": self.bw,
            "num_prbs": self.num_prbs,
            "first_carrier_offset": self.first_carrier_offset,
            "predicted_label": self.classifier is not None
        }
        self.socketio.emit("initialize_plot", data_buffer)

    def _process_iq_data(self, iq_data):
        real_part = iq_data[0::2]  # Even indices: real part
        imag_part = iq_data[1::2]  # Odd indices: imaginary part

        iq_complex = real_part + 1j * imag_part
        magnitude_dB = 20 * np.log10(np.where(np.abs(iq_complex) == 0, 1, np.abs(iq_complex)))

        return magnitude_dB

    def process_iq_data(self, message):
        """Process IQ data by calling the modified version of process_message."""
        # Assume iq_data and prb_list are provided as strings similar to the original message payloads.
        plot, payload = message
        if plot == "iq_data":
            iq_data = payload

            self.sampling_counter += 1        
            if self.sampling_counter >= self.sampling_threshold:
                magnitude_dB = self._process_iq_data(iq_data)[::-1].tolist() # visualization processing is delegated to client
                self.socketio.emit("update_plot", {"magnitude": magnitude_dB})

                if self.classifier:
                    label = self.classifier.predict(iq_data)
                    self.socketio.emit("update_plot", {"predicted_label": label})

                self.sampling_counter = 0

        elif plot == "prb_list":
            prb_list = payload.tolist()
            self.socketio.emit("update_plot", {"prb_list": prb_list})

    def stop(self):
        """Stops the server and kills the thread."""
        if self.run_thread and self.run_thread.is_alive():
            self.run_thread.join()

if __name__ == "__main__":
    server_app = Dashboard()
