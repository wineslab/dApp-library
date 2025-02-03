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
    def __init__(self, buffer_size=100, iq_size=1536, bw=46.08e6, center_freq=3.6192e9, classifier=None):
        # TODO double check killing conditions
        # Flask app setup
        self.app = Flask(__name__)
        self.app.config['TEMPLATES_AUTO_RELOAD'] = True
        self.socketio = SocketIO(self.app)

        # Parameters
        self.BUFFER_SIZE = buffer_size
        self.iq_size = iq_size
        self.BW = bw
        self.CENTER_FREQ = center_freq
        
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
        num_prbs = 106
        iqs_to_show = int(
            self.iq_size / 2
        )  # / 2 since the size received is real and imaginary part
        # Each PRB corresponds to 12 subcarriers
        # Parameter first_carrier_offset (currently set to 900) is the first subcarrier.
        # So PRB 0 is [900-911], 1 is [912-923], etc. They wrap around fft_size (now at 1536)
        # The carrier frequency is in the middle PRB (including DC leak on some radios), this means it falls at (first_carrier_offset + 106*12/2) % fft_size? where % is the modulo operation
        # The first 76 PRB usually have channels that should not be nulled (but we should investigate more options to enable this)
        # So if you want the spectrum waterfall to correspond to the PRB masking plot, you can rotate them as per bullet 2 above and drop [635-899]
        iqs_to_show -= 264  # is the size of the array to drop
        data_buffer = {
            "magnitude": np.zeros((self.BUFFER_SIZE, iqs_to_show)).tolist(),
            "num_prbs": num_prbs,
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
            if iq_data.size != self.iq_size:
                raise ValueError(f'Expected IQ data of size {self.iq_size}, but got {iq_data.size}')

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
            self.socketio.stop()
            self.run_thread.join()

if __name__ == "__main__":
    server_app = Dashboard()
