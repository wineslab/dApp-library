import threading
try:
    from flask import Flask, render_template
    from flask_socketio import SocketIO
except ModuleNotFoundError:
    print(
        "Optional dependencies for GUI not installed.\n"
        "Fix this by running:\n\n"
        "    pip install 'dApps[gui]'  # OR\n"
        "    pip install 'dApps[all]'\n"
    )
    exit(-1)
import numpy as np

class Dashboard:
    # The default values of the dashboard are based on the default configuration
    def __init__(self, buffer_size: int = 100, ofdm_symbol_size: int = 1272, bw: float = 38.16e6, center_freq: float = 3.6192e9,
                 num_prbs: int = 106, first_carrier_offset: int = 900, prb_protected_below: int = 75,
                 classifier=None, adaptiveThreshold=False, control: bool = False,
                 label_callback=None, initial_label: str = "", port: int = 7778,
                 show_controls: bool = False):
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
        # PRBs 0..prb_protected_below-1 are reserved (BWP/PRACH) and never blacklisted.
        # PRBs prb_protected_below..num_prbs-1 are the eligible spectrum-sharing region.
        self.prb_protected_below = prb_protected_below
        self.control = control

        # Route setup
        self.app.add_url_rule("/", view_func=self.index)

        self.classifier = classifier if classifier is not None else None
        self.adaptiveThreshold = adaptiveThreshold
        self.label_callback = label_callback
        self.current_label = initial_label
        self.port = port
        self.show_controls = show_controls

        # SocketIO event handlers
        self.socketio.on_event("connect", self.handle_initial_connection)
        if self.label_callback is not None:
            def _on_set_ground_truth_label(label):
                self.current_label = label
                self.label_callback(label)
            self.socketio.on_event("set_ground_truth_label", _on_set_ground_truth_label)
        self.socketio.on_event("set_sampling_threshold", self._on_set_sampling_threshold)
        self._initialize_plot()

        # Render a new dashboard frame every N IQ batches — does not affect IQ delivery or recording
        self.sampling_counter = 0
        self.sampling_threshold = 5

    def index(self):
        return render_template("index.html")

    def run(self):
        self.socketio.run(self.app, host="0.0.0.0", port=self.port, debug=False,
                          use_reloader=False, allow_unsafe_werkzeug=True)

    def _initialize_plot(self):
        self.run_thread = threading.Thread(target=self.run, daemon=True)
        self.run_thread.start()

    def handle_initial_connection(self):
        data_buffer = {
            "magnitude": np.zeros((self.buffer_size, self.ofdm_symbol_size)).tolist(),
            "center_freq": self.center_freq,
            "bw": self.bw,
            "num_prbs": self.num_prbs,
            "first_carrier_offset": self.first_carrier_offset,
            # PRB zone boundaries so the frontend can draw visual separators:
            # [0, prb_protected_below)   → protected (BWP/PRACH, never blacklisted)
            # [prb_protected_below, num_prbs) → eligible for spectrum-sharing blacklist
            # [num_prbs, ...)             → guard band (never blacklisted)
            # show_prb_zones is only meaningful when control is active (blocking PRBs);
            # without control there is nothing being blocked so zones would be misleading.
            "show_prb_zones": self.control,
            "prb_protected_below": self.prb_protected_below if self.control else None,
            "predicted_label": self.classifier is not None,
            "adaptive_noise_floor": self.adaptiveThreshold,
            "show_label_selector": self.label_callback is not None,
            "current_label": self.current_label,
            "show_controls": self.show_controls,
            "sampling_threshold": self.sampling_threshold,
        }
        self.socketio.emit("initialize_plot", data_buffer)

    def emit_label(self, label: str):
        self.current_label = label
        self.socketio.emit("update_ground_truth_label", label)

    def _on_set_sampling_threshold(self, value):
        try:
            rate = int(value)
        except (TypeError, ValueError):
            return
        if rate < 1:
            return
        self.sampling_threshold = rate
        self.socketio.emit("update_sampling_threshold", rate)
        self.socketio.emit("reset_waterfall")

    def _process_iq_data(self, iq_data):
        real_part = iq_data[0::2]  # Even indices: real part
        imag_part = iq_data[1::2]  # Odd indices: imaginary part

        iq_complex = real_part + 1j * imag_part
        abs_iq = np.abs(iq_complex)
        magnitude_dB = 20 * np.log10(np.where(abs_iq == 0, 1, abs_iq))

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

        elif self.adaptiveThreshold == True and plot == "adaptive_noise_floor":
            adaptive_noise_floor = payload.tolist()
            self.socketio.emit("update_plot", {"adaptive_noise_floor": adaptive_noise_floor})

    def stop(self):
        """Stops the server and kills the thread."""
        # This is probably overkill since now the run thread it's a demon,
        # still with timeout it does not impact the graceful exit
        if self.run_thread and self.run_thread.is_alive():
            self.run_thread.join(timeout=1)

if __name__ == "__main__":
    import argparse
    import time

    parser = argparse.ArgumentParser(description="Dashboard interactive demo")
    parser.add_argument("--initial-label", default="", metavar="LABEL",
                        help="Ground truth label to pre-populate in the GUI")
    parser.add_argument("--port", type=int, default=7778)
    args = parser.parse_args()

    def on_label(label):
        print(f"[demo] ground_truth_label updated: {label!r}")

    demo = Dashboard(
        label_callback=on_label,
        initial_label=args.initial_label,
        port=args.port,
    )
    print(f"Dashboard running at http://localhost:{args.port}")
    print("Type a label and press Enter to push it to the GUI, or Ctrl-C to quit.")
    try:
        while True:
            label = input("label> ").strip()
            if label:
                demo.emit_label(label)
                print(f"  → pushed {label!r} to browser")
    except (KeyboardInterrupt, EOFError):
        pass
    finally:
        demo.stop()
