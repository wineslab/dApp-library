import numpy as np
try:
    import matplotlib
except ModuleNotFoundError:
                    print(
                        "Optional dependencies for GUI not installed.\n"
                        "Fix this by running:\n\n"
                        "    pip install 'dApps[gui]'  # OR\n"
                        "    pip install 'dApps[all]'\n",
                        exc_info=True
                    )
                    exit(-1)
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

class EnergyPlotter:
    def __init__(self, fft_size=1536, bw=40e6, center_freq=3.619e9):
        """
        Initialize the energy plotter.

        fft_size: Number of FFT bins (must match the dApp's fft_size).
        bw: Bandwidth in Hz (for axis labelling).
        center_freq: RF center frequency in Hz (for title).
        """
        self.FFT_SIZE = fft_size
        self.center_freq = center_freq
        self.bw = bw
        self.line2 = None  # lazily created when noise floor data first arrives
        self.fig, self.ax, self.line1 = self.initialize_plot(fft_size)

    def initialize_plot(self, fft_size):
        """
        Initialize the plot with empty data.
        iq_shape: Shape of the IQ data (rows, columns).
        Returns the figure and axes object for later updates.
        """
        plt.ion()
        fig, ax = plt.subplots(figsize=(10, 8))
        self.x = np.arange(fft_size)
        y = np.zeros(fft_size)
        line1, = ax.plot(self.x, y, label="Signal power")
        ax.set_ylim([0, 80])
        plt.title(
            f"Spectrum Sensing at gNB"
            f" (Carrier @{self.center_freq/1e9:.2f} GHz,"
            f" BW = {self.bw/1e6:.2g} MHz)",
            fontsize=18,
        )
        plt.xlabel("Subcarrier", fontsize=12)
        plt.ylabel("Energy [dB]", fontsize=12)
        plt.show()
        return fig, ax, line1

    def process_iq_data(self, data):
        """Update the plot with new spectrum data.

        Args:
            data: Either a plain ``np.ndarray`` (legacy, single line) or a
                ``tuple[np.ndarray, np.ndarray | None]`` of
                ``(power_db, noise_floor_db)``.  Both arrays must be in
                first-carrier-aligned space; this method applies the standard
                ``FFT_SIZE // 2`` centering shift for display.
        """
        if isinstance(data, tuple):
            power_db, noise_floor_db = data
        else:
            power_db, noise_floor_db = data, None

        # Center the spectrum for display (DC at the middle of the x-axis).
        # np.roll(arr, -(N//2)) is equivalent to the previous
        # np.append(arr[N//2:], arr[0:N//2]).
        power_shifted = np.roll(power_db, -(self.FFT_SIZE // 2))
        self.line1.set_ydata(power_shifted)

        if noise_floor_db is not None:
            if self.line2 is None:
                # Lazily create the noise floor line on first use (adaptive mode only)
                self.line2, = self.ax.plot(
                    self.x,
                    np.zeros(self.FFT_SIZE),
                    color="orange",
                    linestyle="--",
                    linewidth=1.2,
                    label="Noise floor",
                )
                self.ax.legend(loc="upper right", fontsize=10)
            else:
                # Re-show the line if it was previously hidden
                self.line2.set_visible(True)
            nf_shifted = np.roll(noise_floor_db, -(self.FFT_SIZE // 2))
            self.line2.set_ydata(nf_shifted)
        elif self.line2 is not None:
            # Hide the noise floor line if it was shown before but data is gone
            self.line2.set_visible(False)

        self.fig.canvas.draw()
        self.fig.canvas.flush_events()


if __name__ == "__main__":
    # Create an instance of EnergyPlotter
    plotter = EnergyPlotter(fft_size=1536)

    # Simulate a loop where IQ data is processed and plotted at each timestep
    for t in range(200):
        iq_data = np.random.randint(-1000, 1000, size=(1536,), dtype='int16')
        plotter.process_iq_data(iq_data)
