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
import numpy as np
import os

print(os.environ.get('DISPLAY'))

class IQPlotter:
    def __init__(self, buffer_size=100, iq_size=1536, bw=46.08e6, center_freq=3.6192e9):
        """
        Initialize the IQ plotter with buffer size and IQ data dimensions.
        buffer_size: Number of columns (timesteps) to store and plot.
        iq_size: Size of the IQ data, should be double the size of ofdm_symbol_size with real and imaginary interleaved.
        bw: Sampling frequency.
        center_freq: Center frequency of the signal.
        """
        self.buffer_size = buffer_size
        self.iq_size = iq_size
        self.bw = bw
        self.center_freq = center_freq
        self.iq_shape = (iq_size // 2, buffer_size)  # Half the size after splitting real and imaginary parts
        
        # Initialize the buffer
        self.buffer = np.zeros(self.iq_shape)

        # We do not show everything otherwise the GUI is too slow
        self.sampling_counter = 0
        self.sampling_threshold = 2

        self.fig, self.ax, self.img = self.initialize_plot(self.iq_shape)

    def initialize_plot(self, iq_shape):
        """
        Initialize the plot with empty data.
        iq_shape: Shape of the IQ data (rows, columns).
        Returns the figure, axes, and image object for later updates.
        """
        fig, ax = plt.subplots(figsize=(12, 8))
        img = ax.imshow(np.zeros(iq_shape), aspect='auto', interpolation='none', origin='lower')
        
        # Initialize colorbar
        plt.colorbar(img)

        # Define y-ticks and labels for frequency
        # yticks = np.linspace(0, iq_shape[0], 5)
        # yticklabels = np.linspace(self.center_freq + self.bw / 2, self.center_freq - self.bw / 2, 5)
        # ax.set_yticks(yticks)
        # ax.set_yticklabels([f'{label:.5e}' for label in yticklabels])

        ax.set_xlabel('Time (samples)', fontsize=14)
        ax.set_ylabel('OFDM Symbols', fontsize=14)
        # ax.set_ylabel('Frequency (Hz)', fontsize=14)
        ax.set_title('Magnitude of IQ Data in Frequency Domain (dB)', fontsize=16)

        return fig, ax, img

    def update_plot(self, new_data):
        """
        Update the plot with new data.
        new_data: New magnitude data to display.
        """
        self.img.set_data(new_data)
        self.img.autoscale()  # Automatically adjust the color scaling
        plt.draw()
        plt.pause(0.0001)  # Small pause to refresh the plot

    def _process_iq_data(self, iq_data):
        # Separate real and imaginary parts
        real_part = iq_data[0::2]  # Even indices: real part
        imag_part = iq_data[1::2]  # Odd indices: imaginary part

        # Create complex IQ data
        iq_complex = real_part + 1j * imag_part

        # Compute magnitude in dB
        magnitude_dB = 20 * np.log10(np.where(np.abs(iq_complex) == 0, 1, np.abs(iq_complex)))

        return magnitude_dB

    def process_iq_data(self, iq_data):
        """
        Process IQ data and update the rolling buffer and the plot.
        iq_data: Complex IQ data (real and imaginary parts already combined).
        """
        # Ensure the input data is of the expected size
        if iq_data.size != self.iq_size:
            raise ValueError(f'Expected IQ data of size {self.iq_size}, but got {iq_data.size}')
        
        if self.sampling_counter < self.sampling_threshold: 
            magnitude_dB = self._process_iq_data(iq_data)

            # Shift the buffer: drop the oldest column, append the new data
            self.buffer = np.roll(self.buffer, -1, axis=1)
            self.buffer[:, -1] = magnitude_dB  # Add new data in the last column

            self.update_plot(self.buffer)
        else:
            self.sampling_counter = 0
        self.sampling_counter += 1

if __name__ == "__main__":
    # Create an instance of IQPlotter
    plotter = IQPlotter(buffer_size=100, iq_size=1536)

    # Simulate a loop where IQ data is processed and plotted at each timestep
    for t in range(200):  # Simulate 200 iterations (replace this with your actual loop)
        # Simulate new incoming IQ data at each timestep (1536 * 1 array, with real and imaginary parts interleaved)
        iq_data = np.random.randint(-1000, 1000, size=(1536,), dtype='int16')  # Replace with actual IQ data
        
        # Process and update the plot with the new data
        plotter.process_iq_data(iq_data)
