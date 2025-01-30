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
    def __init__(self, fft_size=1536, bw=40e6, center_freq=3.619e9) :
        """
        Initialize the IQ plotter with buffer size and IQ data dimensions.
        fft_size: Size of the FFT.
        """
        self.FFT_SIZE = fft_size
        self.center_freq = center_freq
        self.bw = bw
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
        y = np.arange(fft_size)
        line1, = ax.plot(self.x, y)  
        ax.set_ylim([0, 80])
        plt.title(f"Spectrum Sensing at gNB (Carrier @{self.center_freq/1e9:.2f} GHz, BW = {self.bw/1e6:.2} MHz)", fontsize=18)
        plt.xlabel("Subcarrier", fontsize=12)
        plt.ylabel("Energy [dB]", fontsize=12)
        plt.show()
         
        return fig, ax, line1

    def update_plot(self, new_data):
        """
        Update the plot with new data.
        new_data: New data to display.
        """
        self.line1.set_xdata(self.x)
        self.line1.set_ydata(new_data)
        self.fig.canvas.draw()
        self.fig.canvas.flush_events() 

    def process_iq_data(self, abs_iq_av_db):
        """
        Process IQ data and update the rolling buffer and the plot.
        iq_data: Complex IQ data (real and imaginary parts already combined).
        """
        abs_iq_av_db_shift = np.append(abs_iq_av_db[self.FFT_SIZE//2:self.FFT_SIZE],abs_iq_av_db[0:self.FFT_SIZE//2])

        # Update the plot with the new buffer
        self.update_plot(abs_iq_av_db_shift)

if __name__ == "__main__":
    # TODO merge this class with the other 
    # Create an instance of EnergyPlotter
    plotter = EnergyPlotter(buffer_size=100, iq_size=1536)

    # Simulate a loop where IQ data is processed and plotted at each timestep
    for t in range(200):  # Simulate 200 iterations (replace this with your actual loop)
        # Simulate new incoming IQ data at each timestep (1536 * 1 array, with real and imaginary parts interleaved)
        iq_data = np.random.randint(-1000, 1000, size=(1536,), dtype='int16')  # Replace with actual IQ data
        
        # Process and update the plot with the new data
        plotter.process_iq_data(iq_data)