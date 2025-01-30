try:
    import configparser   
    from influxdb_client import InfluxDBClient, Point
except [ImportError, ModuleNotFoundError] as e:
                    print(e.msg)
                    print(
                        "Optional dependencies for API not installed.\n"
                        "Fix this by running:\n\n"
                        "    pip install 'dApps[api]'  # OR\n"
                        "    pip install 'dApps[all]'\n",
                        exc_info=True
                    )
                    exit(-1)
import time
import atexit
import numpy as np
import logging

logging.basicConfig(level=logging.INFO, format="[Influx] [%(created)f] %(levelname)s - %(message)s", filename="plotter.log", filemode='w')

class InfluxPlotter:
    def __init__(self, config_path: str = None, url: str = None, org: str = None, token: str = None, timeout: int = None, verify_ssl: bool = None):
        # Read configuration from a file if the path is provided
        if config_path:
            config = configparser.ConfigParser()
            config.read(config_path)
            influx_config = config['influx2']
            url = influx_config.get('url')
            org = influx_config.get('org')
            token = influx_config.get('token')
            timeout = influx_config.getint('timeout')
            verify_ssl = influx_config.getboolean('verify_ssl')

        # Set up the InfluxDB client
        self.client = InfluxDBClient(url=url, token=token, org=org, timeout=timeout, verify_ssl=verify_ssl)
        self.org = org
        self.bucket = 'dapp_data'  # Default bucket name for data

        # Register atexit function to ensure InfluxDB connection is closed
        atexit.register(self.close_connection)

    def initialize_plot(self, iq_shape: tuple):
        """
        Initializes the InfluxDB bucket to manage data created by other methods.
        """
        # Initialize bucket and any other settings required for Grafana plotting.
        # InfluxDB 2.0 does not require table creation as it uses a bucket-based schema-less storage.
        logging.info(f"InfluxDB initialized for managing IQ shape: {iq_shape}")

    def process_iq_data(self, iq_data: list[int]):
        """
        Processes and pushes IQ data to InfluxDB.

        Parameters:
            iq_data (list): magnitude list of the sensed OFDM symbols
        """
        write_api = self.client.write_api()
        point = Point("magnitude")
        for i, magnitude in enumerate(iq_data):
            point = point.field(f"magnitude_{i}", magnitude)
        write_api.write(bucket=self.bucket, org=self.org, record=point)
        logging.info(f"IQ data pushed to InfluxDB")

    def process_prb_list(self, prb_list: list[int]):
        """
        Processes and pushes PRB blacklist data to InfluxDB.

        Parameters:
            prb_list (list): List of the PRB blacklisted.
        """
        write_api = self.client.write_api()
        point = Point("prb_list")
        for i, prb in enumerate(prb_list):
            point = point.field(f"prb_{i}", prb)
        write_api.write(bucket=self.bucket, org=self.org, record=point)
        logging.info(f"PRB list pushed to InfluxDB")

    def close_connection(self) -> None:
        """
        Ensures that the connection to InfluxDB is properly closed.
        """
        self.client.close()
        logging.info("InfluxDB connection closed.")

# Example usage:
if __name__ == "__main__":
    plotter = InfluxPlotter(config_path="influx_config.ini")
    plotter.initialize_plot((1024, 1024))
    
    for i in range(6000):
        # Generate random IQ data (1024 numbers)
        iq_data = list(np.random.uniform(8, 80, size=1024))
        plotter.process_iq_data(iq_data)
        
        # Generate random PRB list (128 integers)
        prb_list = list(np.random.randint(0, 100, size=128))
        plotter.process_prb_list(prb_list)
        time.sleep(0.1)