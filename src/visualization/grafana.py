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
import atexit
import logging

logging.basicConfig(level=logging.INFO, format="[Grafana] [%(created)f] %(levelname)s - %(message)s", filename="./logs/plotter.log", filemode='a')

class GrafanaPlotter:
    def __init__(self, config_path=None, url=None, org=None, token=None, timeout=None, verify_ssl=None):
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

    def initialize_plot(self, iq_shape):
        """
        Initializes the InfluxDB bucket to manage data created by other methods.
        """
        # Initialize bucket and any other settings required for Grafana plotting.
        # InfluxDB 2.0 does not require table creation as it uses a bucket-based schema-less storage.
        logging.info(f"InfluxDB initialized for managing IQ shape: {iq_shape}")

    def process_iq_data(self, iq_data):
        """
        Processes and pushes IQ data to InfluxDB.

        Parameters:
            iq_data (list): Complex IQ data (real and imaginary parts already combined).
        """
        write_api = self.client.write_api()
        point = Point("iq_data").field("value", iq_data)
        write_api.write(bucket=self.bucket, org=self.org, record=point)
        logging.info(f"IQ data pushed to InfluxDB: {iq_data}")

    def process_prb_list(self, prb_list):
        """
        Processes and pushes PRB blacklist data to InfluxDB.

        Parameters:
            prb_list (list): List of the PRB blacklisted.
        """
        write_api = self.client.write_api()
        point = Point("prb_list").field("blacklist", prb_list)
        write_api.write(bucket=self.bucket, org=self.org, record=point)
        logging.info(f"PRB list pushed to InfluxDB: {prb_list}")

    def close_connection(self):
        """
        Ensures that the connection to InfluxDB is properly closed.
        """
        self.client.close()
        logging.info("InfluxDB connection closed.")

# Example usage:
if __name__ == "__main__":
    plotter = GrafanaPlotter(config_path="influx_config.ini")
    plotter.initialize_plot((1024, 1024))
    plotter.process_iq_data([1+2j, 3+4j, 5+6j])
    plotter.process_prb_list([5, 8, 10])