import threading
import json
import pandas as pd
import logging
from typing import Dict, List
from datetime import datetime

# Define constants
DATA_LOGGING_INTERVAL = 0.1  # seconds
SESSION_DATA_FILE = "session_data.json"
ANALYSIS_DATA_FILE = "analysis_data.csv"

# Define logging configuration
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(threadName)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

class DataLoggerException(Exception):
    """Base exception class for data logger"""
    pass

class InvalidDataException(DataLoggerException):
    """Exception for invalid data"""
    pass

class DataLogger:
    """Multi-threaded data logging system for eye tracking and physiological data"""
    def __init__(self, config: Dict):
        """
        Initialize data logger with configuration

        Args:
        - config (Dict): Configuration dictionary
        """
        self.config = config
        self.eye_data = []
        self.physiological_data = []
        self.lock = threading.Lock()
        self.session_data = {}

    def log_eye_data(self, data: Dict):
        """
        Log eye tracking data

        Args:
        - data (Dict): Eye tracking data dictionary

        Raises:
        - InvalidDataException: If data is invalid
        """
        if not isinstance(data, dict):
            raise InvalidDataException("Invalid data type")
        with self.lock:
            self.eye_data.append(data)
            logger.info(f"Logged eye data: {data}")

    def log_physiological_data(self, data: Dict):
        """
        Log physiological data

        Args:
        - data (Dict): Physiological data dictionary

        Raises:
        - InvalidDataException: If data is invalid
        """
        if not isinstance(data, dict):
            raise InvalidDataException("Invalid data type")
        with self.lock:
            self.physiological_data.append(data)
            logger.info(f"Logged physiological data: {data}")

    def save_session_data(self):
        """
        Save session data to file
        """
        with self.lock:
            self.session_data["eye_data"] = self.eye_data
            self.session_data["physiological_data"] = self.physiological_data
            with open(SESSION_DATA_FILE, "w") as f:
                json.dump(self.session_data, f)
            logger.info(f"Saved session data to {SESSION_DATA_FILE}")

    def export_analysis_data(self):
        """
        Export analysis data to CSV file
        """
        with self.lock:
            analysis_data = []
            for eye_data in self.eye_data:
                analysis_data.append({
                    "timestamp": eye_data["timestamp"],
                    "x": eye_data["x"],
                    "y": eye_data["y"],
                    "velocity": eye_data["velocity"]
                })
            for physiological_data in self.physiological_data:
                analysis_data.append({
                    "timestamp": physiological_data["timestamp"],
                    "heart_rate": physiological_data["heart_rate"],
                    "skin_conductance": physiological_data["skin_conductance"]
                })
            df = pd.DataFrame(analysis_data)
            df.to_csv(ANALYSIS_DATA_FILE, index=False)
            logger.info(f"Exported analysis data to {ANALYSIS_DATA_FILE}")

    def start_logging(self):
        """
        Start data logging thread
        """
        logging_thread = threading.Thread(target=self.log_data)
        logging_thread.start()

    def log_data(self):
        """
        Log data at regular intervals
        """
        while True:
            with self.lock:
                if self.eye_data:
                    logger.info(f"Logging eye data: {self.eye_data[-1]}")
                if self.physiological_data:
                    logger.info(f"Logging physiological data: {self.physiological_data[-1]}")
            threading.sleep(DATA_LOGGING_INTERVAL)

class EyeData:
    """Eye tracking data model"""
    def __init__(self, timestamp: float, x: float, y: float, velocity: float):
        """
        Initialize eye data

        Args:
        - timestamp (float): Timestamp
        - x (float): X coordinate
        - y (float): Y coordinate
        - velocity (float): Velocity
        """
        self.timestamp = timestamp
        self.x = x
        self.y = y
        self.velocity = velocity

    def to_dict(self) -> Dict:
        """
        Convert to dictionary

        Returns:
        - Dict: Dictionary representation
        """
        return {
            "timestamp": self.timestamp,
            "x": self.x,
            "y": self.y,
            "velocity": self.velocity
        }

class PhysiologicalData:
    """Physiological data model"""
    def __init__(self, timestamp: float, heart_rate: float, skin_conductance: float):
        """
        Initialize physiological data

        Args:
        - timestamp (float): Timestamp
        - heart_rate (float): Heart rate
        - skin_conductance (float): Skin conductance
        """
        self.timestamp = timestamp
        self.heart_rate = heart_rate
        self.skin_conductance = skin_conductance

    def to_dict(self) -> Dict:
        """
        Convert to dictionary

        Returns:
        - Dict: Dictionary representation
        """
        return {
            "timestamp": self.timestamp,
            "heart_rate": self.heart_rate,
            "skin_conductance": self.skin_conductance
        }

def main():
    # Create data logger
    data_logger = DataLogger({
        "logging_interval": DATA_LOGGING_INTERVAL
    })

    # Create eye data
    eye_data = EyeData(datetime.now().timestamp(), 10.0, 20.0, 5.0)

    # Create physiological data
    physiological_data = PhysiologicalData(datetime.now().timestamp(), 60.0, 10.0)

    # Log eye data
    data_logger.log_eye_data(eye_data.to_dict())

    # Log physiological data
    data_logger.log_physiological_data(physiological_data.to_dict())

    # Save session data
    data_logger.save_session_data()

    # Export analysis data
    data_logger.export_analysis_data()

    # Start logging thread
    data_logger.start_logging()

if __name__ == "__main__":
    main()