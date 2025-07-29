import numpy as np
import pandas as pd
import logging
from typing import List, Dict, Tuple
from enum import Enum
from dataclasses import dataclass
from threading import Lock

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Constants
VELOCITY_THRESHOLD = 50  # pixels per second
FLOW_THEORY_THRESHOLD = 0.5  # seconds
FIXATION_DURATION_THRESHOLD = 200  # milliseconds
DWELL_TIME_THRESHOLD = 500  # milliseconds

# Data structures
@dataclass
class EyeTrackingData:
    """Data structure for eye tracking data"""
    x: List[float]
    y: List[float]
    timestamp: List[float]

@dataclass
class Metrics:
    """Data structure for calculated metrics"""
    fixation_duration: float
    saccade_velocity: float
    dwell_time: float

class MetricsCalculator:
    """Class for calculating eye tracking metrics"""
    def __init__(self, velocity_threshold: float = VELOCITY_THRESHOLD, flow_theory_threshold: float = FLOW_THEORY_THRESHOLD,
                 fixation_duration_threshold: float = FIXATION_DURATION_THRESHOLD, dwell_time_threshold: float = DWELL_TIME_THRESHOLD):
        """
        Initialize the metrics calculator with thresholds.

        Args:
        - velocity_threshold (float): Velocity threshold for saccade detection (default: 50 pixels per second)
        - flow_theory_threshold (float): Flow theory threshold for engagement assessment (default: 0.5 seconds)
        - fixation_duration_threshold (float): Fixation duration threshold for fixation detection (default: 200 milliseconds)
        - dwell_time_threshold (float): Dwell time threshold for dwell time calculation (default: 500 milliseconds)
        """
        self.velocity_threshold = velocity_threshold
        self.flow_theory_threshold = flow_theory_threshold
        self.fixation_duration_threshold = fixation_duration_threshold
        self.dwell_time_threshold = dwell_time_threshold
        self.lock = Lock()

    def calculate_fixation_duration(self, eye_tracking_data: EyeTrackingData) -> float:
        """
        Calculate the fixation duration from eye tracking data.

        Args:
        - eye_tracking_data (EyeTrackingData): Eye tracking data

        Returns:
        - float: Fixation duration in milliseconds
        """
        with self.lock:
            try:
                # Calculate fixation duration using the formula from the paper
                fixation_duration = np.mean([eye_tracking_data.timestamp[i+1] - eye_tracking_data.timestamp[i] for i in range(len(eye_tracking_data.timestamp) - 1)])
                return fixation_duration
            except Exception as e:
                logger.error(f"Error calculating fixation duration: {str(e)}")
                return 0.0

    def measure_saccade_velocity(self, eye_tracking_data: EyeTrackingData) -> float:
        """
        Measure the saccade velocity from eye tracking data.

        Args:
        - eye_tracking_data (EyeTrackingData): Eye tracking data

        Returns:
        - float: Saccade velocity in pixels per second
        """
        with self.lock:
            try:
                # Calculate saccade velocity using the formula from the paper
                saccade_velocity = np.mean([np.sqrt((eye_tracking_data.x[i+1] - eye_tracking_data.x[i])**2 + (eye_tracking_data.y[i+1] - eye_tracking_data.y[i])**2) / (eye_tracking_data.timestamp[i+1] - eye_tracking_data.timestamp[i]) for i in range(len(eye_tracking_data.timestamp) - 1)])
                return saccade_velocity
            except Exception as e:
                logger.error(f"Error measuring saccade velocity: {str(e)}")
                return 0.0

    def compute_dwell_time(self, eye_tracking_data: EyeTrackingData) -> float:
        """
        Compute the dwell time from eye tracking data.

        Args:
        - eye_tracking_data (EyeTrackingData): Eye tracking data

        Returns:
        - float: Dwell time in milliseconds
        """
        with self.lock:
            try:
                # Calculate dwell time using the formula from the paper
                dwell_time = np.sum([eye_tracking_data.timestamp[i+1] - eye_tracking_data.timestamp[i] for i in range(len(eye_tracking_data.timestamp) - 1)])
                return dwell_time
            except Exception as e:
                logger.error(f"Error computing dwell time: {str(e)}")
                return 0.0

    def calculate_metrics(self, eye_tracking_data: EyeTrackingData) -> Metrics:
        """
        Calculate all metrics from eye tracking data.

        Args:
        - eye_tracking_data (EyeTrackingData): Eye tracking data

        Returns:
        - Metrics: Calculated metrics
        """
        with self.lock:
            try:
                fixation_duration = self.calculate_fixation_duration(eye_tracking_data)
                saccade_velocity = self.measure_saccade_velocity(eye_tracking_data)
                dwell_time = self.compute_dwell_time(eye_tracking_data)
                return Metrics(fixation_duration, saccade_velocity, dwell_time)
            except Exception as e:
                logger.error(f"Error calculating metrics: {str(e)}")
                return Metrics(0.0, 0.0, 0.0)

class MetricsCalculatorException(Exception):
    """Exception class for metrics calculator"""
    pass

def main():
    # Example usage
    eye_tracking_data = EyeTrackingData([1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [0.0, 0.1, 0.2])
    metrics_calculator = MetricsCalculator()
    metrics = metrics_calculator.calculate_metrics(eye_tracking_data)
    logger.info(f"Fixation duration: {metrics.fixation_duration} ms")
    logger.info(f"Saccade velocity: {metrics.saccade_velocity} pixels/s")
    logger.info(f"Dwell time: {metrics.dwell_time} ms")

if __name__ == "__main__":
    main()