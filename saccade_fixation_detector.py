import numpy as np
import math
from scipy import signal
import logging
import json
import os
from typing import List, Dict
from saccade_fixation_detector.config import Config
from saccade_fixation_detector.exceptions import InvalidInputError, ConfigurationError
from saccade_fixation_detector.models import Saccade, Fixation
from saccade_fixation_detector.utils import calculate_mean, calculate_std

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SaccadeFixationDetector:
    """
    Velocity-threshold identification algorithm for detecting saccades and fixations.
    """

    def __init__(self, config: Config):
        """
        Initialize the SaccadeFixationDetector with a configuration object.

        Args:
            config (Config): The configuration object.
        """
        self.config = config
        self.eye_positions = []
        self.eye_velocities = []
        self.saccades = []
        self.fixations = []

    def calculate_angular_velocity(self, eye_positions: List[List[float]]) -> List[float]:
        """
        Calculate the angular velocity of the eye movements.

        Args:
            eye_positions (List[List[float]]): A list of eye positions.

        Returns:
            List[float]: A list of angular velocities.
        """
        try:
            # Calculate the differences in x and y coordinates
            dx = np.diff([pos[0] for pos in eye_positions])
            dy = np.diff([pos[1] for pos in eye_positions])

            # Calculate the angular velocities
            angular_velocities = np.arctan2(dy, dx)

            # Convert the angular velocities to radians per second
            angular_velocities = np.deg2rad(angular_velocities) / self.config.sample_rate

            return angular_velocities.tolist()
        except Exception as e:
            logger.error(f"Error calculating angular velocity: {str(e)}")
            raise

    def detect_saccades(self, angular_velocities: List[float]) -> List[Saccade]:
        """
        Detect saccades based on the angular velocities.

        Args:
            angular_velocities (List[float]): A list of angular velocities.

        Returns:
            List[Saccade]: A list of detected saccades.
        """
        try:
            # Apply a low-pass filter to the angular velocities
            filtered_velocities = signal.lfilter([1], [1, -self.config.low_pass_filter_coefficient], angular_velocities)

            # Calculate the absolute values of the filtered velocities
            absolute_velocities = np.abs(filtered_velocities)

            # Detect saccades based on the absolute velocities
            saccades = []
            for i in range(1, len(absolute_velocities)):
                if absolute_velocities[i] > self.config.saccade_threshold:
                    saccade_start = i - 1
                    saccade_end = i
                    while saccade_end < len(absolute_velocities) and absolute_velocities[saccade_end] > self.config.saccade_threshold:
                        saccade_end += 1
                    saccade = Saccade(saccade_start, saccade_end, absolute_velocities[saccade_start:saccade_end])
                    saccades.append(saccade)

            return saccades
        except Exception as e:
            logger.error(f"Error detecting saccades: {str(e)}")
            raise

    def detect_fixations(self, angular_velocities: List[float]) -> List[Fixation]:
        """
        Detect fixations based on the angular velocities.

        Args:
            angular_velocities (List[float]): A list of angular velocities.

        Returns:
            List[Fixation]: A list of detected fixations.
        """
        try:
            # Apply a low-pass filter to the angular velocities
            filtered_velocities = signal.lfilter([1], [1, -self.config.low_pass_filter_coefficient], angular_velocities)

            # Calculate the absolute values of the filtered velocities
            absolute_velocities = np.abs(filtered_velocities)

            # Detect fixations based on the absolute velocities
            fixations = []
            for i in range(1, len(absolute_velocities)):
                if absolute_velocities[i] < self.config.fixation_threshold:
                    fixation_start = i - 1
                    fixation_end = i
                    while fixation_end < len(absolute_velocities) and absolute_velocities[fixation_end] < self.config.fixation_threshold:
                        fixation_end += 1
                    fixation = Fixation(fixation_start, fixation_end, absolute_velocities[fixation_start:fixation_end])
                    fixations.append(fixation)

            return fixations
        except Exception as e:
            logger.error(f"Error detecting fixations: {str(e)}")
            raise

    def process_eye_positions(self, eye_positions: List[List[float]]) -> Dict[str, List]:
        """
        Process the eye positions and detect saccades and fixations.

        Args:
            eye_positions (List[List[float]]): A list of eye positions.

        Returns:
            Dict[str, List]: A dictionary containing the detected saccades and fixations.
        """
        try:
            # Calculate the angular velocities
            angular_velocities = self.calculate_angular_velocity(eye_positions)

            # Detect saccades
            saccades = self.detect_saccades(angular_velocities)

            # Detect fixations
            fixations = self.detect_fixations(angular_velocities)

            return {"saccades": saccades, "fixations": fixations}
        except Exception as e:
            logger.error(f"Error processing eye positions: {str(e)}")
            raise

class Config:
    """
    Configuration class for the SaccadeFixationDetector.
    """

    def __init__(self, config_file: str):
        """
        Initialize the Config object with a configuration file.

        Args:
            config_file (str): The configuration file.
        """
        try:
            with open(config_file, "r") as f:
                config = json.load(f)
            self.sample_rate = config["sample_rate"]
            self.low_pass_filter_coefficient = config["low_pass_filter_coefficient"]
            self.saccade_threshold = config["saccade_threshold"]
            self.fixation_threshold = config["fixation_threshold"]
        except Exception as e:
            logger.error(f"Error loading configuration: {str(e)}")
            raise

class Saccade:
    """
    Saccade class.
    """

    def __init__(self, start: int, end: int, velocities: List[float]):
        """
        Initialize the Saccade object.

        Args:
            start (int): The start index of the saccade.
            end (int): The end index of the saccade.
            velocities (List[float]): The velocities of the saccade.
        """
        self.start = start
        self.end = end
        self.velocities = velocities

class Fixation:
    """
    Fixation class.
    """

    def __init__(self, start: int, end: int, velocities: List[float]):
        """
        Initialize the Fixation object.

        Args:
            start (int): The start index of the fixation.
            end (int): The end index of the fixation.
            velocities (List[float]): The velocities of the fixation.
        """
        self.start = start
        self.end = end
        self.velocities = velocities

if __name__ == "__main__":
    config_file = "config.json"
    config = Config(config_file)
    detector = SaccadeFixationDetector(config)
    eye_positions = [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0], [9.0, 10.0]]
    result = detector.process_eye_positions(eye_positions)
    print(result)