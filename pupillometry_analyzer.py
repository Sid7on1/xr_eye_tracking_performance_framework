import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
import logging
import json
import os
from typing import List, Dict, Tuple
from enum import Enum
from dataclasses import dataclass
from scipy.stats import norm

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Constants and configuration
CONFIG_FILE = 'pupillometry_analyzer_config.json'
DEFAULT_CONFIG = {
    'pupil_dilation_threshold': 0.1,
    'cognitive_load_threshold': 0.5,
    'filter_order': 4,
    'filter_cutoff': 0.1,
    'plotting_enabled': True
}

@dataclass
class PupilData:
    """Data class for pupil diameter data"""
    timestamp: float
    diameter: float

class CognitiveLoad(Enum):
    """Enum for cognitive load levels"""
    LOW = 1
    MEDIUM = 2
    HIGH = 3

class PupillometryAnalyzer:
    """Main class for pupillometry analysis"""
    def __init__(self, config_file: str = CONFIG_FILE):
        self.config = self.load_config(config_file)
        self.pupil_data = []
        self.filtered_data = []

    def load_config(self, config_file: str) -> Dict:
        """Load configuration from file"""
        try:
            with open(config_file, 'r') as f:
                config = json.load(f)
                return {**DEFAULT_CONFIG, **config}
        except FileNotFoundError:
            logger.warning(f'Config file not found: {config_file}')
            return DEFAULT_CONFIG

    def analyze_pupil_dilation(self, pupil_data: List[PupilData]) -> float:
        """Analyze pupil dilation and return the average dilation factor"""
        if not pupil_data:
            logger.error('No pupil data available')
            return 0.0

        # Calculate dilation factor
        dilation_factors = [data.diameter for data in pupil_data]
        avg_dilation_factor = np.mean(dilation_factors)

        # Filter data using a Butterworth filter
        self.filtered_data = self.filter_data(pupil_data)

        # Calculate the average dilation factor of the filtered data
        avg_filtered_dilation_factor = np.mean([data.diameter for data in self.filtered_data])

        return avg_filtered_dilation_factor

    def calculate_cognitive_load(self, dilation_factor: float) -> CognitiveLoad:
        """Calculate cognitive load based on the dilation factor"""
        if dilation_factor < self.config['pupil_dilation_threshold']:
            return CognitiveLoad.LOW
        elif dilation_factor < self.config['pupil_dilation_threshold'] + self.config['cognitive_load_threshold']:
            return CognitiveLoad.MEDIUM
        else:
            return CognitiveLoad.HIGH

    def filter_data(self, pupil_data: List[PupilData]) -> List[PupilData]:
        """Filter pupil data using a Butterworth filter"""
        if not pupil_data:
            logger.error('No pupil data available')
            return []

        # Create a time array
        time_array = np.array([data.timestamp for data in pupil_data])

        # Create a signal array
        signal_array = np.array([data.diameter for data in pupil_data])

        # Design a Butterworth filter
        nyq = 0.5 / (time_array[-1] - time_array[0])
        b, a = signal.butter(self.config['filter_order'], self.config['filter_cutoff'] / nyq, btype='low')

        # Filter the signal
        filtered_signal = signal.filtfilt(b, a, signal_array)

        # Create a new list of PupilData objects with the filtered data
        filtered_data = [PupilData(data.timestamp, filtered_signal[i]) for i, data in enumerate(pupil_data)]

        return filtered_data

    def plot_data(self, pupil_data: List[PupilData], filtered_data: List[PupilData]) -> None:
        """Plot the pupil data and filtered data"""
        if not pupil_data:
            logger.error('No pupil data available')
            return

        # Create a time array
        time_array = np.array([data.timestamp for data in pupil_data])

        # Create a signal array
        signal_array = np.array([data.diameter for data in pupil_data])

        # Create a filtered signal array
        filtered_signal_array = np.array([data.diameter for data in filtered_data])

        # Plot the data
        plt.plot(time_array, signal_array, label='Original Data')
        plt.plot(time_array, filtered_signal_array, label='Filtered Data')
        plt.legend()
        plt.show()

def main() -> None:
    # Create a PupillometryAnalyzer object
    analyzer = PupillometryAnalyzer()

    # Load pupil data from file
    try:
        with open('pupil_data.json', 'r') as f:
            pupil_data = json.load(f)
            pupil_data = [PupilData(data['timestamp'], data['diameter']) for data in pupil_data]
    except FileNotFoundError:
        logger.error('Pupil data file not found')
        return

    # Analyze pupil dilation
    avg_dilation_factor = analyzer.analyze_pupil_dilation(pupil_data)

    # Calculate cognitive load
    cognitive_load = analyzer.calculate_cognitive_load(avg_dilation_factor)

    # Print the results
    logger.info(f'Average dilation factor: {avg_dilation_factor}')
    logger.info(f'Cognitive load: {cognitive_load.name}')

    # Plot the data
    if analyzer.config['plotting_enabled']:
        analyzer.plot_data(pupil_data, analyzer.filtered_data)

if __name__ == '__main__':
    main()