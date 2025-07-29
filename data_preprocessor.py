import numpy as np
from scipy import signal
import logging
import json
from typing import List, Tuple
from dataclasses import dataclass
from enum import Enum
from abc import ABC, abstractmethod

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define constants and configuration
@dataclass
class Config:
    sampling_rate: int = 1000
    filter_cutoff: float = 10.0
    filter_order: int = 5
    spike_threshold: float = 5.0

class PreprocessingError(Exception):
    """Base class for preprocessing errors"""
    pass

class InvalidSampleError(PreprocessingError):
    """Raised when an invalid sample is encountered"""
    pass

class SpikeRemovalError(PreprocessingError):
    """Raised when spike removal fails"""
    pass

class DataPreprocessor(ABC):
    """Abstract base class for data preprocessors"""
    def __init__(self, config: Config):
        self.config = config

    @abstractmethod
    def remove_invalid_samples(self, data: np.ndarray) -> np.ndarray:
        """Remove invalid samples from the data"""
        pass

    @abstractmethod
    def median_filtering(self, data: np.ndarray) -> np.ndarray:
        """Apply median filtering to the data"""
        pass

    @abstractmethod
    def smooth_data(self, data: np.ndarray) -> np.ndarray:
        """Smooth the data using a low-pass filter"""
        pass

class InvalidSampleRemover(DataPreprocessor):
    """Removes invalid samples from the data"""
    def remove_invalid_samples(self, data: np.ndarray) -> np.ndarray:
        """Remove samples with values outside the valid range"""
        valid_range = (0, 1)
        valid_samples = np.logical_and(data >= valid_range[0], data <= valid_range[1])
        return data[valid_samples]

class MedianFilter(DataPreprocessor):
    """Applies median filtering to the data"""
    def median_filtering(self, data: np.ndarray) -> np.ndarray:
        """Apply median filtering to the data"""
        return signal.medfilt(data, kernel_size=3)

class LowPassFilter(DataPreprocessor):
    """Smooths the data using a low-pass filter"""
    def smooth_data(self, data: np.ndarray) -> np.ndarray:
        """Smooth the data using a low-pass filter"""
        nyq = 0.5 * self.config.sampling_rate
        cutoff = self.config.filter_cutoff / nyq
        b, a = signal.butter(self.config.filter_order, cutoff, btype='low')
        return signal.filtfilt(b, a, data)

class SpikeRemover(DataPreprocessor):
    """Removes spikes from the data"""
    def remove_spikes(self, data: np.ndarray) -> np.ndarray:
        """Remove spikes from the data"""
        threshold = self.config.spike_threshold
        return np.where(np.abs(data) > threshold, 0, data)

class DataPreprocessorFactory:
    """Factory class for creating data preprocessors"""
    @staticmethod
    def create_preprocessor(config: Config) -> DataPreprocessor:
        """Create a data preprocessor based on the configuration"""
        preprocessor = InvalidSampleRemover(config)
        preprocessor = MedianFilter(config)
        preprocessor = LowPassFilter(config)
        preprocessor = SpikeRemover(config)
        return preprocessor

def main():
    # Load configuration from JSON file
    with open('config.json') as f:
        config = Config(**json.load(f))

    # Create a data preprocessor factory
    factory = DataPreprocessorFactory()

    # Create a data preprocessor
    preprocessor = factory.create_preprocessor(config)

    # Generate some sample data
    np.random.seed(0)
    data = np.random.rand(1000)

    # Preprocess the data
    preprocessed_data = preprocessor.remove_invalid_samples(data)
    preprocessed_data = preprocessor.median_filtering(preprocessed_data)
    preprocessed_data = preprocessor.smooth_data(preprocessed_data)
    preprocessed_data = preprocessor.remove_spikes(preprocessed_data)

    # Log the preprocessed data
    logger.info('Preprocessed data: %s', preprocessed_data)

if __name__ == '__main__':
    main()