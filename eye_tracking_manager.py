import logging
import threading
import numpy as np
from typing import List, Tuple
from sranipal_api import SRanipalEye
from enum import Enum
from dataclasses import dataclass
from abc import ABC, abstractmethod

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class EyeTrackingStatus(Enum):
    """Enum for eye tracking status"""
    CALIBRATED = 1
    ACQUIRING_DATA = 2
    BUFFER_MANAGEMENT = 3
    ERROR = 4

@dataclass
class EyeTrackingData:
    """Data class for eye tracking data"""
    gaze_position: Tuple[float, float]
    pupil_size: float
    velocity: float

class EyeTrackerCalibrationException(Exception):
    """Exception for eye tracker calibration errors"""
    pass

class EyeTrackerAcquisitionException(Exception):
    """Exception for eye tracker data acquisition errors"""
    pass

class EyeTrackerBufferManagementException(Exception):
    """Exception for eye tracker buffer management errors"""
    pass

class EyeTracker(ABC):
    """Abstract base class for eye trackers"""
    @abstractmethod
    def calibrate(self) -> None:
        """Calibrate the eye tracker"""
        pass

    @abstractmethod
    def acquire_data(self) -> EyeTrackingData:
        """Acquire eye tracking data"""
        pass

class SRanipalEyeTracker(EyeTracker):
    """Concrete class for SRanipal eye tracker"""
    def __init__(self) -> None:
        self.eye_tracker = SRanipalEye()

    def calibrate(self) -> None:
        """Calibrate the SRanipal eye tracker"""
        try:
            self.eye_tracker.calibrate()
        except Exception as e:
            logging.error(f"Error calibrating eye tracker: {e}")
            raise EyeTrackerCalibrationException("Error calibrating eye tracker")

    def acquire_data(self) -> EyeTrackingData:
        """Acquire eye tracking data from SRanipal eye tracker"""
        try:
            gaze_position = self.eye_tracker.get_gaze_position()
            pupil_size = self.eye_tracker.get_pupil_size()
            velocity = self.eye_tracker.get_velocity()
            return EyeTrackingData(gaze_position, pupil_size, velocity)
        except Exception as e:
            logging.error(f"Error acquiring eye tracking data: {e}")
            raise EyeTrackerAcquisitionException("Error acquiring eye tracking data")

class EyeTrackingManager:
    """Main class for eye tracking data acquisition and management"""
    def __init__(self, eye_tracker: EyeTracker) -> None:
        self.eye_tracker = eye_tracker
        self.status = EyeTrackingStatus.ERROR
        self.buffer = []
        self.lock = threading.Lock()

    def calibrate_eye_tracker(self) -> None:
        """Calibrate the eye tracker"""
        try:
            self.eye_tracker.calibrate()
            self.status = EyeTrackingStatus.CALIBRATED
            logging.info("Eye tracker calibrated successfully")
        except EyeTrackerCalibrationException as e:
            logging.error(f"Error calibrating eye tracker: {e}")
            self.status = EyeTrackingStatus.ERROR

    def acquire_real_time_data(self) -> None:
        """Acquire real-time eye tracking data"""
        try:
            data = self.eye_tracker.acquire_data()
            with self.lock:
                self.buffer.append(data)
            self.status = EyeTrackingStatus.ACQUIRING_DATA
            logging.info("Acquiring real-time eye tracking data")
        except EyeTrackerAcquisitionException as e:
            logging.error(f"Error acquiring eye tracking data: {e}")
            self.status = EyeTrackingStatus.ERROR

    def buffer_management(self) -> None:
        """Manage the eye tracking data buffer"""
        try:
            with self.lock:
                if len(self.buffer) > 100:
                    self.buffer.pop(0)
            self.status = EyeTrackingStatus.BUFFER_MANAGEMENT
            logging.info("Managing eye tracking data buffer")
        except Exception as e:
            logging.error(f"Error managing eye tracking data buffer: {e}")
            self.status = EyeTrackingStatus.ERROR

    def get_buffer(self) -> List[EyeTrackingData]:
        """Get the eye tracking data buffer"""
        with self.lock:
            return self.buffer.copy()

def main() -> None:
    eye_tracker = SRanipalEyeTracker()
    eye_tracking_manager = EyeTrackingManager(eye_tracker)
    eye_tracking_manager.calibrate_eye_tracker()
    threading.Thread(target=eye_tracking_manager.acquire_real_time_data).start()
    threading.Thread(target=eye_tracking_manager.buffer_management).start()

if __name__ == "__main__":
    main()