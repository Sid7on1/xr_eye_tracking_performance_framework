import logging
import numpy as np
from typing import Tuple, List
from sranipal_api import SRanipal_Eye
import threading
from enum import Enum
from dataclasses import dataclass
from abc import ABC, abstractmethod

# Constants
CALIBRATION_POINTS = 5
IPD_ADJUSTMENT_THRESHOLD = 5  # mm
CALIBRATION_VALIDATION_THRESHOLD = 10  # pixels

# Logging configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class CalibrationStatus(Enum):
    """Enum for calibration status"""
    NOT_CALIBRATED = 1
    CALIBRATING = 2
    CALIBRATED = 3

@dataclass
class CalibrationPoint:
    """Dataclass for calibration point"""
    x: int
    y: int

class CalibrationException(Exception):
    """Exception class for calibration errors"""
    pass

class CalibrationSystem(ABC):
    """Abstract base class for calibration system"""
    def __init__(self):
        self.calibration_status = CalibrationStatus.NOT_CALIBRATED
        self.calibration_points = []
        self.ipd = 0
        self.lock = threading.Lock()

    @abstractmethod
    def run_calibration(self) -> None:
        """Run calibration process"""
        pass

    @abstractmethod
    def validate_calibration(self) -> bool:
        """Validate calibration"""
        pass

    @abstractmethod
    def adjust_ipd(self, ipd: float) -> None:
        """Adjust IPD"""
        pass

class EyeCalibrationSystem(CalibrationSystem):
    """Concrete class for eye calibration system"""
    def __init__(self, eye: SRanipal_Eye):
        super().__init__()
        self.eye = eye

    def run_calibration(self) -> None:
        """Run calibration process"""
        try:
            logger.info("Starting calibration process")
            self.calibration_status = CalibrationStatus.CALIBRATING
            self.eye.start_calibration()
            for i in range(CALIBRATION_POINTS):
                logger.info(f"Calibrating point {i+1} of {CALIBRATION_POINTS}")
                point = self.eye.get_calibration_point()
                self.calibration_points.append(CalibrationPoint(point[0], point[1]))
                self.eye.move_to_next_calibration_point()
            self.eye.end_calibration()
            self.calibration_status = CalibrationStatus.CALIBRATED
            logger.info("Calibration process completed")
        except Exception as e:
            logger.error(f"Error during calibration: {e}")
            raise CalibrationException("Error during calibration")

    def validate_calibration(self) -> bool:
        """Validate calibration"""
        try:
            logger.info("Validating calibration")
            if len(self.calibration_points) != CALIBRATION_POINTS:
                logger.error("Invalid number of calibration points")
                return False
            for point in self.calibration_points:
                if point.x < 0 or point.x > self.eye.get_screen_width() or point.y < 0 or point.y > self.eye.get_screen_height():
                    logger.error("Calibration point out of bounds")
                    return False
            logger.info("Calibration validated")
            return True
        except Exception as e:
            logger.error(f"Error during validation: {e}")
            raise CalibrationException("Error during validation")

    def adjust_ipd(self, ipd: float) -> None:
        """Adjust IPD"""
        try:
            logger.info(f"Adjusting IPD to {ipd} mm")
            if abs(ipd - self.ipd) > IPD_ADJUSTMENT_THRESHOLD:
                self.ipd = ipd
                self.eye.set_ipd(ipd)
                logger.info("IPD adjusted")
            else:
                logger.info("IPD adjustment not needed")
        except Exception as e:
            logger.error(f"Error during IPD adjustment: {e}")
            raise CalibrationException("Error during IPD adjustment")

class CalibrationConfig:
    """Class for calibration configuration"""
    def __init__(self, calibration_points: List[CalibrationPoint], ipd: float):
        self.calibration_points = calibration_points
        self.ipd = ipd

class CalibrationManager:
    """Class for calibration manager"""
    def __init__(self, eye: SRanipal_Eye):
        self.eye = eye
        self.calibration_system = EyeCalibrationSystem(eye)
        self.config = None

    def start_calibration(self) -> None:
        """Start calibration process"""
        self.calibration_system.run_calibration()

    def validate_calibration(self) -> bool:
        """Validate calibration"""
        return self.calibration_system.validate_calibration()

    def adjust_ipd(self, ipd: float) -> None:
        """Adjust IPD"""
        self.calibration_system.adjust_ipd(ipd)

    def save_config(self) -> None:
        """Save calibration configuration"""
        self.config = CalibrationConfig(self.calibration_system.calibration_points, self.calibration_system.ipd)
        logger.info("Calibration configuration saved")

    def load_config(self, config: CalibrationConfig) -> None:
        """Load calibration configuration"""
        self.calibration_system.calibration_points = config.calibration_points
        self.calibration_system.ipd = config.ipd
        logger.info("Calibration configuration loaded")

def main() -> None:
    eye = SRanipal_Eye()
    calibration_manager = CalibrationManager(eye)
    calibration_manager.start_calibration()
    if calibration_manager.validate_calibration():
        calibration_manager.adjust_ipd(65)
        calibration_manager.save_config()
    else:
        logger.error("Calibration failed")

if __name__ == "__main__":
    main()