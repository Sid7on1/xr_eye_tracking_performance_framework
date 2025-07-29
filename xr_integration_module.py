import json
import logging
import threading
from typing import Dict, List, Tuple
from unity_api import UnityAPI
from xr_eye_tracking_performance_framework.config import Config
from xr_eye_tracking_performance_framework.exceptions import (
    EyeTrackingDataException,
    UnityCommunicationException,
)
from xr_eye_tracking_performance_framework.models import EyeTrackingData

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class XREyeTrackingIntegration:
    """
    Class responsible for Unity XR integration for seamless eye tracking data communication.
    """

    def __init__(self, config: Config):
        """
        Initialize the XREyeTrackingIntegration class.

        Args:
        - config (Config): Configuration object containing settings and parameters.
        """
        self.config = config
        self.unity_api = UnityAPI()
        self.eye_tracking_data = None
        self.scene_updates = None
        self.lock = threading.Lock()

    def unity_communication(self) -> None:
        """
        Establish communication with the Unity API.
        """
        try:
            self.unity_api.connect()
            logger.info("Connected to Unity API")
        except Exception as e:
            logger.error(f"Failed to connect to Unity API: {str(e)}")
            raise UnityCommunicationException("Failed to connect to Unity API")

    def send_eye_data(self, eye_tracking_data: EyeTrackingData) -> None:
        """
        Send eye tracking data to the Unity API.

        Args:
        - eye_tracking_data (EyeTrackingData): Eye tracking data to be sent.
        """
        try:
            with self.lock:
                self.eye_tracking_data = eye_tracking_data
                self.unity_api.send_data(json.dumps(eye_tracking_data.to_dict()))
                logger.info("Sent eye tracking data to Unity API")
        except Exception as e:
            logger.error(f"Failed to send eye tracking data: {str(e)}")
            raise EyeTrackingDataException("Failed to send eye tracking data")

    def receive_scene_updates(self) -> Dict:
        """
        Receive scene updates from the Unity API.

        Returns:
        - Dict: Scene updates received from the Unity API.
        """
        try:
            with self.lock:
                self.scene_updates = self.unity_api.receive_data()
                logger.info("Received scene updates from Unity API")
                return json.loads(self.scene_updates)
        except Exception as e:
            logger.error(f"Failed to receive scene updates: {str(e)}")
            raise UnityCommunicationException("Failed to receive scene updates")

    def get_eye_tracking_data(self) -> EyeTrackingData:
        """
        Get the current eye tracking data.

        Returns:
        - EyeTrackingData: Current eye tracking data.
        """
        with self.lock:
            return self.eye_tracking_data

    def get_scene_updates(self) -> Dict:
        """
        Get the current scene updates.

        Returns:
        - Dict: Current scene updates.
        """
        with self.lock:
            return self.scene_updates

    def close_connection(self) -> None:
        """
        Close the connection to the Unity API.
        """
        try:
            self.unity_api.disconnect()
            logger.info("Disconnected from Unity API")
        except Exception as e:
            logger.error(f"Failed to disconnect from Unity API: {str(e)}")
            raise UnityCommunicationException("Failed to disconnect from Unity API")


class EyeTrackingData:
    """
    Class representing eye tracking data.
    """

    def __init__(self, velocity: float, fixation: float, saccade: float):
        """
        Initialize the EyeTrackingData class.

        Args:
        - velocity (float): Velocity of the eye movement.
        - fixation (float): Fixation duration.
        - saccade (float): Saccade duration.
        """
        self.velocity = velocity
        self.fixation = fixation
        self.saccade = saccade

    def to_dict(self) -> Dict:
        """
        Convert the eye tracking data to a dictionary.

        Returns:
        - Dict: Eye tracking data as a dictionary.
        """
        return {"velocity": self.velocity, "fixation": self.fixation, "saccade": self.saccade}


class Config:
    """
    Class representing the configuration.
    """

    def __init__(self, unity_api_url: str, eye_tracking_data_interval: int):
        """
        Initialize the Config class.

        Args:
        - unity_api_url (str): URL of the Unity API.
        - eye_tracking_data_interval (int): Interval at which eye tracking data is sent.
        """
        self.unity_api_url = unity_api_url
        self.eye_tracking_data_interval = eye_tracking_data_interval


def main() -> None:
    """
    Main function.
    """
    config = Config("http://localhost:8080", 1000)
    xr_eye_tracking_integration = XREyeTrackingIntegration(config)
    xr_eye_tracking_integration.unity_communication()

    # Example usage:
    eye_tracking_data = EyeTrackingData(10.0, 5.0, 2.0)
    xr_eye_tracking_integration.send_eye_data(eye_tracking_data)
    scene_updates = xr_eye_tracking_integration.receive_scene_updates()
    print(scene_updates)

    xr_eye_tracking_integration.close_connection()


if __name__ == "__main__":
    main()