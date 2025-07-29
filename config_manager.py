import json
import logging
import os
from typing import Dict, Any

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ConfigManager:
    """
    Configuration manager for eye tracking parameters and thresholds.
    """

    def __init__(self, config_file: str = "config.json"):
        """
        Initialize the ConfigManager with a configuration file.

        Args:
            config_file (str): The path to the configuration file. Defaults to "config.json".
        """
        self.config_file = config_file
        self.config = self.load_config()

    def load_config(self) -> Dict[str, Any]:
        """
        Load the configuration from the file.

        Returns:
            Dict[str, Any]: The loaded configuration.
        """
        try:
            with open(self.config_file, "r") as f:
                config = json.load(f)
            logger.info("Loaded configuration from file.")
            return config
        except FileNotFoundError:
            logger.warning("Configuration file not found. Creating a new one.")
            return self.create_default_config()
        except json.JSONDecodeError as e:
            logger.error(f"Failed to load configuration: {e}")
            raise

    def create_default_config(self) -> Dict[str, Any]:
        """
        Create a default configuration.

        Returns:
            Dict[str, Any]: The default configuration.
        """
        default_config = {
            "eye_tracking": {
                "velocity_threshold": 0.5,
                "saccade_detection": {
                    "threshold": 0.2,
                    "min_duration": 100,
                    "max_duration": 500
                },
                "fixation_detection": {
                    "threshold": 0.8,
                    "min_duration": 200,
                    "max_duration": 1000
                }
            }
        }
        self.save_config(default_config)
        return default_config

    def save_config(self, config: Dict[str, Any]) -> None:
        """
        Save the configuration to the file.

        Args:
            config (Dict[str, Any]): The configuration to save.
        """
        try:
            with open(self.config_file, "w") as f:
                json.dump(config, f, indent=4)
            logger.info("Saved configuration to file.")
        except Exception as e:
            logger.error(f"Failed to save configuration: {e}")

    def validate_parameters(self, config: Dict[str, Any]) -> None:
        """
        Validate the configuration parameters.

        Args:
            config (Dict[str, Any]): The configuration to validate.
        """
        required_params = ["eye_tracking"]
        for param in required_params:
            if param not in config:
                raise ValueError(f"Missing required parameter: {param}")
        eye_tracking = config["eye_tracking"]
        required_eye_tracking_params = ["velocity_threshold", "saccade_detection", "fixation_detection"]
        for param in required_eye_tracking_params:
            if param not in eye_tracking:
                raise ValueError(f"Missing required parameter: {param}")
        saccade_detection = eye_tracking["saccade_detection"]
        required_saccade_detection_params = ["threshold", "min_duration", "max_duration"]
        for param in required_saccade_detection_params:
            if param not in saccade_detection:
                raise ValueError(f"Missing required parameter: {param}")
        fixation_detection = eye_tracking["fixation_detection"]
        required_fixation_detection_params = ["threshold", "min_duration", "max_duration"]
        for param in required_fixation_detection_params:
            if param not in fixation_detection:
                raise ValueError(f"Missing required parameter: {param}")
        try:
            velocity_threshold = float(eye_tracking["velocity_threshold"])
            saccade_threshold = float(saccade_detection["threshold"])
            fixation_threshold = float(fixation_detection["threshold"])
            min_duration = int(saccade_detection["min_duration"])
            max_duration = int(saccade_detection["max_duration"])
            min_fixation_duration = int(fixation_detection["min_duration"])
            max_fixation_duration = int(fixation_detection["max_duration"])
            if velocity_threshold < 0 or velocity_threshold > 1:
                raise ValueError("Velocity threshold must be between 0 and 1")
            if saccade_threshold < 0 or saccade_threshold > 1:
                raise ValueError("Saccade threshold must be between 0 and 1")
            if fixation_threshold < 0 or fixation_threshold > 1:
                raise ValueError("Fixation threshold must be between 0 and 1")
            if min_duration < 0 or max_duration < 0:
                raise ValueError("Min and max duration must be non-negative")
            if min_fixation_duration < 0 or max_fixation_duration < 0:
                raise ValueError("Min and max fixation duration must be non-negative")
            if min_duration > max_duration:
                raise ValueError("Min duration must be less than or equal to max duration")
            if min_fixation_duration > max_fixation_duration:
                raise ValueError("Min fixation duration must be less than or equal to max fixation duration")
        except ValueError as e:
            raise ValueError(f"Invalid parameter value: {e}")

    def get_config(self) -> Dict[str, Any]:
        """
        Get the current configuration.

        Returns:
            Dict[str, Any]: The current configuration.
        """
        return self.config

def main():
    config_manager = ConfigManager()
    config = config_manager.get_config()
    print(json.dumps(config, indent=4))

if __name__ == "__main__":
    main()