import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import tkinter as tk
from tkinter import ttk
import logging
import json
import os
from typing import Dict, List, Tuple
from dataclasses import dataclass
from enum import Enum
from scipy.signal import savgol_filter
from scipy.stats import linregress

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Constants and configuration
CONFIG_FILE = 'visualization_config.json'
DEFAULT_CONFIG = {
    'plot_interval': 100,  # milliseconds
    'saccade_threshold': 10,  # degrees
    'gaze_pattern_window': 100,  # frames
    'metric_update_interval': 1000  # milliseconds
}

@dataclass
class EyeTrackingData:
    """Represents eye tracking data"""
    timestamp: float
    x_gaze: float
    y_gaze: float
    pupil_diameter: float

class VisualizationMode(Enum):
    """Enum for visualization modes"""
    GAZE_PATTERN = 1
    SACCADE = 2
    METRICS = 3

class VisualizationTools:
    """Provides real-time visualization of eye tracking data and user state"""
    def __init__(self, config_file: str = CONFIG_FILE):
        self.config_file = config_file
        self.config = self.load_config()
        self.eye_data = []
        self.gaze_pattern_window = []
        self.saccade_data = []
        self.metrics = {}
        self.plot_interval = self.config['plot_interval']
        self.saccade_threshold = self.config['saccade_threshold']
        self.gaze_pattern_window_size = self.config['gaze_pattern_window']
        self.metric_update_interval = self.config['metric_update_interval']
        self.mode = VisualizationMode.GAZE_PATTERN
        self.root = tk.Tk()
        self.root.title('Eye Tracking Visualization')
        self.plot_frame = tk.Frame(self.root)
        self.plot_frame.pack(fill='both', expand=True)
        self.plot = plt.Figure(figsize=(8, 6))
        self.ax = self.plot.add_subplot(111)
        self.ax.set_xlim(-100, 100)
        self.ax.set_ylim(-100, 100)
        self.ax.set_aspect('equal')
        self.plot_canvas = tk.Canvas(self.root, width=800, height=600)
        self.plot_canvas.pack(fill='both', expand=True)
        self.plot_id = self.plot_canvas.create_window((0, 0), window=self.plot, anchor='nw')
        self.update_plot()
        self.root.after(self.plot_interval, self.update_plot)
        self.root.after(self.metric_update_interval, self.update_metrics)
        self.root.mainloop()

    def load_config(self) -> Dict:
        """Loads configuration from file"""
        if os.path.exists(self.config_file):
            with open(self.config_file, 'r') as f:
                return json.load(f)
        else:
            return DEFAULT_CONFIG

    def update_plot(self):
        """Updates the plot with the latest eye tracking data"""
        if self.mode == VisualizationMode.GAZE_PATTERN:
            self.plot_gaze_patterns()
        elif self.mode == VisualizationMode.SACCADE:
            self.visualize_saccades()
        elif self.mode == VisualizationMode.METRICS:
            self.display_metrics()
        self.plot_canvas.itemconfig(self.plot_id, window=self.plot)
        self.root.after(self.plot_interval, self.update_plot)

    def plot_gaze_patterns(self):
        """Plots the gaze patterns"""
        self.ax.clear()
        self.ax.set_xlim(-100, 100)
        self.ax.set_ylim(-100, 100)
        self.ax.set_aspect('equal')
        if len(self.eye_data) > 0:
            x_gaze = [data.x_gaze for data in self.eye_data]
            y_gaze = [data.y_gaze for data in self.eye_data]
            self.ax.plot(x_gaze, y_gaze, 'b-')
            self.ax.scatter(x_gaze, y_gaze, c='r')
        self.plot_canvas.itemconfig(self.plot_id, window=self.plot)

    def visualize_saccades(self):
        """Visualizes saccades"""
        self.ax.clear()
        self.ax.set_xlim(-100, 100)
        self.ax.set_ylim(-100, 100)
        self.ax.set_aspect('equal')
        if len(self.eye_data) > 0:
            x_gaze = [data.x_gaze for data in self.eye_data]
            y_gaze = [data.y_gaze for data in self.eye_data]
            saccade_data = self.get_saccade_data(x_gaze, y_gaze)
            self.ax.plot(x_gaze, y_gaze, 'b-')
            self.ax.scatter(x_gaze, y_gaze, c='r')
            self.ax.scatter(saccade_data['start_x'], saccade_data['start_y'], c='g')
            self.ax.scatter(saccade_data['end_x'], saccade_data['end_y'], c='g')
        self.plot_canvas.itemconfig(self.plot_id, window=self.plot)

    def get_saccade_data(self, x_gaze: List[float], y_gaze: List[float]) -> Dict:
        """Gets saccade data"""
        saccade_data = {}
        for i in range(1, len(x_gaze)):
            if abs(x_gaze[i] - x_gaze[i-1]) > self.saccade_threshold or abs(y_gaze[i] - y_gaze[i-1]) > self.saccade_threshold:
                saccade_data['start_x'] = x_gaze[i-1]
                saccade_data['start_y'] = y_gaze[i-1]
                saccade_data['end_x'] = x_gaze[i]
                saccade_data['end_y'] = y_gaze[i]
                break
        return saccade_data

    def display_metrics(self):
        """Displays metrics"""
        self.ax.clear()
        self.ax.set_xlim(-100, 100)
        self.ax.set_ylim(-100, 100)
        self.ax.set_aspect('equal')
        if len(self.eye_data) > 0:
            x_gaze = [data.x_gaze for data in self.eye_data]
            y_gaze = [data.y_gaze for data in self.eye_data]
            self.ax.plot(x_gaze, y_gaze, 'b-')
            self.ax.scatter(x_gaze, y_gaze, c='r')
            self.ax.text(0.05, 0.9, f'Gaze Angle: {self.get_gaze_angle(x_gaze, y_gaze)}', transform=self.ax.transAxes)
            self.ax.text(0.05, 0.8, f'Pupil Diameter: {self.get_pupil_diameter(self.eye_data)}', transform=self.ax.transAxes)
        self.plot_canvas.itemconfig(self.plot_id, window=self.plot)

    def get_gaze_angle(self, x_gaze: List[float], y_gaze: List[float]) -> float:
        """Gets gaze angle"""
        if len(x_gaze) > 1:
            slope, intercept, r_value, p_value, std_err = linregress(x_gaze, y_gaze)
            return np.arctan(slope) * 180 / np.pi
        else:
            return 0

    def get_pupil_diameter(self, eye_data: List[EyeTrackingData]) -> float:
        """Gets pupil diameter"""
        if len(eye_data) > 0:
            return eye_data[-1].pupil_diameter
        else:
            return 0

    def update_metrics(self):
        """Updates metrics"""
        if len(self.eye_data) > 0:
            self.metrics['gaze_angle'] = self.get_gaze_angle([data.x_gaze for data in self.eye_data], [data.y_gaze for data in self.eye_data])
            self.metrics['pupil_diameter'] = self.get_pupil_diameter(self.eye_data)
        self.root.after(self.metric_update_interval, self.update_metrics)

    def add_eye_data(self, eye_data: EyeTrackingData):
        """Adds eye tracking data"""
        self.eye_data.append(eye_data)
        if len(self.eye_data) > self.gaze_pattern_window_size:
            self.eye_data.pop(0)
        self.update_plot()

def main():
    config_file = 'visualization_config.json'
    config = {
        'plot_interval': 100,  # milliseconds
        'saccade_threshold': 10,  # degrees
        'gaze_pattern_window': 100,  # frames
        'metric_update_interval': 1000  # milliseconds
    }
    with open(config_file, 'w') as f:
        json.dump(config, f)
    visualization_tools = VisualizationTools(config_file)

if __name__ == '__main__':
    main()