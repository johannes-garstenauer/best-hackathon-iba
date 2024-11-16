import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from datetime import timedelta
from model import generate_binary_classifications
import os
import customtkinter as ctk
import pygame
from matplotlib.patches import Patch

class DataDisplay:
    def __init__(self, data_file):
        self.data = self.load_data(data_file)
        self.classifications = generate_binary_classifications(data_file)
        self.display_window = None
        self.visualization_window = None
        
        # Play sound when initialized
        try:
            pygame.mixer.init()
            sound_path = os.path.join(os.path.dirname(__file__), "assets", "Charlotte_ProcessingFinished.mp3")
            sound = pygame.mixer.Sound(sound_path)
            sound.play()
        except Exception as e:
            print(f"Could not play sound: {e}")
        
        self.show_summary_window()

    def load_data(self, file_path):
        """Load data from CSV or NPY file"""
        try:
            if file_path.endswith('.csv'):
                data = pd.read_csv(file_path)
                numeric_cols = data.select_dtypes(include=[np.number]).columns
                return data[numeric_cols]
            elif file_path.endswith('.npy'):
                return np.load(file_path)
            else:
                raise ValueError("Unsupported file format")
        except Exception as e:
            raise ValueError(f"Error loading data: {str(e)}")

    def format_time(self, seconds):
        """Format seconds to HH:MM:SS"""
        return str(timedelta(seconds=int(seconds)))

    def show_summary_window(self):
        """Display summary window with statistics"""
        if self.display_window is not None:
            self.display_window.destroy()

        self.display_window = ctk.CTkToplevel()
        self.display_window.title("Data Analysis Summary")
        
        # Create main frame
        main_frame = ctk.CTkFrame(self.display_window)
        main_frame.pack(fill="both", expand=True, padx=20, pady=20)

        # Calculate statistics
        total_points = len(self.classifications)
        anomaly_points = np.sum(self.classifications == 1)
        normal_points = np.sum(self.classifications == 0)
        anomaly_percentage = (anomaly_points / total_points) * 100
        normal_percentage = (normal_points / total_points) * 100

        # Statistics labels
        stats_text = f"""
        Total Points: {total_points}
        
        Anomaly Points: {anomaly_points}
        Anomaly Percentage: {anomaly_percentage:.2f}%
        
        Normal Points: {normal_points}
        Normal Percentage: {normal_percentage:.2f}%
        """
        
        stats_label = ctk.CTkLabel(
            main_frame,
            text=stats_text,
            justify="left",
            font=("Arial", 14)
        )
        stats_label.pack(pady=20)

        # Buttons frame
        button_frame = ctk.CTkFrame(main_frame)
        button_frame.pack(fill="x", pady=20)

        # Show visualization button
        viz_button = ctk.CTkButton(
            button_frame,
            text="Show Visualization",
            command=self.show_visualization,
            height=40,
            font=("Arial", 14)
        )
        viz_button.pack(pady=10)

        # Close button
        close_button = ctk.CTkButton(
            button_frame,
            text="Close",
            command=self.close_display,
            height=40,
            font=("Arial", 14)
        )
        close_button.pack(pady=10)

        # Update window size to fit content
        self.display_window.update_idletasks()
        width = main_frame.winfo_reqwidth() + 40
        height = main_frame.winfo_reqheight() + 40
        self.display_window.geometry(f"{width}x{height}")
        
        # Center window
        x = (self.display_window.winfo_screenwidth() // 2) - (width // 2)
        y = (self.display_window.winfo_screenheight() // 2) - (height // 2)
        self.display_window.geometry(f"+{x}+{y}")

    def get_anomaly_periods(self):
        """Get start and end times of anomaly periods"""
        anomaly_periods = []
        start_idx = None
        
        for i in range(len(self.classifications)):
            # Start of anomaly period
            if self.classifications[i] == 1 and (i == 0 or self.classifications[i-1] == 0):
                start_idx = i
            # End of anomaly period
            elif self.classifications[i] == 0 and i > 0 and self.classifications[i-1] == 1 and start_idx is not None:
                anomaly_periods.append((start_idx, i-1))
                start_idx = None
                
        # Handle case where anomaly extends to the end
        if start_idx is not None:
            anomaly_periods.append((start_idx, len(self.classifications)-1))
            
        return anomaly_periods

    def show_visualization(self):
        """Display time series visualization"""
        if self.visualization_window is not None:
            self.visualization_window.destroy()

        # Play sound when visualization is shown
        try:
            sound_path = os.path.join(os.path.dirname(__file__), "assets", "Charlotte_Graph.mp3")
            sound = pygame.mixer.Sound(sound_path)
            sound.play()
        except Exception as e:
            print(f"Could not play sound: {e}")

        self.visualization_window = ctk.CTkToplevel()
        self.visualization_window.title("Time Series Visualization")

        # Create main frame
        main_frame = ctk.CTkFrame(self.visualization_window)
        main_frame.pack(fill="both", expand=True, padx=20, pady=20)

        # Create figure
        fig = Figure(figsize=(12, 4), dpi=100)
        ax = fig.add_subplot(111)

        # Get data values
        if isinstance(self.data, pd.DataFrame):
            values = self.data.mean(axis=1).values
        else:
            values = np.mean(self.data, axis=1) if self.data.ndim > 1 else self.data

        time_points = np.arange(len(values))
        
        # Draw base timeline
        ax.hlines(y=1, xmin=0, xmax=len(values), color='gray', linewidth=2)
        
        # Find continuous segments of normal and anomaly points
        current_class = self.classifications[0]
        start_idx = 0
        
        for i in range(1, len(self.classifications)):
            if self.classifications[i] != current_class or i == len(self.classifications) - 1:
                # Draw the segment
                color = 'red' if current_class == 1 else 'green'
                end_idx = i if self.classifications[i] != current_class else i + 1
                ax.hlines(y=1, xmin=start_idx, xmax=end_idx, 
                         colors=color, linewidth=5)
                
                # Start new segment
                start_idx = i
                current_class = self.classifications[i]

        # Configure axes
        ax.set_xlabel('Time (HH:MM:SS)')
        ax.set_title('Timeline of Normal and Anomaly Periods')

        # Format x-axis ticks
        tick_positions = np.linspace(0, len(values)-1, 10)
        ax.set_xticks(tick_positions)
        ax.set_xticklabels([self.format_time(t) for t in tick_positions])
        
        # Remove y-axis ticks
        ax.set_yticks([])

        # Add legend
        legend_elements = [
            Patch(facecolor='green', label='Normal (0)', alpha=0.8),
            Patch(facecolor='red', label='Anomaly (1)', alpha=0.8)
        ]
        ax.legend(handles=legend_elements, loc='upper right')

        # Add grid
        ax.grid(axis="x", linestyle="--", alpha=0.7)

        # Create canvas
        canvas = FigureCanvasTkAgg(fig, master=main_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True, pady=(0, 20))

        # Create button frame
        button_frame = ctk.CTkFrame(main_frame)
        button_frame.pack(fill="x", pady=10)

        # Close button
        close_button = ctk.CTkButton(
            button_frame,
            text="Close",
            command=self.visualization_window.destroy,
            height=40,
            font=("Arial", 14)
        )
        close_button.pack(pady=10)

        # Update window size to fit content
        self.visualization_window.update_idletasks()
        width = main_frame.winfo_reqwidth() + 40
        height = main_frame.winfo_reqheight() + 40
        self.visualization_window.geometry(f"{width}x{height}")
        
        # Center window
        x = (self.visualization_window.winfo_screenwidth() // 2) - (width // 2)
        y = (self.visualization_window.winfo_screenheight() // 2) - (height // 2)
        self.visualization_window.geometry(f"+{x}+{y}")

    def close_display(self):
        """Close all windows"""
        if self.visualization_window:
            self.visualization_window.destroy()
        if self.display_window:
            self.display_window.destroy()

def display_data_visualization(file_path):
    """Helper function to create and show the visualization"""
    try:
        return DataDisplay(file_path)
    except Exception as e:
        print(f"Error displaying data: {str(e)}")
        return None
