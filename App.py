import tkinter as tk
from tkinter import ttk
import os
from pathlib import Path
from model import generate_binary_classifications
import threading
import time
from Display import display_data_visualization

class PathExplorerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Path Explorer")
        self.root.geometry("600x400")
        
        # Create main frame
        self.main_frame = ttk.Frame(root, padding="10")
        self.main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Path entry
        self.path_label = ttk.Label(self.main_frame, text="Enter Path to CSV/NPY file:")
        self.path_label.grid(row=0, column=0, sticky=tk.W, pady=5)
        
        self.path_var = tk.StringVar()
        self.path_entry = ttk.Entry(self.main_frame, textvariable=self.path_var, width=50)
        self.path_entry.grid(row=0, column=1, columnspan=2, sticky=(tk.W, tk.E), pady=5)
        
        # OS Selection
        self.os_var = tk.StringVar(value="windows")
        self.windows_radio = ttk.Radiobutton(
            self.main_frame, 
            text="Windows", 
            variable=self.os_var, 
            value="windows"
        )
        self.linux_radio = ttk.Radiobutton(
            self.main_frame, 
            text="Linux", 
            variable=self.os_var, 
            value="linux"
        )
        self.windows_radio.grid(row=1, column=0, sticky=tk.W, pady=5)
        self.linux_radio.grid(row=1, column=1, sticky=tk.W, pady=5)
        
        # Process button
        self.process_button = ttk.Button(
            self.main_frame,
            text="Process File",
            command=self.process_file
        )
        self.process_button.grid(row=2, column=0, columnspan=3, pady=10)
        
        # Status frame
        self.status_frame = ttk.LabelFrame(self.main_frame, text="Status", padding="5")
        self.status_frame.grid(row=3, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=5)
        
        # Path status
        self.path_status = ttk.Label(self.status_frame, text="", wraplength=550)
        self.path_status.pack(fill=tk.X, pady=5)
        
        # Result label
        self.result_label = ttk.Label(self.main_frame, text="")
        self.result_label.grid(row=4, column=0, columnspan=3, pady=10)
        
        # Processing notification label
        self.processing_label = ttk.Label(
            self.main_frame, 
            text="", 
            foreground="blue",
            font=("Arial", 10, "italic")
        )
        self.processing_label.grid(row=6, column=0, columnspan=3, pady=5)
        
        # Processing animation dots
        self.processing = False
        self.dot_count = 0
        
        # Store combinations
        self.current_combinations = []
        self.current_file_path = None

    def normalize_path(self, path_string):
        # Convert path according to selected OS
        if self.os_var.get() == "windows":
            return path_string.replace('/', '\\')
        else:
            return path_string.replace('\\', '/')

    def validate_path(self, path_string):
        if not path_string:
            return False, "Please enter a file path"
            
        if not (path_string.endswith('.csv') or path_string.endswith('.npy')):
            return False, "File must be either .csv or .npy format"
            
        path = Path(path_string)
        
        if not path.exists():
            parent_exists = path.parent.exists()
            if parent_exists:
                return False, f"File not found. The directory exists, but the file '{path.name}' is missing."
            else:
                return False, f"Directory '{path.parent}' does not exist. Please check the path."
                
        if not path.is_file():
            return False, "The path exists but is not a file"
            
        return True, "Path is valid"

    def update_processing_animation(self):
        if self.processing:
            self.dot_count = (self.dot_count + 1) % 4
            dots = "." * self.dot_count
            self.processing_label.config(text=f"Processing{dots}")
            self.root.after(500, self.update_processing_animation)

    def start_processing_animation(self):
        self.processing = True
        self.processing_label.config(text="Processing")
        self.update_processing_animation()

    def stop_processing_animation(self):
        self.processing = False
        self.processing_label.config(text="")

    def process_file_thread(self, normalized_path):
        try:
            # Display the visualization directly
            display_data_visualization(normalized_path)
            self.root.after(0, lambda: self.stop_processing_animation())
            
        except Exception as e:
            self.root.after(0, lambda: self.stop_processing_animation())
            self.root.after(0, lambda: self.result_label.config(
                text=f"Error: {str(e)}"
            ))

    def process_file(self):
        path_string = self.path_var.get()
        normalized_path = self.normalize_path(path_string)
        
        # Validate path and show status
        is_valid, message = self.validate_path(normalized_path)
        self.path_status.config(text=message)
        
        if not is_valid:
            self.result_label.config(text="")
            return
        
        # Clear previous result
        self.result_label.config(text="")
        
        # Start processing animation
        self.start_processing_animation()
        
        # Process file in separate thread
        processing_thread = threading.Thread(
            target=self.process_file_thread,
            args=(normalized_path,)
        )
        processing_thread.daemon = True
        processing_thread.start()

def main():
    root = tk.Tk()
    app = PathExplorerApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()
