import customtkinter as ctk
import os
from pathlib import Path
from model import generate_binary_classifications
import threading
import time
from Display import display_data_visualization
import pygame

class PathExplorerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Anomaly Explorer")
        self.root.geometry("800x600")
        
        # Initialize pygame mixer and play startup sound
        try:
            pygame.mixer.init()
            sound_path = os.path.join(os.path.dirname(__file__), "assets", "Charlotte_enterPath.mp3")
            sound = pygame.mixer.Sound(sound_path)
            sound_thread = threading.Thread(
                target=lambda: sound.play(),
                daemon=True
            )
            sound_thread.start()
        except Exception as e:
            print(f"Could not play sound: {e}")
        
        # Set the theme
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")
        
        # Create main frame with more padding
        self.main_frame = ctk.CTkFrame(root)
        self.main_frame.pack(fill="both", expand=True, padx=30, pady=30)
        
        # Create a frame for path entry with colored border
        self.path_entry_frame = ctk.CTkFrame(
            self.main_frame,
            fg_color="#1f538d",  # Dark blue background
            border_width=2,
            border_color="#2f93ff",  # Light blue border
            corner_radius=10
        )
        self.path_entry_frame.pack(fill="x", pady=(10, 20), padx=10)
        
        # Path entry label inside the colored frame
        self.path_label = ctk.CTkLabel(
            self.path_entry_frame, 
            text="Enter Path to CSV/NPY file:",
            font=("Arial", 14),
            text_color="white"
        )
        self.path_label.pack(pady=(10, 5), padx=15, anchor="w")
        
        # Path entry inside the colored frame
        self.path_entry = ctk.CTkEntry(
            self.path_entry_frame,
            width=500,
            height=35,
            placeholder_text="Enter file path...",
            fg_color="white",
            text_color="black",
            placeholder_text_color="gray"
        )
        self.path_entry.pack(pady=(0, 15), padx=15, fill="x")
        
        # OS Selection Frame
        self.os_frame = ctk.CTkFrame(self.main_frame)
        self.os_frame.pack(fill="x", pady=(0, 15))
        
        self.os_var = ctk.StringVar(value="windows")
        self.windows_radio = ctk.CTkRadioButton(
            self.os_frame, 
            text="Windows", 
            variable=self.os_var, 
            value="windows"
        )
        self.windows_radio.pack(side="left", padx=20)
        
        self.linux_radio = ctk.CTkRadioButton(
            self.os_frame, 
            text="Linux", 
            variable=self.os_var, 
            value="linux"
        )
        self.linux_radio.pack(side="left", padx=20)
        
        # Process button
        self.process_button = ctk.CTkButton(
            self.main_frame,
            text="Process File",
            command=self.process_file,
            height=40,
            font=("Arial", 14)
        )
        self.process_button.pack(pady=15)
        
        # Status frame
        self.status_frame = ctk.CTkFrame(self.main_frame)
        self.status_frame.pack(fill="x", pady=15, padx=10)
        
        self.status_label = ctk.CTkLabel(
            self.status_frame,
            text="Status:",
            font=("Arial", 14)
        )
        self.status_label.pack(anchor="w", padx=10, pady=5)
        
        # Path status
        self.path_status = ctk.CTkLabel(
            self.status_frame,
            text="",
            wraplength=550,
            font=("Arial", 12)
        )
        self.path_status.pack(fill="x", padx=10, pady=5)
        
        # Result label
        self.result_label = ctk.CTkLabel(
            self.main_frame,
            text="",
            font=("Arial", 12)
        )
        self.result_label.pack(pady=10)
        
        # Update processing notification label with light blue color
        self.processing_label = ctk.CTkLabel(
            self.main_frame, 
            text="",
            font=("Arial", 12, "italic"),
            text_color="#00BFFF"  # Light blue color
        )
        self.processing_label.pack(pady=10)
        
        # Processing animation dots
        self.processing = False
        self.dot_count = 0
        
        # Store combinations
        self.current_combinations = []
        self.current_file_path = None
        
        # Add more spacing before the separator
        self.separator = ctk.CTkFrame(self.main_frame, height=2)
        self.separator.pack(fill="x", pady=20)
        
        # Move close button to bottom with more padding
        self.close_button = ctk.CTkButton(
            self.main_frame,
            text="Close Application",
            command=self.close_application,
            height=40,
            font=("Arial", 14),
            fg_color="red",
            hover_color="darkred"
        )
        self.close_button.pack(pady=20)
        
        # Store reference to child windows
        self.child_windows = []

    def close_application(self):
        """Properly close the application and all its windows"""
        # Close all child windows
        for window in self.child_windows:
            try:
                window.destroy()
            except:
                pass
                
        # Clean up pygame resources
        try:
            pygame.mixer.quit()
        except:
            pass
            
        # Destroy the main window
        self.root.quit()
        self.root.destroy()
        
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
            self.processing_label.configure(text=f"Processing{dots}")
            self.root.after(500, self.update_processing_animation)

    def start_processing_animation(self):
        self.processing = True
        self.processing_label.configure(text="Processing")
        self.update_processing_animation()

    def stop_processing_animation(self):
        self.processing = False
        self.processing_label.configure(text="")

    def process_file_thread(self, normalized_path):
        try:
            # Create the visualization
            display_data_visualization(normalized_path)
            
            # Keep track of the new windows
            for window in self.root.winfo_children():
                if isinstance(window, ctk.CTkToplevel):
                    self.child_windows.append(window)
                    
            self.root.after(0, lambda: self.stop_processing_animation())
        except Exception as e:
            self.root.after(0, lambda: self.stop_processing_animation())
            self.root.after(0, lambda: self.result_label.configure(
                text=f"Error: {str(e)}"
            ))

    def process_file(self):
        path_string = self.path_entry.get()
        normalized_path = self.normalize_path(path_string)
        
        # Validate path and show status
        is_valid, message = self.validate_path(normalized_path)
        self.path_status.configure(text=message)
        
        if not is_valid:
            # Play error sound when path is invalid
            try:
                error_sound_path = os.path.join(os.path.dirname(__file__), "assets", "Charlotte_wrongPath.mp3")
                error_sound = pygame.mixer.Sound(error_sound_path)
                sound_thread = threading.Thread(
                    target=lambda: error_sound.play(),
                    daemon=True
                )
                sound_thread.start()
            except Exception as e:
                print(f"Could not play error sound: {e}")
            
            self.result_label.configure(text="")
            return
        
        # Play success sound when path is valid and processing starts
        try:
            success_sound_path = os.path.join(os.path.dirname(__file__), "assets", "Charlotte_DataProcessed.mp3")
            success_sound = pygame.mixer.Sound(success_sound_path)
            sound_thread = threading.Thread(
                target=lambda: success_sound.play(),
                daemon=True
            )
            sound_thread.start()
        except Exception as e:
            print(f"Could not play success sound: {e}")
        
        # Clear previous result
        self.result_label.configure(text="")
        
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
    root = ctk.CTk()
    app = PathExplorerApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()
