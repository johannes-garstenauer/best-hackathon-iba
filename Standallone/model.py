import numpy as np
import pandas as pd

def generate_binary_classifications(file_path):
    try:
        # Check file extension and get data length
        if file_path.endswith('.csv'):
            data = pd.read_csv(file_path)
            num_points = len(data)
        elif file_path.endswith('.npy'):
            data = np.load(file_path)
            num_points = len(data)
        else:
            print(f"Unsupported file format: {file_path}")
            return None
            
        # Initialize array with zeros
        classifications = np.zeros(num_points, dtype=int)
        
        # Calculate number of initial anomalies (10% of total)
        num_initial_anomalies = int(0.1 * num_points)
        
        # Generate random positions for initial anomalies
        # Ensure positions are at least 4 steps apart and not in the last 3 positions
        valid_positions = []
        current_pos = 0
        
        while current_pos < (num_points - 3):
            valid_positions.append(current_pos)
            current_pos += 4
        
        if len(valid_positions) >= num_initial_anomalies:
            anomaly_positions = np.random.choice(
                valid_positions,
                size=num_initial_anomalies,
                replace=False
            )
            
            # Set anomalies and their following 3 positions
            for pos in anomaly_positions:
                classifications[pos:pos + 4] = 1
            
            print(f"Generated classifications shape: {classifications.shape}")
            print(f"Number of initial anomalies: {num_initial_anomalies}")
            print(f"Total anomaly points: {np.sum(classifications == 1)}")
            print(f"Percentage of initial anomalies: {(num_initial_anomalies / num_points) * 100:.1f}%")
            print(f"Total percentage with following points: {(np.sum(classifications == 1) / num_points) * 100:.1f}%")
            
            return classifications
        else:
            print("Error: Not enough valid positions for required anomalies")
            return None
        
    except Exception as e:
        print(f"Error processing file: {str(e)}")
        return None
