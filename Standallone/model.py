import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pandas as pd

class NNAutoencoderSmall(nn.Module):
    lr = 0.0001
    percentile = 94
    def __init__(self, input_dim, lr=0.0001,percentile = 94):
        super(NNAutoencoderSmall, self).__init__()
        self.lr = lr
        self.percentile = percentile
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 14),
            nn.ReLU(),
            nn.Linear(14, 7),
            nn.ReLU(),
            nn.Linear(7, 3),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(3, 7),
            nn.ReLU(),
            nn.Linear(7, 14),
            nn.ReLU(),
            nn.Linear(14, input_dim),
            nn.Sigmoid()
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

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
                  

        data_tensor = torch.tensor(data, dtype=torch.float32)

        model = NNAutoencoderSmall(data_tensor.shape[1])
        model.load_state_dict(torch.load("../model/autoencoder_f1_0.65.pth", weights_only=False))
        model.eval()
        pred = model(data_tensor).detach().numpy()

        # Calculate reconstruction error
        mse = np.mean((data - pred) ** 2, axis=1)

        # Set a threshold for anomaly detection (e.g., 95th percentile)
        threshold = np.percentile(mse, model.percentile)

        # Identify anomalies
        classifications = mse > threshold
        classifications = [1 if c else 0 for c in classifications.tolist()]
        return np.array(classifications)
        
    except Exception as e:
        print(f"Error processing file: {str(e)}")
        return None
