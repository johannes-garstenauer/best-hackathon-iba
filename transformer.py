import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.preprocessing import Normalizer, StandardScaler
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader

BATCH_SIZE = 100
SEQ_LENGTH = 100

# Überprüfen, ob CUDA verfügbar ist
if torch.cuda.is_available():
    device = torch.device("cuda")  # Verwende die GPU
    print("CUDA ist verfügbar. Verwende die GPU.")
else:
    device = torch.device("cpu")  # Fallback auf die CPU
    print("CUDA ist nicht verfügbar. Verwende die CPU.")

class TimeSeriesDataset(Dataset):
    def __init__(self, data, seq_length):
        """
        :param data: Pandas DataFrame mit den Sensordaten
        :param seq_length: Länge der Sequenzen, die für das Training verwendet werden
        """
        self.data = data
        self.seq_length = seq_length

    def __len__(self):
        # Die Länge des Datasets ist die Anzahl der möglichen Sequenzen
        return len(self.data) - self.seq_length

    def __getitem__(self, idx):
        # Erstelle eine Sequenz und das zugehörige Ziel
        x = self.data[idx:idx + self.seq_length]  # Eingabesequenz
        y = self.data[idx + self.seq_length]      # Zielwert (nächster Zeitstempel)
        return torch.FloatTensor(x), torch.FloatTensor(y)
    
class TransformerModel(nn.Module):
    def __init__(self, input_dim, model_dim, num_heads, num_layers):
        super(TransformerModel, self).__init__()
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(model_dim, num_heads),
            num_layers
        )
        self.fc = nn.Linear(model_dim, input_dim)

    def forward(self, x):
        x = self.encoder(x)
        return self.fc(x)
        
X_train = np.load("./data/train_data.npy")
X_test = np.load("./data/test_data.npy")
y_test = np.load("./data/test_labels.npy")

normalizer = Normalizer().fit(X_train)
scaler = StandardScaler().fit(X_train)

X_train = normalizer.transform(X_train)
X_train = scaler.transform(X_train)
X_test = normalizer.transform(X_test)
X_test = scaler.transform(X_test)

dataset = TimeSeriesDataset(X_train, seq_length=SEQ_LENGTH)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)
dataloader_size = len(dataloader)
dataloader_size_100 = dataloader_size // 100

# Training
model = TransformerModel(input_dim=X_train.shape[1], model_dim=X_train.shape[1], num_heads=41, num_layers=8)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
loss_fn = nn.MSELoss()

model.to(device)

for epoch in range(5):
    for index, batch in enumerate(dataloader):
        optimizer.zero_grad()
        predictions = model(batch[0].to(device))
        loss = loss_fn(predictions, batch[1].to(device))  # Vorhersage gegen Eingabe
        loss.backward()
        optimizer.step()
        if index % dataloader_size_100 == 0:
            print(f"Epoche: {epoch}:{index // dataloader_size_100}, Loss:{loss.item()}")

### Anomalieerkennung
##for new_data in new_data_loader:
##    predictions = model(new_data)
##    prediction_error = calculate_error(predictions, new_data)
##    if prediction_error > threshold:
##        mark_as_anomaly(new_data)
