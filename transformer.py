import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.preprocessing import Normalizer, StandardScaler
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader

BATCH_SIZE = 10
SEQ_LENGTH = 10

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

dataset_train = TimeSeriesDataset(X_train, seq_length=SEQ_LENGTH)
dataloader_train = DataLoader(dataset_train, batch_size=BATCH_SIZE, shuffle=False)
dataloader_train_size = len(dataloader_train)
dataloader_train_size_100 = dataloader_train_size // 100

dataset_test = TimeSeriesDataset(X_test, seq_length=1)
dataloader_test = DataLoader(dataset_test, batch_size=1, shuffle=False)
dataloader_test_size = len(dataloader_train)
dataloader_test_size_100 = dataloader_train_size // 100


loss_fn = nn.MSELoss()


# Training
def train():
    model = TransformerModel(input_dim=X_train.shape[1], model_dim=X_train.shape[1], num_heads=41, num_layers=2)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    model.to(device)

    for epoch in range(3):
        for index, batch in enumerate(dataloader_train):
            optimizer.zero_grad()
            predictions = model(batch[0].to(device)[0,:,:])
            loss = loss_fn(predictions, batch[1].to(device))  # Vorhersage gegen Eingabe
            loss.backward()
            optimizer.step()
            if index % dataloader_train_size_100 == 0:
                print(f"Epoche: {epoch}:{index // dataloader_train_size_100}, Loss:{loss.item()}")
                ##if (index // dataloader_train_size_100 > 5):
                ##    break

        torch.save(model, "./transformer")


def predict():
    model = torch.load("./transformer_1", weights_only=False)
    model.eval()

    loss_array = np.zeros(shape=(len(dataloader_test)), dtype=np.float32)
    anomaly_array = np.zeros(shape=(len(dataloader_test)), dtype=np.int64)

    for index, batch in enumerate(dataloader_test):
        predictions = model(batch[0].to(device)[0,:,:])
        loss = loss_fn(predictions, batch[1].to(device))
        loss_array[index] = loss
        if index > 0:
            quantile_90 = np.percentile(loss_array[:index], 90)
            anomaly_array[index] = 1 if loss > quantile_90 else 0
        else:
            anomaly_array[0] = 0

        if((index+2) % 1000 == 0):
            ##print(np.unique(anomaly_array[:index], return_counts=True))

            y_part_test = y_test[:index]
            y_predict = anomaly_array[:index]
            accuracy = accuracy_score(y_part_test, y_predict)
            f1 = f1_score(y_part_test, y_predict)
            precision = precision_score(y_part_test, y_predict)
            recall = recall_score(y_part_test, y_predict)
            print(
f"""
Accuracy :  {accuracy}
f1       :  {f1}
precision:  {precision}
recall   :  {recall}
""")
            print()


train()
##predict()