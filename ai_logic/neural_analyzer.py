import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os

# Das neuronale Netzwerk: Ein 1D-CNN für Binärdaten
class SectorNet(nn.Module):
    def __init__(self, num_classes=12):
        super(SectorNet, self).__init__()
        # Wir betrachten den Sektor als Signal (1 Kanal, 4096 Bytes lang)
        self.features = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=16, stride=8), # Extrahiert Muster
            nn.ReLU(),
            nn.MaxPool1d(4),
            nn.Conv1d(32, 64, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten()
        )
        self.classifier = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, num_classes)
        )
        self.labels = ["JPEG", "PNG", "PDF", "ZIP", "EXE", "SQLITE", "MFT", "SYSTEM", "TEXT", "ENCRYPTED", "VIDEO", "UNKNOWN"]

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)

class NeuralAnalyzer:
    def __init__(self, model_path="ai-logic/sector_model.pth"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_path = model_path
        self.model = SectorNet().to(self.device)
        
        if os.path.exists(self.model_path):
            self.model.load_state_dict(torch.load(self.model_path, map_location=self.device))
        
        self.model.eval()

    def predict(self, binary_data):
        """Klassifiziert einen 4KB Sektor"""
        # Daten vorbereiten (0-255 -> 0.0-1.0)
        input_data = np.frombuffer(binary_data, dtype=np.uint8).astype(np.float32) / 255.0
        
        # Auf 4096 Bytes normieren
        if len(input_data) != 4096:
            temp = np.zeros(4096, dtype=np.float32)
            temp[:len(input_data)] = input_data[:4096]
            input_data = temp

        # Tensor-Form: [Batch, Channels, Length] -> [1, 1, 4096]
        tensor = torch.from_numpy(input_data).unsqueeze(0).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            output = self.model(tensor)
            prob = torch.softmax(output, dim=1)
            conf, idx = torch.max(prob, 1)
            
        return self.model.labels[idx.item()], conf.item() * 100

    def train_step(self, binary_data, label_name):
        """Ermöglicht das Lernen 'on the fly'"""
        if label_name not in self.model.labels: return
        
        self.model.train()
        optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()
        
        # Daten wie bei Predict vorbereiten...
        # (Hier würde ein kleiner Backpropagation-Step folgen)
        # Für den Anfang reicht uns aber erst mal die Prediction.