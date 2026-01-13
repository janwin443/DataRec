import torch
import torch.nn as nn
import numpy
from torch.utils.data import Dataset, DataLoader
import os
import heatmap_gen as hmg

class SektorDataset(Dataset):
    def __init__(self, ordner_pfad):
        self.daten_pfade = []
        self.labels = []

        # durch alle dateien im ordner durchsuchen

        for dateiname in os.listdir(ordner_pfad):
            if dateiname.endswith(".bin"):
                voller_pfad = os.path.join(ordner_pfad, dateiname)
                self.daten_pfade.append(voller_pfad)

                # wenn im name NTFS_MFT steht => MFT gefunden! => Label 1
                if "NTFS_MFT" in dateiname:
                    self.labels.append(1)
                elif "JPEG" in dateiname:
                    self.labels.append(2)
                elif "RIFF" in dateiname:
                    self.labels.append(3)
                else:
                    self.labels.append(0)
    def __len__(self):
        return len(self.daten_pfade)

    def __getitem__(self, idx):
        with open(self.daten_pfade[idx], "rb") as f:
            raw_data = f.read(4096) # SECTOR_SIZE
            byte_daten = list(raw_data[:1024])

            # umwandeln in (0 - 1)
            tensor = torch.tensor(byte_daten, dtype=torch.float32) / 255.0

            return tensor.unsqueeze(0), self.labels[idx]


class detector(nn.Module):
    def __init__(self):
        super().__init__()
        # alle Bytes sind zahlen von 0 - 255. 
        # Conv1d (Lupe) schaut nur auf 1024 Bytes gleichzeitig.
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=32, kernel_size=16, stride=8)
        self.pool = nn.MaxPool1d(2)

        # der detektiv zum schluss:
        self.fc1 = nn.Linear(32 * 63, 64)
        self.fc2 = nn.Linear(64, 4) # output: MFT eintrag, keine MFT!

    def forward(self, x):
        # x ist der input (batch, 1, 1024)
        x = torch.relu(self.conv1(x))
        x = self.pool(x)
        x = torch.flatten(x, 1)  # flach machen
        x = torch.relu(self.fc1(x))
        return self.fc2(x)

    def file_to_tensor(pfad):
        with open(pfad, "rb") as f:
            # datei lesen und DEC draus machen (0 - 255)
            daten = f.read()
            byte_liste = list(daten)

            # in pytorch-tensor umwandeln (readable format)
            # teilen durch 255 damit die zahlen zwischen 0 und 1 liegen
            tensor = torch.tensor(byte_liste, dtype=torch.float32) / 255.0
            return tensor

class training(nn.Module):
    def training(model, data_source, runden, save_path="mft_detektor.pth"):
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        kriterium = torch.nn.CrossEntropyLoss()
        try:
            for r in range(runden):
                for x, y in data_source:
                    prediction = model(x)
                    loss = kriterium(prediction, y)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                print(f"Runde {r} abgeschlossen! Fehler: {loss.item()}")
        except KeyboardInterrupt:
            torch.save(model.state_dict(), save_path)


def export_for_rust(model_path, export_path):
    m = detector()
    m.load_state_dict(torch.load(model_path, weights_only=True))
    m.eval()

    # example input (dummy)
    example_input = torch.rand(1, 1, 1024)

    # tracing (den denkprozess)
    traced_script_module = torch.jit.trace(m, example_input)
    traced_script_module.save(export_path)
    print("--- Export successfully completed ---")
    print(f"File created: {export_path}")



# ----------------------------------- ENDE -----------------------------------------------------------------



if __name__ == "__main__":
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("using NVIDIA GPU (CUDA)")
    elif hasattr(torch, 'xpu') and torch.xpu.is_available():
        device = torch.device("xpu")
        print("using Intel GPU (XPU)")
    else:
        device = torch.device("cpu")
        print("using CPU")
    MODEL_PATH = "mft_detektor.pth"
    if os.path.exists(MODEL_PATH):
        detector().load_state_dict(torch.load(MODEL_PATH, weights_only=True))
        print("loaded old model.")
    
    # daten vorbereiten
    dataset = SektorDataset("../shared/extracted")
    loader = DataLoader(dataset, batch_size=32, shuffle=True)

    # weiter trainieren
    training.training(detector(), loader, 100)
    
    # neues wissen speichern
    torch.save(detector().state_dict(), MODEL_PATH)
    export_for_rust("mft_detektor.pth", "mft_detektor.pt")
    print("Training ended and saved new model")