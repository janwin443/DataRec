import numpy as np
import torch

def generate_full_heatmap(model, disk_path, sector_size=1024):
    model.eval()
    results = []

    with open(disk_path, "rb") as f:
        while True:
            data = f.read(sector_size)
            if not data or len(data) < sector_size: break

            # vorbereiten
            tensor = torch.tensor(list(data), dtype=torch.float32) / 255.0
            tensor = tensor.unsqueeze(0).unsqueeze(0) # (1, 1, 1024)

            # raten
            with torch.no_grad():
                output = model(tensor)
                # softmax macht aus zahlen wahrscheinlichkeiten (0.0 - 1.0)
                probs = torch.softmax(output, dim=1).numpy()[0]

            results.append(probs)

    # als datei speichern
    heatmap_matrix = np.array(results)
    np.save("disk_heatmap.npy", heatmap_matrix)
    print(f"saved heatmap")