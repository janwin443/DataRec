import os
import pandas as pd
import base64
from .classifier import classifier # Nutzt dein bestehendes KI-Modell
try:
    from .neural_analyzer import NeuralAnalyzer
    analyzer = NeuralAnalyzer()
except Exception as e:
    print(f"Konnte NeuralAnalyzer nicht laden: {e}")
    analyzer = None

class AIRefinement:
    _last_processed_row = 0

    def __init__(self, shared_path="../shared/"):
        self.csv_path = os.path.join(shared_path, "findings.csv")
        self.extracted_path = os.path.join(shared_path, "extracted/")
        self.output_path = os.path.join(shared_path, "ai_recovered/")
        os.makedirs(self.output_path, exist_ok=True)

    def process_unknown_sectors(self):
        """Analysiert Sektoren, die als 'Unknown' markiert wurden"""
        if not os.path.exists(self.csv_path): return
        
        try:    
            df = pd.read_csv(self.csv_path)
            # Nur Sektoren betrachten, die keine klare Signatur haben
            new_data = df.iloc[AIRefinement._last_processed_row:]

            if new_data.empty: return

            unknowns = new_data[new_data['label'].isin(['Unknown', 'High_Entropy'])]        
            
            for _, row in unknowns.iterrows():
                sector_idx = row['sector']
                file_path = os.path.join(self.extracted_path, f"sector_{sector_idx}.bin")
                
                if os.path.exists(file_path):
                    with open(file_path, "rb") as f:
                        data = f.read()
                        
                    # KI-Klassifizierung auf den kompletten Sektor anwenden
                    label, confidence = classifier.classify_sector(data)
                    
                    if confidence > 90:
                        print(f"[AI] Sektor {sector_idx} als {label} identifiziert ({confidence}%)")
                        self.save_refined_fragment(sector_idx, data, label)
            
            AIRefinement._last_processed_row = len(df)
        
        except Exception as e:
            print(f"Refiner error: {e}")

    def save_refined_fragment(self, idx, data, label):
        target_dir = os.path.join(self.output_path, label)
        os.makedirs(target_dir, exist_ok=True)
        with open(os.path.join(target_dir, f"ai_frag_{idx}.bin"), "wb") as f:
            f.write(data)

if __name__ == "__main__":
    refiner = AIRefinement()
    refiner.process_unknown_sectors()