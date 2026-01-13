import pandas as pd
import os

def analyze_sector_data(sector_list):
    """
    Verarbeitet eine Liste von Sektor-Dictionaries (für Live-Updates).
    Gibt DataFrame zurück.
    """
    if not sector_list:
        return pd.DataFrame()
    
    df = pd.DataFrame(sector_list)
    return df

def analyze_findings_csv(file_path):
    """
    Wird am Ende des Scans aufgerufen.
    Liest die findings.csv, erstellt Statistiken und speichert einen gefilterten Report.
    """
    if os.path.exists(file_path):
        try:
            df = pd.read_csv(file_path)
            
            # 1. Statistik erstellen
            print(f"--- Analyse gestartet für {len(df)} Sektoren ---")
            
            # Bilder filtern
            images = df[df['label'].str.contains('JPEG|PNG|GIF', na=False, case=False)]
            print(f"Gefundene Bilder: {len(images)}")
            
            # Verschlüsselung erkennen (Entropie > 7.5)
            if 'entropy' in df.columns:
                encrypted = df[df['entropy'] > 7.5]
                print(f"Hohe Entropie (Verschlüsselt/Komprimiert): {len(encrypted)}")
                
                # Speichere eine separate Liste der verschlüsselten Sektoren
                if not encrypted.empty:
                    encrypted.to_csv(file_path.replace("findings.csv", "report_encrypted.csv"), index=False)

            # 2. Finalen Report speichern (nur relevante Funde, kein 'Unknown')
            relevant_data = df[df['label'] != 'Unknown']
            report_path = file_path.replace("findings.csv", "final_report.csv")
            relevant_data.to_csv(report_path, index=False)
            
            print(f"Bericht gespeichert unter: {report_path}")
            return df
            
        except Exception as e:
            print(f"Fehler bei der Pandas-Analyse: {e}")
            return None
    return None