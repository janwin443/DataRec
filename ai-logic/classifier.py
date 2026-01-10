import numpy as np
import math
from collections import Counter
import base64

class ForensicClassifier:
    def __init__(self):
        self.labels = ["Encrypted", "Text", "MFT_Entry", "Media", "Zero"]

    def calculate_entropy(self, data):
        if not data: return 0
        entropy = 0
        counter = Counter(data)
        length = len(data)
        for count in counter.values():
            p_x = count / length
            entropy += - p_x * math.log2(p_x)
        return entropy

    def classify_sector(self, b64_data):
        try:
            raw_bytes = base64.b64decode(b64_data)
            if not raw_bytes: return "Empty", 0.0

            entropy = self.calculate_entropy(raw_bytes)
            
            # KI-Feature Extraction
            null_count = raw_bytes.count(0)
            text_chars = sum(1 for b in raw_bytes if 32 <= b <= 126)
            
            # 1. MFT Erkennung (KI-Backup für Rust)
            # MFT Einträge haben oft "FILE" am Start oder 0x30 Attribute und viele Nullen
            if raw_bytes.startswith(b"FILE") or (b"\x30\x00\x00\x00" in raw_bytes and null_count > 100):
                return "MFT Struktur (Fragment)", 95.0

            # 2. Leere Sektoren
            if null_count > len(raw_bytes) * 0.95:
                return "Leer (Zero-Fill)", 100.0
                
            # 3. Verschlüsselung vs. Kompression
            if entropy > 7.8:
                return "Verschlüsselter Container", 85.0
            
            # 4. Text / Logs
            if text_chars > len(raw_bytes) * 0.8:
                return "Text / Logdatei", 90.0

            # 5. Ordnerstruktur Fragmente (INDX)
            if b"INDX" in raw_bytes:
                return "NTFS Index (Ordner)", 99.0

            return "Unbekanntes Fragment", 40.0
        except:
            return "Fehler", 0.0

classifier = ForensicClassifier()