import customtkinter as tk
import importlib
from tkinter import filedialog, ttk
import os
import glob
import json
import subprocess
import threading
import queue
import time
import sys
import base64 # Hinzugefügt für KI Datentransfer

# KI Module dynamisch laden
try:
    rc = importlib.import_module("ai-logic.reconstructor")
    classifier_mod = importlib.import_module("ai-logic.classifier")
    classifier = classifier_mod.classifier
except Exception as e:
    print(f"KI Module nicht geladen: {e}")
    classifier = None

# Windows Taskbar Icon Fix
if sys.platform == "win32":
    import ctypes
    myappid = 'datarec.forensics.v1'
    ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(myappid)

LANGUAGES = {
    "Deutsch": {
        "tab_scan": "Scan", 
        "tab_analysis": "Standardauswertung", 
        "tab_ki_tree": "KI-Ordnerstruktur", 
        "tab_settings": "Einstellungen",
        "config": "Scan Konfiguration", "start": "Scan starten", "stop": "Abbrechen",
        "theme": "Design:", "lang": "Sprache:", "found": "Erkannte Fragmente:",
        "block": "Block", "size": "Größe", "content": "Inhalt",
        "duration": "Vergangen:", "eta": "ETA:", "progress": "Fortschritt:"
    },
    "English": {
        "tab_scan": "Scan", 
        "tab_analysis": "Standard Analysis", 
        "tab_ki_tree": "AI Folder Structure", 
        "tab_settings": "Settings",
        "config": "Scan Configuration", "start": "Start Scan", "stop": "Abort",
        "theme": "Theme:", "lang": "Language:", "found": "Detected Fragments:",
        "block": "Block", "size": "Size", "content": "Content",
        "duration": "Elapsed:", "eta": "ETA:", "progress": "Progress:"
    }
}

FILE_COLORS = {
    "NTFS_MFT": "#3498db", "JPEG": "#f1c40f", "PNG": "#e67e22",
    "GIF": "#d35400", "PDF": "#c0392b", "EXE": "#2ecc71",
    "DLL": "#27ae60", "ZIP_Office": "#9b59b6", "SQLite": "#1abc9c",
    "MP4_MOV": "#34495e", "RIFF": "#16a085", "RAR": "#7f8c8d",
    "7Z": "#95a5a6", "High_Entropy": "#e74c3c", "Unknown": "#333333"
}

class VisualizerFrame(tk.CTkFrame):
    def __init__(self, master, app_instance, **kwargs):
        super().__init__(master, **kwargs)
        self.app = app_instance
        self.blocks = []
        self.block_data = {}
        # FIX: Speicher für Farben, um Flackern zu verhindern
        self.block_colors = ["#333333"] * 800 
        self.total_sectors_cache = 1
        self.sector_size_cache = 4096
        
        from tkinter import Canvas
        self.canvas = Canvas(self, width=940, height=180, bg="#1a1a1a", highlightthickness=0)
        self.canvas.pack(pady=10, padx=10)
        
        self.tooltip = tk.CTkLabel(self.app, text="", fg_color="#2c3e50", text_color="white", 
                                   corner_radius=6, font=("Arial", 12, "bold"), 
                                   padx=10, pady=6, justify="left")
        
        self.create_grid()
        self.canvas.bind("<Motion>", self.on_hover)
        self.canvas.bind("<Leave>", self.hide_tooltip)

    def create_grid(self, rows=8, cols=100):
        self.canvas.delete("all")
        self.blocks = []
        w, h = 940/cols, 180/rows
        for r in range(rows):
            for c in range(cols):
                color = self.block_colors[r * cols + c]
                rect = self.canvas.create_rectangle(
                    c*w, r*h, (c+1)*w-1, (r+1)*h-1, 
                    fill=color, outline="#1a1a1a", activeoutline="cyan"
                )
                self.blocks.append(rect)

    def mark_block(self, index, label, total_sectors, sector_size):
        if 0 <= index < len(self.blocks):
            self.total_sectors_cache = total_sectors
            self.sector_size_cache = int(sector_size)
            
            new_color = FILE_COLORS.get(label, FILE_COLORS["Unknown"])
            
            # FIX: Nur überschreiben, wenn die neue Info wertvoller ist als die alte (kein Flackern zu Grau)
            current_color = self.block_colors[index]
            if label != "Unknown" or current_color == "#333333":
                self.block_colors[index] = new_color
                self.canvas.itemconfig(self.blocks[index], fill=new_color)
            
            if index not in self.block_data:
                self.block_data[index] = []
            
            if label and label not in ["High_Entropy", "", "Unknown"] and label not in self.block_data[index]:
                if len(self.block_data[index]) < 3:
                    self.block_data[index].append(label)

    def on_hover(self, event):
        item = self.canvas.find_closest(event.x, event.y)
        if not item or not self.blocks: 
            self.hide_tooltip()
            return
            
        try:
            rect_id = item[0]
            idx = self.blocks.index(rect_id)
            sectors_per_block = self.total_sectors_cache / len(self.blocks)
            bytes_per_block = sectors_per_block * self.sector_size_cache
            
            size_str = f"{bytes_per_block / (1024**2):.1f} MB" if bytes_per_block < 1024**3 else f"{bytes_per_block / (1024**3):.2f} GB"
            
            data_list = self.block_data.get(idx, [])
            content_str = ", ".join(data_list) if data_list else "-"
            
            l = LANGUAGES[self.app.current_lang]
            tooltip_text = f"{l['block']} #{idx}\n{l['size']}: {size_str}\n{l['content']}: {content_str}"
            
            self.tooltip.configure(text=tooltip_text)
            abs_x = self.canvas.winfo_rootx() - self.app.winfo_rootx() + event.x
            abs_y = self.canvas.winfo_rooty() - self.app.winfo_rooty() + event.y
            self.tooltip.place(x=abs_x + 20, y=abs_y - 40)
            self.tooltip.lift()
        except:
            self.hide_tooltip()

    def hide_tooltip(self, event=None):
        self.tooltip.place_forget()

class App(tk.CTk):
    def __init__(self):
        super().__init__()
        self.title("DataRec Forensics AI")
        self.geometry("1100x950")
        self.current_lang = "Deutsch"

        self.tabview = tk.CTkTabview(self)
        self.tabview.pack(padx=20, pady=10, fill="both", expand=True)
        
        self.tab_scan = self.tabview.add("Scan")
        self.tab_analyse = self.tabview.add("Standardauswertung")
        self.tab_ki_tree = self.tabview.add("KI-Ordnerstruktur")
        self.tab_settings = self.tabview.add("Einstellungen")

        self.setup_scan_tab()
        self.setup_analyse_tab()
        self.setup_ki_tree_tab() # Neuer Tab Setup
        self.setup_settings_tab()

        self.visualizer = VisualizerFrame(self, self)
        self.visualizer.pack(padx=20, pady=20, fill="x")

        self.data_queue = queue.Queue()
        self.is_scanning = False
        self.rust_process = None
        self.selected_source = None
        self.start_time = 0

    def setup_scan_tab(self):
        self.header_scan = tk.CTkLabel(self.tab_scan, text="Scan Konfiguration", font=("Arial", 22, "bold"))
        self.header_scan.pack(pady=15)
        self.source_var = tk.StringVar(value="--- Quelle wählen ---")
        self.dropdown = tk.CTkOptionMenu(self.tab_scan, values=[], variable=self.source_var, command=self.handle_selection, width=500)
        self.dropdown.pack(pady=10)
        self.refresh_drives()
        self.sector_var = tk.StringVar(value="4096")
        self.sector_dropdown = tk.CTkOptionMenu(self.tab_scan, values=["512", "1024", "2048", "4096", "Custom"], variable=self.sector_var, command=self.handle_sector_change)
        self.sector_dropdown.pack(pady=10)
        self.custom_sector_entry = tk.CTkEntry(self.tab_scan, placeholder_text="Custom Bytes...", width=150)
        self.custom_sector_entry.pack(pady=5)
        self.custom_sector_entry.configure(state="disabled")
        self.stats_frame = tk.CTkFrame(self.tab_scan, fg_color="transparent")
        self.stats_frame.pack(pady=20)
        self.status_label = tk.CTkLabel(self.stats_frame, text="Sektor: 0 | 0.0 MB/s", font=("Arial", 14))
        self.status_label.pack()
        self.progress_label = tk.CTkLabel(self.stats_frame, text="Fortschritt: 0.00%", font=("Arial", 18, "bold"), text_color="#3498db")
        self.progress_label.pack(pady=8)
        self.time_label = tk.CTkLabel(self.stats_frame, text="Vergangen: 00:00 | ETA: 00:00", font=("Arial", 13))
        self.time_label.pack()
        self.start_button = tk.CTkButton(self.tab_scan, text="Start", fg_color="#27ae60", command=self.run_scan, width=250, height=40)
        self.start_button.pack(pady=10)
        self.stop_button = tk.CTkButton(self.tab_scan, text="Stop", fg_color="#c0392b", state="disabled", command=self.stop_scan, width=250)
        self.stop_button.pack(pady=5)

    def setup_analyse_tab(self):
        self.lbl_found = tk.CTkLabel(self.tab_analyse, text="Gefundene Signaturen:", font=("Arial", 18, "bold"))
        self.lbl_found.pack(pady=10)
        self.log_text = tk.CTkTextbox(self.tab_analyse, font=("Courier New", 12))
        self.log_text.pack(padx=20, pady=10, fill="both", expand=True)
        self.log_text.configure(state="disabled")

    def setup_ki_tree_tab(self): # NEU: Tab für die KI-rekonstruierte Struktur
        self.ki_header = tk.CTkLabel(self.tab_ki_tree, text="KI-basierte Verzeichnisrekonstruktion", font=("Arial", 20, "bold"))
        self.ki_header.pack(pady=10)
        
        self.tree_container = tk.CTkFrame(self.tab_ki_tree)
        self.tree_container.pack(padx=20, pady=10, fill="both", expand=True)
        
        # Die Baumansicht für die Ordnerstruktur
        self.folder_tree = ttk.Treeview(self.tree_container, columns=("size", "confidence"), show="tree headings")
        self.folder_tree.heading("#0", text="Name / Pfad")
        self.folder_tree.heading("size", text="Größe")
        self.folder_tree.heading("confidence", text="KI-Konfidenz")
        self.folder_tree.pack(fill="both", expand=True)
        
        self.ki_status_label = tk.CTkLabel(self.tab_ki_tree, text="Warte auf KI-Modell zur Rekonstruktion...", font=("Arial", 12, "italic"))
        self.ki_status_label.pack(pady=10)

    def setup_settings_tab(self):
        self.settings_container = tk.CTkFrame(self.tab_settings, fg_color="transparent")
        self.settings_container.pack(expand=True)
        self.lbl_theme = tk.CTkLabel(self.settings_container, text="Design:")
        self.lbl_theme.pack(pady=5)
        tk.CTkOptionMenu(self.settings_container, values=["Dark", "Light"], command=lambda v: tk.set_appearance_mode(v)).pack(pady=10)
        self.lbl_lang = tk.CTkLabel(self.settings_container, text="Sprache:")
        self.lbl_lang.pack(pady=5)
        tk.CTkOptionMenu(self.settings_container, values=["Deutsch", "English"], command=self.change_language).pack(pady=10)

    def format_time(self, seconds):
        if seconds <= 0 or seconds > 3600000: return "00:00"
        m, s = divmod(int(seconds), 60)
        h, m = divmod(m, 60)
        return f"{h:02d}:{m:02d}:{s:02d}" if h > 0 else f"{m:02d}:{s:02d}"

    def change_language(self, lang):
        self.current_lang = lang
        l = LANGUAGES[lang]
        self.tabview._segmented_button._buttons_dict["Scan"].configure(text=l["tab_scan"])
        self.tabview._segmented_button._buttons_dict["Standardauswertung"].configure(text=l["tab_analysis"])
        self.tabview._segmented_button._buttons_dict["KI-Ordnerstruktur"].configure(text=l["tab_ki_tree"])
        self.tabview._segmented_button._buttons_dict["Einstellungen"].configure(text=l["tab_settings"])
        self.header_scan.configure(text=l["config"])
        self.start_button.configure(text=l["start"])
        self.stop_button.configure(text=l["stop"])
        self.lbl_theme.configure(text=l["theme"])
        self.lbl_lang.configure(text=l["lang"])
        self.lbl_found.configure(text=l["found"])

    def refresh_drives(self):
        drives = sorted(glob.glob("/dev/sd[a-z]") + glob.glob("/dev/nvme[0-9]n[0-9]"))
        if sys.platform == "win32":
            import string
            drives += [f"{d}:\\" for d in string.ascii_uppercase if os.path.exists(f"{d}:\\")]
        drives.append("--- Image-Datei wählen ---")
        self.dropdown.configure(values=drives)

    def handle_selection(self, choice):
        if "Image" in choice:
            path = filedialog.askopenfilename()
            if path: self.selected_source = path; self.source_var.set(os.path.basename(path))
        else: self.selected_source = choice; self.source_var.set(choice)

    def handle_sector_change(self, choice):
        self.custom_sector_entry.configure(state="normal" if choice == "Custom" else "disabled")

    def run_scan(self):
        if not self.selected_source: return
        self.is_scanning = True
        self.start_time = time.time()
        self.start_button.configure(state="disabled")
        self.stop_button.configure(state="normal")
        
        # Reset Visualizer
        self.visualizer.block_colors = ["#333333"] * 800
        self.visualizer.block_data = {}
        self.visualizer.create_grid()
        
        self.log_text.configure(state="normal")
        self.log_text.delete("0.0", tk.END)
        self.log_text.configure(state="disabled")
        
        s_size = self.custom_sector_entry.get() if self.sector_var.get() == "Custom" else self.sector_var.get()
        if not s_size.isdigit(): s_size = "4096"
        
        threading.Thread(target=self.rust_worker, args=(self.selected_source, s_size), daemon=True).start()
        self.after(100, self.process_queue)

    def stop_scan(self):
        if self.rust_process: self.rust_process.terminate()

    def rust_worker(self, source, s_size):
        base_dir = os.path.dirname(os.path.abspath(__file__))
        rust_proj_dir = os.path.join(base_dir, "core-scanner")
        bin_path = os.path.join(rust_proj_dir, "target", "release", "core-scanner")
        cmd = [bin_path, source, s_size] if os.path.exists(bin_path) else ["cargo", "run", "--release", "-q", "--", source, s_size]
        cwd_to_use = base_dir if os.path.exists(bin_path) else rust_proj_dir
        try:
            self.rust_process = subprocess.Popen(cmd, cwd=cwd_to_use, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
            for line in iter(self.rust_process.stdout.readline, ''):
                if line.startswith('{'):
                    try: self.data_queue.put(json.loads(line))
                    except: pass
            self.rust_process.wait()
        finally:
            self.is_scanning = False
            self.data_queue.put({"type": "finished"})

    def process_queue(self):
        try:
            while True:
                data = self.data_queue.get_nowait()
                if data["type"] == "status":
                    # --- 1. UI Standard-Updates ---
                    elapsed = time.time() - self.start_time
                    prog = data["progress"] / 100.0
                    eta = (elapsed / prog - elapsed) if prog > 0.005 else 0
                    self.status_label.configure(text=f"Sektor: {data['sector']} | {data['speed']:.1f} MB/s")
                    self.progress_label.configure(text=f"Fortschritt: {data['progress']:.2f}%")
                    self.time_label.configure(text=f"Vergangen: {self.format_time(elapsed)} | ETA: {self.format_time(eta)}")
                    
                    total = data.get("total_sectors", 1)
                    idx = int((data["sector"] / total) * (len(self.visualizer.blocks)-1))
                    
                    # Werte aus dem JSON extrahieren (von main.rs geliefert)
                    label = data.get("label", "Unknown")
                    sector_idx = data["sector"]
                    preview = data.get("preview", "")

                    # Visualizer aktualisieren
                    self.visualizer.mark_block(idx, label, total, self.sector_var.get())
# --- 2. Hierarchische Analyse (MFT -> Carving -> KI) ---
                    
                    # Knoten initialisieren, falls noch nicht vorhanden
                    if not hasattr(self, 'tree_nodes'):
                        self.tree_nodes = {
                            "MFT": self.folder_tree.insert("", "end", text="📁 Dateisystem (MFT)", open=True),
                            "CARVED": self.folder_tree.insert("", "end", text="🔍 Gecarvte Dateien", open=True),
                            "AI": self.folder_tree.insert("", "end", text="🤖 KI-Klassifiziert", open=True)
                        }

                    # Extrahiere MFT-spezifische Daten (die wir in main.rs hinzugefügt haben)
                    filename = data.get("filename", "")
                    is_active = data.get("is_active", True)

                    # LOGIK A: MFT Funde (Hier nutzen wir jetzt den echten Dateinamen!)
                    if label == "NTFS_MFT":
                        display_name = filename if filename else f"MFT_Record_{sector_idx}"
                        status = "Aktiv" if is_active else "GELÖSCHT"
                        
                        self.folder_tree.insert(self.tree_nodes["MFT"], "end", 
                                                text=display_name, 
                                                values=("1024 B", f"{status} (MFT)"))

                    # LOGIK B: Klassisches Carving (Bekannte Dateitypen wie JPEG, PNG, etc.)
                    elif label not in ["Unknown", "High_Entropy"]:
                        # Hier fügen wir die Datei in den Carved-Ordner ein
                        self.folder_tree.insert(self.tree_nodes["CARVED"], "end", 
                                                text=f"Rec_{sector_idx}.{label.lower()}", 
                                                values=("Fragment", "Magic Bytes erkannt"))
                        
                        # Eintrag in das Standard-Text-Log (Standardauswertung Tab)
                        self.log_text.configure(state="normal")
                        self.log_text.insert(tk.END, f"[FUND @ {sector_idx}] Typ: {label}\n")
                        self.log_text.see(tk.END)
                        self.log_text.configure(state="disabled")

                    # LOGIK C: KI-Analyse für Unbekanntes
                    elif label in ["High_Entropy", "Unknown"] and classifier and preview:
                        # Hier rufen wir dein KI-Modell auf
                        ai_label, conf = classifier.classify_sector(preview)
                        
                        if conf > 65: # Schwellenwert für Relevanz
                                self.folder_tree.insert(self.tree_nodes["AI"], "end", 
                                                    text=f"Sektor_{sector_idx}", 
                                                    values=(f"{ai_label}", f"{conf}% Konfidenz"))

                elif data["type"] == "finished":
                    self.start_button.configure(state="normal")
                    self.stop_button.configure(state="disabled")
                    self.ki_status_label.configure(text="Scan abgeschlossen.")
                    
                    # HIER: Reconstructor starten, um CSV auszuwerten
                    if rc:
                        threading.Thread(target=rc.analyze_findings_csv, args=("../shared/findings.csv",), daemon=True).start()
                    
                    return
        except queue.Empty: pass
        if self.is_scanning: self.after(100, self.process_queue)

if __name__ == '__main__':
    app = App()
    app.mainloop()