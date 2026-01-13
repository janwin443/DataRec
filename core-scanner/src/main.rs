use std::fs::{self, File, OpenOptions};
use std::io::{self, Read, Write, Seek, SeekFrom};
use std::path::Path;
use std::time::Instant;
use std::env;
use rayon::prelude::*;
use serde_json::json;
use base64::{Engine as _, engine::general_purpose};

mod entropy;
use entropy::calculate_entropy;

fn main() -> io::Result<()> {
    let args: Vec<String> = env::args().collect();
    let target_path = args.get(1).expect("Fehler: Kein Pfad übergeben");
    let sector_size = args.get(2)
        .and_then(|s| s.parse::<usize>().ok())
        .unwrap_or(4096);

    run_scanner(target_path, sector_size)
}

/// Versucht, die echte Dateigröße anhand des Headers zu erraten
fn detect_file_size(sector: &[u8], label: &str) -> usize {
    match label {
        // PNGs haben die Größe oft in Chunks, aber wir nehmen pauschal mehr an, 
        // oder müssten den IEND Chunk suchen. Hier vereinfacht:
        "PNG" => 5 * 1024 * 1024, // Scannt bis zu 5 MB blind (Carving Logic benötigt Stream)
        "JPEG" => 3 * 1024 * 1024,
        "PDF" => 2 * 1024 * 1024,
        "ZIP_Office" => 10 * 1024 * 1024, // Office Docs können groß sein
        "NTFS_MFT" => 1024, // MFT Einträge sind exakt 1024 Bytes
        _ => 4096 // Fallback: 1 Sektor
    }
}

/// Extrahiert rudimentäre Infos aus einem MFT Record (FILE0)
fn parse_mft_record(data: &[u8]) -> Option<(String, bool)> {
    if data.len() < 1024 { return None; }
    // MFT Header Check "FILE"
    if &data[0..4] != b"FILE" { return None; }
    
    // Flags Offset 0x16 (22): 0x01 = InUse, 0x02 = Directory
    let flags = u16::from_le_bytes([data[22], data[23]]);
    let is_active = (flags & 0x01) != 0;
    let is_folder = (flags & 0x02) != 0;

    // Dateinamen liegen in Attributen (0x30). 
    // Das ist komplex zu parsen ohne Crate 'ntfs'. 
    // Wir machen einen Heuristic-Scan nach UTF-16 Strings im Record.
    let mut filename = "Unknown_MFT_File".to_string();
    
    // Suche nach lesbaren UTF-16 Strings (einfache Heuristik)
    for i in 0..data.len()-20 {
        // Filename Attribute ID ist oft um Offset 0x30 herum, aber variabel.
        // Wir suchen einfach nach einer Sequenz von lesbaren Chars mit 0x00 dazwischen
        if data[i+1] == 0x00 && data[i+3] == 0x00 && data[i].is_ascii_alphanumeric() {
            let mut extracted = String::new();
            let mut j = i;
            while j < data.len()-1 && data[j+1] == 0x00 && (data[j].is_ascii_graphic() || data[j] == b' ') {
                extracted.push(data[j] as char);
                j += 2;
            }
            if extracted.len() > 3 {
                filename = extracted;
                break; // Ersten Treffer nehmen
            }
        }
    }

    if is_folder {
        filename = format!("[DIR] {}", filename);
    }

    Some((filename, is_active))
}

fn identify_label(sector: &[u8]) -> Option<String> {
    if sector.len() < 8 { return None; }
    let label = match &sector[0..8] {
        s if s.starts_with(b"\x89PNG\r\n\x1a\n") => "PNG",
        s if s.starts_with(&[0xFF, 0xD8, 0xFF]) => "JPEG",
        s if s.starts_with(b"GIF87a") || s.starts_with(b"GIF89a") => "GIF",
        s if s.starts_with(b"%PDF-") => "PDF",
        s if s.starts_with(b"PK\x03\x04") => "ZIP_Office",
        s if s.starts_with(b"Rar!\x1a\x07") => "RAR",
        s if s.starts_with(b"7z\xBC\xAF\x27\x1C") => "7Z",
        s if s.starts_with(b"FILE0") => "NTFS_MFT",
        s if s.starts_with(b"MZ") => "EXE",
        s if s.starts_with(b"SQLite format 3\0") => "SQLite",
        s if s.starts_with(b"RIFF") => "RIFF",
        s if s.starts_with(&[0x00, 0x00, 0x00, 0x18, b'f', b't', b'y', b'p']) => "MP4_MOV",
        _ => return None,
    };
    Some(label.to_string())
}

fn run_scanner(path: &str, sector_size: usize) -> io::Result<()> {
    rotate_logs()?;
    let metadata = fs::metadata(path)?;
    let total_size = metadata.len();
    let total_sectors = total_size / sector_size as u64;
    let _ = fs::create_dir_all("../shared/extracted");
    
    // File muss "cloneable" sein oder wir öffnen es für Reader neu
    let mut file = File::open(path)?;
    
    let chunk_size = 1024 * 1024 * 4; // 4MB Chunks
    let start_time = Instant::now();
    let mut current_pos: u64 = 0;

    // Buffer für Datei-Extraction (wenn wir mehr als 1 Sektor lesen wollen)
    let mut extraction_file = File::open(path)?;

    while current_pos < total_size {
        let mut buffer = vec![0u8; chunk_size];
        let bytes_read = file.read(&mut buffer)?;
        if bytes_read == 0 { break; }
        buffer.truncate(bytes_read);

        let findings: Vec<(u64, f64, String, Vec<u8>)> = buffer
            .par_chunks(sector_size)
            .enumerate()
            .filter_map(|(idx, sector)| {
                let sector_index = (current_pos / sector_size as u64) + idx as u64;
                let ent = calculate_entropy(sector);
                let detected_label = identify_label(sector);

                // Filter: Signatur oder sehr hohe Entropie
                if detected_label.is_some() || (ent > 7.8 && sector_index % 500 == 0) {
                    let final_label = detected_label.unwrap_or_else(|| "High_Entropy".to_string());
                    Some((sector_index, ent, final_label, sector.to_vec()))
                } else {
                    None
                }
            })
            .collect();

        // ... (Anfang bleibt gleich)
        // Ergebnisse verarbeiten
        for (idx, ent, label, data) in findings {
            // 1. MESSUNG: Wir schicken nur bei Treffern ODER in festen Intervallen ein Update
            // Das verhindert das Ruckeln in der GUI massiv.
            let should_send = label != "High_Entropy" || idx % 1000 == 0;

            let mut mft_info = String::new();
            let mut is_active = false;
            
            if label == "NTFS_MFT" {
                if let Some((fname, active)) = parse_mft_record(&data) {
                    mft_info = fname;
                    is_active = active;
                }
            }

            // ... (Deine Carving-Logik mit extraction_file bleibt hier)

            if should_send {
                // KI PREVIEW: Für die echte KI-Wiederherstellung brauchen wir mehr als 64 Bytes!
                // Ich empfehle 256 Bytes, damit das Modell genug Kontext hat.
                let preview_len = std::cmp::min(data.len(), 256); 
                let b64_preview = general_purpose::STANDARD.encode(&data[..preview_len]);

                let status = json!({
                    "type": "status",
                    "sector": idx,
                    "total_sectors": total_sectors,
                    "label": label,
                    "entropy": ent,
                    "preview": b64_preview, // Das hier füttert jetzt dein Python-KI-Modell
                    "filename": if label == "NTFS_MFT" { mft_info } else { "".to_string() },
                    "is_active": is_active,
                    "progress": (current_pos as f64 / total_size as f64) * 100.0,
                    "speed": (current_pos as f64 / 1_048_576.0) / (start_time.elapsed().as_secs_f64() + 0.001)
                });

                println!("{}", status.to_string());
                let _ = io::stdout().flush();
            }

            // WICHTIG: Die Datei-Logik (log_finding) sollte immer laufen, 
            // damit die findings.csv für den Reconstructor vollständig ist.
            let _ = log_finding(idx, ent, &label);
        }
        current_pos += bytes_read as u64;
// ... (Rest bleibt gleich)
    }
    println!("{}", json!({"type": "finished"}).to_string());
    Ok(())
}

fn log_finding(sector: u64, entropy: f64, label: &str) -> io::Result<()> {
    let mut file = OpenOptions::new().append(true).open("../shared/findings.csv")?;
    writeln!(file, "{},{:.4},\"{}\"", sector, entropy, label)?;
    Ok(())
}

fn rotate_logs() -> io::Result<()> {
    if !Path::new("../shared").exists() { fs::create_dir_all("../shared")?; }
    let mut file = File::create("../shared/findings.csv")?; 
    writeln!(file, "sector,entropy,label")?;
    Ok(())
}

fn save_sector(sector_idx: u64, data: &[u8], label: &str) -> io::Result<()> {
    let safe_label = label.replace(" ", "_");
    let file_name = format!("../shared/extracted/sector_{}_{}.bin", sector_idx, safe_label);
    let mut file = File::create(file_name)?;
    file.write_all(data)?;
    Ok(())
}

fn save_full_file(sector_idx: u64, data: &[u8], label: &str) -> io::Result<()> {
    // Speichert die gecarvte (größere) Datei
    let ext = label.split('_').next().unwrap_or("bin").to_lowercase();
    let file_name = format!("../shared/extracted/restored_{}.{}", sector_idx, ext);
    let mut file = File::create(file_name)?;
    file.write_all(data)?;
    Ok(())
}

fn extract_name(data: &[u8]) -> String {
    if data.len() < 512 { return "".to_string(); }
    // Suche nach einem plausiblen UTF-16 Namen im MFT Record (sehr vereinfacht)
    // Wir suchen nach einer Sequenz von ASCII gefolgt von Null-Bytes
    for i in 40..data.len() - 10 {
        if data[i].is_ascii_alphanumeric() && data[i+1] == 0x00 && data[i+2].is_ascii_alphanumeric() {
            let mut name = String::new();
            let mut j = i;
            while j < data.len() - 1 && data[j] != 0x00 && data[j].is_ascii_graphic() {
                name.push(data[j] as char);
                j += 2; // NTFS nutzt UTF-16LE, also überspringen wir die Null-Bytes
                if name.len() > 50 { break; }
            }
            if name.len() > 2 { return name; }
        }
    }
    "Unbekannte_Datei".to_string()
}