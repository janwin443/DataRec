use std::fs::File;
use std::io::{self, Read, Seek, SeekFrom, Write};
use std::time::Instant;

// Module einbinden
mod entropy;
use entropy::{calculate_entropy, is_interesting};

fn main() -> io::Result<()> {
    loop {
        println!("\n======================================");
        println!("      \x1b[95mDataRec AI Forensic Core\x1b[0m        ");
        println!("======================================");
        println!("1. Start Full Scan (/mnt/festplatte.img)");
        println!("2. Disk Info anzeigen");
        println!("3. Beenden");
        print!("\nAuswahl: ");
        
        io::stdout().flush()?;

        let mut input = String::new();
        io::stdin().read_line(&mut input)?;
        let choice = input.trim();

        match choice {
            "1" => {
                if let Err(e) = run_scanner() {
                    println!("\n\x1b[91mFehler beim Scan:\x1b[0m {}", e);
                }
            },
            "2" => display_disk_info(),
            "3" => {
                println!("Programm beendet. Bis bald!");
                break;
            },
            _ => println!("\x1b[91mUngültige Auswahl, bitte 1, 2 oder 3 wählen.\x1b[0m"),
        }
    }
    Ok(())
}

fn display_disk_info() {
    let path = "/mnt/festplatte.img";
    let sector_size: u64 = 4096;
    let total_sectors: u64 = 805306368;
    let total_size_gb = (total_sectors * sector_size) as f64 / 1_073_741_824.0;

    println!("\n--- \x1b[94mDisk Information\x1b[0m ---");
    println!("Pfad:          {}", path);
    println!("Sektor-Größe:  {} Bytes", sector_size);
    println!("Anzahl Sektoren: {}", total_sectors);
    println!("Gesamtgröße:   {:.2} GB (~3.3 TB)", total_size_gb);
    println!("--------------------------");
}

fn run_scanner() -> io::Result<()> {
    let path = "/mnt/festplatte.img";
    let sector_size: u64 = 4096;
    let total_sectors: u64 = 805306368;

    let mut file = File::open(path)?;
    let mut buffer = vec![0u8; sector_size as usize];
    let start_time = Instant::now();

    println!("\n\x1b[94m[INFO]\x1b[0m Öffne Image: {}", path);
    println!("\x1b[94m[INFO]\x1b[0m Scan läuft... Drücke Strg+C zum Abbrechen.\n");

    for i in 0..total_sectors {
        // Sektor lesen
        file.seek(SeekFrom::Start(i * sector_size))?;
        if file.read_exact(&mut buffer).is_err() { break; }

        // Analyse
        let ent = calculate_entropy(&buffer);
        
        // Funde melden (ohne den Balken zu unterbrechen)
        if buffer.starts_with(b"FILE0") {
            println!("\n\x1b[93m[MFT RECORD]\x1b[0m Sektor: {} | Entropie: {:.4}", i, ent);
        } else if i % 500_000 == 0 && is_interesting(ent) {
            // Gelegentliche Stichprobe für interessante Daten
            // (Damit das Terminal nicht mit "Daten gefunden" geflutet wird)
        }

        // Fortschrittsbalken-Update (alle 250.000 Sektoren)
        if i % 250_000 == 0 && i > 0 {
            let elapsed = start_time.elapsed().as_secs_f64();
            let progress = (i as f64 / total_sectors as f64) * 100.0;
            
            let bytes_processed = i * sector_size;
            let gb_processed = bytes_processed as f64 / 1_073_741_824.0;
            let mb_per_sec = (bytes_processed as f64 / 1_048_576.0) / elapsed;

            let bar_width = 25;
            let filled = ((progress / 100.0) * bar_width as f64) as usize;
            let bar: String = std::iter::repeat("█").take(filled)
                .chain(std::iter::repeat("░").take(bar_width - filled))
                .collect();

            print!(
                "\r\x1b[92mProgress: [{}] {:.2}% | {:.1} GB | {:.1} MB/s\x1b[0m", 
                bar, progress, gb_processed, mb_per_sec
            );
            io::stdout().flush()?;
        }
    }

    let duration = start_time.elapsed();
    println!(
        "\n\n\x1b[92m[FERTIG]\x1b[0m Scan beendet nach {:.1} Minuten.", 
        duration.as_secs_f64() / 60.0
    );
    Ok(())
}