use tch::{CModule, Tensor, Kind, no_grad};
use memmap2::MmapOptions;
use std::fs::File;
use std::io::{Write, BufWriter}; // BufWriter für Performance
use anyhow::Result;

fn main() -> Result<()> {
    // 1. Pfade definieren
    let model_path = "../ai-logic/mft_detektor.pt";
    let args: Vec<String> = std::env::args().collect();
    // Nutzt das Argument der GUI, sonst Fallback auf das Test-Image
    let image_path = args.get(1).map(|s| s.as_str()).unwrap_or("../shared/festplatte.img");
    let sector_size: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(4096);

    // 2. KI Modell laden
    let model = CModule::load(model_path)
        .expect("Konnte das Modell nicht finden! Hast du es in Python exportiert?");
    
    // 3. Image Datei öffnen
    let file = File::open(image_path)?;
    let mmap = unsafe { MmapOptions::new().map(&file)? };

    println!("Scanning {} ({:.2} GB)...", image_path, mmap.len() as f64 / 1e9);

    // 4. Datei VOR der Schleife zum Schreiben öffnen
    // Wir nutzen einen BufWriter, damit Rust nicht für jedes f32 einzeln die Festplatte weckt
    let out_file = File::create("../shared/heatmap_output.bin")?;
    let mut writer = BufWriter::new(out_file);

    // 5. KI-Analyse (Inferenz)
    // Wir geben das Result aus der Closure nach oben weiter
    no_grad(|| -> Result<()> {
        for (i, chunk) in mmap.chunks_exact(sector_size).enumerate() {
            
            // KI-Häppchen vorbereiten (1024 Bytes)
            let input_chunk = &chunk[..1024];
            let input_data: Vec<f32> = input_chunk.iter().map(|&b| b as f32 / 255.0).collect();

            let input = Tensor::from_slice(&input_data)
                .view([1, 1, 1024])
                .to_kind(Kind::Float);

            // KI entscheiden lassen
            let output = model.forward_ts(&[input])?; // Hier direkt das ? nutzen
            let probabilities = output.softmax(-1, Kind::Float);
            
            // MFT Score extrahieren
            let score = probabilities.double_value(&[0, 1]) as f32;
            
            // SOFORT SCHREIBEN
            writer.write_all(&score.to_le_bytes())?;

            if i % 10000 == 0 {
                let progress = (i as f64 / (mmap.len() / sector_size) as f64) * 100.0;
                    // Dieses Format braucht die GUI:
                    println!("{}", serde_json::json!({
                        "type": "status",
                        "sector": i,
                        "progress": progress,
                        "speed": 0.0, // Geschwindigkeit optional
                        "total_sectors": mmap.len() / sector_size
                    }));
            }
        }
        Ok(())
    })?; // Das doppelte ?? ist nötig, um das Result der Closure UND von no_grad zu entpacken

    // Puffer am Ende leeren
    writer.flush()?;

    println!("Scan beendet. Heatmap unter ../shared/heatmap_output.bin");
    Ok(())
}