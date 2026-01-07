mod entropy;

use entropy::{calculate_entropy, is_interesting};
use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::path::Path;

fn main() -> std::io::Result<()> {
    let path = "/mnt/festplatte.img";
    let file = File::open(path)?;
    let metadata = file.metadata()?;
    let total_bytes = metadata.len();
    let sector_size: u64 = 4096;
    let total_sectors = total_bytes / sector_size;
    let remainder = total_bytes % sector_size;
    let mut buffer = vec![0u8; sector_size as usize];
    let entropy_value = calculate_entropy(&buffer);



    println!("Initializing DataRec 1.0...");
    println!("Disk/IMG used: {}", path);
    println!("Total Size: {} Bytes", total_bytes);
    println!("Total Sectors with size {}: {}", sector_size, total_sectors);

    if remainder > 0 {
    println!("Warning: Disk not ending with a complete sector (overflow {} bytes)", remainder);
    }

    // Starting CLI

    

    Ok(())

}