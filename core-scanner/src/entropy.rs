pub fn calculate_entropy(data: &[u8]) -> f64 {
    if data.is_empty() {
        return 0.0;
    }

    // 1. Table for all 256-Byte possible values (0x00 - 0xFF (HEX))
    let mut counts = [0usize; 256];
    for &byte in data {
        counts[byte as usize] += 1;
    }

    let mut entropy = 0.0;
    let len = data.len() as f64;

    // 2. Shannon-formula: H = -sum(p_i * log2(p_i))
    for &count in counts.iter() {
        if count > 0 {
            let p = count as f64 / len;
            entropy -= p * p.log2();
        }
    }

    entropy
}

pub fn is_interesting(entropy: f64) -> bool {
    entropy > 0.001
}