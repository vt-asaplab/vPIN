use serde_json::Value;
use std::{env, fs::File, io::Read, str::FromStr};
use std::path::PathBuf;

pub fn load_data() -> (usize, Vec<u128>, Vec<Vec<i64>>, Vec<Vec<i64>>, usize) {
    // Get the current working directory
    let mut current_dir = env::current_dir().expect("Failed to get current directory");

    // Navigate up two levels from the current directory
    current_dir.pop(); 
    current_dir.pop(); 
    current_dir.pop(); 
    
    // Define the base directory path
    let base_dir = current_dir.join("rust_files");

    let file_path = base_dir.join("pointMult/weight.json");
    let mut file = File::open(&file_path).expect("Failed to open file");
    let mut contents = String::new();
    file.read_to_string(&mut contents).expect("Failed to read file");

    let parsed: Vec<String> = serde_json::from_str(&contents).expect("Failed to parse JSON");

    let weights: Vec<u128> = parsed
        .into_iter()
        .map(|weight_str| u128::from_str(weight_str.as_str()).expect("Failed to parse weight"))
        .collect();

    let weights_len = weights.len() as usize;

    let file2_path = base_dir.join("pointMult/point_mult_px_byte.json");
    let mut file2 = File::open(&file2_path).expect("Failed to open file");
    let mut contents2 = String::new();
    file2.read_to_string(&mut contents2).expect("Failed to read file");

    let parsed2: Vec<Vec<Value>> = serde_json::from_str(&contents2).expect("Failed to parse JSON");

    let mut point_mult_x_byte: Vec<Vec<i64>> = vec![];
    for row in parsed2 {
        let mut inner_row: Vec<i64> = vec![];
        for value in row {
            if let Some(num) = value.as_i64() {
                inner_row.push(num);
            }
        }
        point_mult_x_byte.push(inner_row);
    }

    let file3_path = base_dir.join("pointMult/point_mult_py_byte.json");
    let mut file3 = File::open(&file3_path).expect("Failed to open file");
    let mut contents3 = String::new();
    file3.read_to_string(&mut contents3).expect("Failed to read file");

    // Parse the JSON into a nested vector
    let parsed3: Vec<Vec<Value>> = serde_json::from_str(&contents3).expect("Failed to parse JSON");

    let mut point_mult_y_byte: Vec<Vec<i64>> = vec![];
    for row2 in parsed3 {
        let mut inner_row2: Vec<i64> = vec![];
        for value2 in row2 {
            if let Some(num2) = value2.as_i64() {
                inner_row2.push(num2);
            }
        }
        point_mult_y_byte.push(inner_row2);
    }
        
    (weights_len, weights, point_mult_x_byte, point_mult_y_byte, 128)
}
