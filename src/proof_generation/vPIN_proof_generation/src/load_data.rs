use serde_json::Value;
use std::{fs::File, io::Read, path::{Path, PathBuf}, str::FromStr, env};


pub fn load_data(network: &str) -> (usize, Vec<u128>, Vec<Vec<i64>>, Vec<Vec<i64>>, usize) {
    let file1_path_str = format!("rust_files/{}/pointMult/weight.json", network);
    let file2_path_str = format!("rust_files/{}/pointMult/point_mult_px_byte.json", network);
    let file3_path_str = format!("rust_files/{}/pointMult/point_mult_py_byte.json", network);

    let file_path = Path::new(&file1_path_str);
    let file2_path = Path::new(&file2_path_str);
    let file3_path = Path::new(&file3_path_str);

    let mut file = File::open(&file_path).expect("Failed to open file");
    let mut contents = String::new();
    file.read_to_string(&mut contents).expect("Failed to read file");

    let parsed: Vec<String> = serde_json::from_str(&contents).expect("Failed to parse JSON");

    let weights: Vec<u128> = parsed
        .into_iter()
        .map(|weight_str| u128::from_str(weight_str.as_str()).expect("Failed to parse weight"))
        .collect();

    let weights_len = weights.len() as usize;

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
