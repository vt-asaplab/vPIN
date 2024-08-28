use serde_json::Value;
use std::{env, fs::File, io::Read, str::FromStr};
use std::path::PathBuf;

pub fn load_data_add() -> (usize, Vec<Vec<i64>>, Vec<Vec<i64>>, Vec<Vec<i64>>, Vec<Vec<i64>>, Vec<i64>) {
    // Get the current working directory
    let mut current_dir = env::current_dir().expect("Failed to get current directory");

    // Navigate up two levels from the current directory
    current_dir.pop(); 
    current_dir.pop(); 
    current_dir.pop(); 
    
    // Define the base directory path
    let base_dir = current_dir.join("rust_files");

    let file1_path = base_dir.join("pointAdd/point_add_px_byte.json");
    let mut file1 = File::open(&file1_path).expect("Failed to open file");
    let mut contents1 = String::new();
    file1.read_to_string(&mut contents1).expect("Failed to read file");

    let parsed1: Vec<Vec<Value>> = serde_json::from_str(&contents1).expect("Failed to parse JSON");

    let mut point_add_px_byte: Vec<Vec<i64>> = vec![];
    for row in parsed1 {
        let mut inner_row: Vec<i64> = vec![];
        for value in row {
            if let Some(num) = value.as_i64() {
                inner_row.push(num);
            }
        }
        point_add_px_byte.push(inner_row);
    }

    let len = point_add_px_byte.len() as usize;

    let file2_path = base_dir.join("pointAdd/point_add_py_byte.json");
    let mut file2 = File::open(&file2_path).expect("Failed to open file");
    let mut contents2 = String::new();
    file2.read_to_string(&mut contents2).expect("Failed to read file");

    let parsed2: Vec<Vec<Value>> = serde_json::from_str(&contents2).expect("Failed to parse JSON");

    let mut point_add_py_byte: Vec<Vec<i64>> = vec![];
    for row in parsed2 {
        let mut inner_row: Vec<i64> = vec![];
        for value in row {
            if let Some(num) = value.as_i64() {
                inner_row.push(num);
            }
        }
        point_add_py_byte.push(inner_row);
    }

    let file3_path = base_dir.join("pointAdd/point_add_rx_byte.json");
    let mut file3 = File::open(&file3_path).expect("Failed to open file");
    let mut contents3 = String::new();
    file3.read_to_string(&mut contents3).expect("Failed to read file");

    let parsed3: Vec<Vec<Value>> = serde_json::from_str(&contents3).expect("Failed to parse JSON");

    let mut point_add_rx_byte: Vec<Vec<i64>> = vec![];
    for row in parsed3 {
        let mut inner_row: Vec<i64> = vec![];
        for value in row {
            if let Some(num) = value.as_i64() {
                inner_row.push(num);
            }
        }
        point_add_rx_byte.push(inner_row);
    }

    let file4_path = base_dir.join("pointAdd/point_add_ry_byte.json");
    let mut file4 = File::open(&file4_path).expect("Failed to open file");
    let mut contents4 = String::new();
    file4.read_to_string(&mut contents4).expect("Failed to read file");

    let parsed4: Vec<Vec<Value>> = serde_json::from_str(&contents4).expect("Failed to parse JSON");

    let mut point_add_ry_byte: Vec<Vec<i64>> = vec![];
    for row in parsed4 {
        let mut inner_row: Vec<i64> = vec![];
        for value in row {
            if let Some(num) = value.as_i64() {
                inner_row.push(num);
            }
        }
        point_add_ry_byte.push(inner_row);
    }
      
    let file5_path = base_dir.join("pointAdd/point_add_rz_byte.json");
    let mut file5 = File::open(&file5_path).expect("Failed to open file");
    let mut contents5 = String::new();
    file5.read_to_string(&mut contents5).expect("Failed to read file");

    let parsed5: Vec<Value> = serde_json::from_str(&contents5).expect("Failed to parse JSON");

    let mut point_add_rz_byte: Vec<i64> = vec![];
    for value in parsed5 {
        if let Some(num) = value.as_i64() {
            point_add_rz_byte.push(num);
        }
    }

    (len, point_add_px_byte, point_add_py_byte, point_add_rx_byte, point_add_ry_byte, point_add_rz_byte)
}
