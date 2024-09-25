pub mod point_addition;
pub mod point_mult;
pub mod commit_test;
pub mod load_data;
pub mod load_data_add;

use std::env;
mod proof_point_add;
mod proof_point_mult;

use proof_point_add::proof_point_add;
use proof_point_mult::proof_point_mult;

fn main() {
    let args: Vec<String> = env::args().collect();
    let network = if args.len() > 1 { &args[1] } else { "1" };

    println!("network: {}", network);

    let (proof_size_add, proof_gen_time_add, proof_ver_time_add) = proof_point_add(network); // Call the proof for point addition
    println!("");
    
    let (proof_size_mult, proof_gen_time_mult, proof_ver_time_mult) = if network == "L2" || network == "L4" {
        println!("Number of Point Multiplications: 0");
        println!("Proof size: 0 bytes");
        println!("Proof generation time: 0 ms");
        println!("Proof verification time: 0 ms");
        (0, 0, 0)
    } else {
        proof_point_mult(network)
    };

    // Calculate the total proof size, generation time, and verification time
    let total_proof_size = proof_size_add + proof_size_mult;
    let total_proof_gen_time = proof_gen_time_add + proof_gen_time_mult;
    let total_proof_ver_time = proof_ver_time_add + proof_ver_time_mult;

    
    // Print the total values
    println!("\n====================================");
    println!("Total proof size: {} bytes", total_proof_size);
    println!("Total proof generation time: {} ms", total_proof_gen_time);
    println!("Total proof verification time: {} ms", total_proof_ver_time);
    println!("====================================");

}
