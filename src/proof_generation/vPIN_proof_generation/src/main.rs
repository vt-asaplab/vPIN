pub mod point_addition;
pub mod point_mult;
pub mod commit_test;
pub mod load_data;
pub mod load_data_add;


mod proof_point_add;
mod proof_point_mult;

use proof_point_add::proof_point_add;
use proof_point_mult::proof_point_mult;

fn main() {
    //let (proof_size_add, proof_gen_time_add, proof_ver_time_add) = proof_point_add(); // Call the proof for point addition
    println!("");
    let (proof_size_mult, proof_gen_time_mult, proof_ver_time_mult) = proof_point_mult(); // Call the proof for point multiplication

    /*

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

    */
}
