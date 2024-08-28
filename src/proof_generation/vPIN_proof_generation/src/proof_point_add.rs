#![allow(non_snake_case)]
extern crate curve25519_dalek;
extern crate libspartan;
extern crate merlin;
extern crate sys_info;

use uint::construct_uint;
use curve25519_dalek::scalar::Scalar;
use libspartan::{InputsAssignment, Instance, SNARKGens, VarsAssignment, SNARK};
use merlin::Transcript;
use rand::rngs::OsRng;
use libspartan::random::RandomTape;
use libspartan::dense_mlpoly::{DensePolynomial, EqPolynomial, PolyCommitment, PolyCommitmentBlinds, PolyEvalProof};
use libspartan::r1csinstance::{R1CSEvalProof, R1CSInstance};
use libspartan::r1csproof::{R1CSGens, R1CSProof};
use std::time::{Instant, SystemTime};
use serde::{Serialize, Deserialize};

use crate::point_addition::point_addition;
use crate::point_mult::point_mult;
use crate::commit_test::{my_lib_verify, my_lib_prove, my_dense_mlpoly_commit};

pub fn proof_point_add() -> (usize, u128, u128) {
let sy_time = SystemTime::now();

let (
    num_cons,
    num_vars,
    num_inputs,
    num_non_zero_entries,
    inst,
    padded_vars_para,
    padded_vars_input,
    padded_vars,
    assignment_inputs,
  ) = point_addition();

  // produce public parameters
  let gens = SNARKGens::new(num_cons, num_vars, num_inputs, num_non_zero_entries);

  // create a commitment to the R1CS instance
  let (comm, decomm) = SNARK::encode(&inst, &gens);

  let mut random_tape_1 = RandomTape::new(&[2u8]);

  //commit to the var_para assignments
  let poly_vars_para = DensePolynomial::new(padded_vars_para.assignment.clone());
  let (comm_vars_para, blind_vars_para) = poly_vars_para.commit(&gens.gens_r1cs_sat.gens_pc, Some(&mut random_tape_1));
  
  //commit to the inputs
  let poly_vars_inputs = DensePolynomial::new(padded_vars_input.assignment.clone());
  let (comm_vars_input, blind_vars_input) = poly_vars_inputs.commit(&gens.gens_r1cs_sat.gens_pc, Some(&mut random_tape_1));

  let poly_vars = DensePolynomial::new(padded_vars.assignment.clone());

  let blind_para = blind_vars_para.blinds;
  let blind_input = blind_vars_input.blinds;
  let (comm_vars, blind_vars) = my_dense_mlpoly_commit(&poly_vars, &gens.gens_r1cs_sat.gens_pc,
                                                      blind_para, blind_input);
  
  let mut poly_prime = poly_vars.clone();
  for i in 0..poly_vars.len {
    poly_prime.Z[i] = poly_vars_para.Z[i] + poly_vars_inputs.Z[i]
  }

    // poly_prime.Z = poly_vars_para.Z + poly_vars_inputs.Z;
  assert_eq!(poly_prime.num_vars, poly_vars.num_vars);

  let a = comm_vars_para.C[0].decompress().unwrap();
  let b = comm_vars_input.C[0].decompress().unwrap();
  let c = a + b;
  let c_prime = comm_vars.C[0].decompress().unwrap();
  assert_eq!(c, c_prime);

  let mut combine_comm_vars = vec![];
  for i in 0..comm_vars_para.C.len() {
      combine_comm_vars.push((comm_vars_para.C[i].decompress().unwrap() + comm_vars_input.C[i].decompress().unwrap()).compress());
  };

  let combine_commitment = PolyCommitment { C: combine_comm_vars };

  // produce a proof of satisfiability
  let mut prover_transcript = Transcript::new(b"snark_example");
  let proof = my_lib_prove(
      &inst,
      &decomm,
      padded_vars,
      &assignment_inputs,
      &gens,
      &mut prover_transcript,
      poly_vars,
      combine_commitment,
      blind_vars,
  );

  let serialized_proof = bincode::serialize(&proof).expect("Serialization failed");
  let proof_size_bytes = serialized_proof.len();
  println!("Proof size: {} bytes", proof_size_bytes);

  let proof_gen_time = SystemTime::now().duration_since(sy_time).unwrap().as_millis();
  println!("Proof generation time: {:?} ms", proof_gen_time);
  let sy_time2 = SystemTime::now();

  // verify the proof of satisfiability
  let mut verifier_transcript = Transcript::new(b"snark_example");
  assert!(my_lib_verify(proof, &comm, &assignment_inputs, &mut verifier_transcript, &gens, comm_vars_para, comm_vars_input)
      .is_ok());
  println!("Proof verification successful!");
  
  let proof_verification_time = SystemTime::now().duration_since(sy_time2).unwrap().as_millis();
  println!("Proof verification time: {:?} ms", proof_verification_time);

  (proof_size_bytes, proof_gen_time, proof_verification_time)
}