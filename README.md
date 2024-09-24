# Privacy-Preserving Verifiable Neural Network Inference Service

This repository contains the full implementation of **vPIN** paper (accepted to ACSAC 2024).

**Warning**: This code is a proof-of-concept prototype and is not ready for production use.

## Code Structure
- **src/cnn_networks/Pre_computed_table/**: Contains `baby-step-giant-step.py` for generating the pre-computed table.
- **src/cnn_networks/**: Contains `Server.py` and `Client.py` files for five different CNN networks used to generate results for Figure 2.
- **src/convolution/**: Contains `Server.py` and `Client.py` files for convolutional layer with different filter sizes and input sizes to generate results for Figure 3.
- **src/LeNet/**: Contains `Server.py` and `Client.py` files for LeNet model used to generate results for Table 2.
- **src/proof_generation/vPIN_proof_generation/src/**: Contains files for generating proof for point additions and point multiplications.
- **src/accuracy/**: Contains `train_test_lenet5.py` for assessing accuracy.


## Prerequisites

Before running the scripts, ensure that you have the following installed:

- **Python 3.8+**
- **Rust** (rustc 1.72.0-nightly)
- **Cargo** (for Rust package management)
- **Required Python Packages**:
  - torch: 2.4.0
  - torchvision: 0.19.0
  - torchmetrics: 1.4.1
  - ecdsa: 0.19.0
  - numpy: 2.0.1

### Installing Python and Rust:

1. To install Python and pip, run the following commands:
   ```bash
   sudo apt-get update
   sudo apt-get install -y python3 python3-pip
   ```

2. To install the required Python packages with their specified versions, run the following command:
   ```bash
   pip3 install -r requirements.txt
   ```

3. To install Rust and Cargo, run the following commands:
   ```bash
   curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
   . "$HOME/.cargo/env"
   rustup toolchain install nightly-2023-06-26
   rustup default nightly-2023-06-26  # Set this version globally
   rustc --version  # Verify the Rust compiler version
   cargo --version  # Verify the Cargo version
   ```

## How to Run

1. **Generate Pre-computed Table**

   If you do not already have the pre-computed table, you need to generate it by running:

   ```bash
   ./script.sh -b
   ```

   If you already have the pre-computed table, you can skip this step.

2. **Run the CNN Networks or LeNet or Convolutional Layers**

   You can choose between running the CNN networks, LeNet, or convolutional layers:

   - **Run CNN Networks**:
     ```bash
     ./script.sh -c -A  # For CNN network A
     ./script.sh -c -B  # For CNN network B
     ./script.sh -c -C  # For CNN network C
     ./script.sh -c -D  # For CNN network D
     ./script.sh -c -E  # For CNN network E
     ```

   - **Run LeNet**:
     ```bash
     ./script.sh -l
     ```

   - **Run Convolutional Layers**:
     ```bash
     ./script.sh -d < filter_size: 3|5|7 > < input_size: 32|64|128|256 >
     ```

     For example:
     ```bash
     ./script.sh -d 3 32
     ```

3. **Generate Proofs**

   After generating the witnesses, run the Rust code to generate proofs for point additions and point multiplications:

   First, navigate to the `src/proof_generation/vPIN_proof_generation/src` directory, and then run the following command:

   ```bash
   cargo run main.rs
   ```

4. **Check Accuracy**

   To assess the accuracy, you can run:

   ```bash
   ./script.sh -a
   ```

## Acknowledgments

This project uses the [Spartan repository](https://github.com/microsoft/Spartan), located in the `src/proof_generation/spartan/` directory, as a library for generating proofs for our witnesses. We have modified certain files within the Spartan repository to integrate it with the vPIN framework. We extend our sincere gratitude to the contributors of the Spartan project for providing a robust foundation upon which we could build our proof generation system.

Additionally, we acknowledge the [ezDPS implementation](https://github.com/vt-asaplab/ezDPS) for providing the `commitment_test.rs` file, located in `src/proof_generation/vPIN_proof_generation/src/`. This file has been essential in creating commitments to auxiliary witnesses in our project.

