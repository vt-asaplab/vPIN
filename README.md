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

### Installing Python and Required Packages:

1. **Update the package list and install Python and pip**:
   ```bash
   sudo apt-get update
   sudo apt-get install -y python3 python3-pip
   ```

2. **Install the required Python packages**:
   ```bash
   pip3 install -r requirements.txt
   ```

### Installing Rust and Cargo

1. **Install Rust and Cargo**:
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

     *Note: If you already have the pre-computed table, you can skip this step.*

    To generate table, run the following command:
     ```bash
     ./script.sh -b
     ```
    - **Resource Requirements and Duration**:
      - **RAM**: Approximately **3 GB**
      - **Time**: Approximately **50 minutes**
      - **Storage**: Approximately **230 MB**     

3. **Run Experiments and Generate Proofs**

   Choose between running CNN networks, the LeNet model, or convolutional layers. Each script generates witnesses in the `vPIN/src/rust_files` directory, executes Rust code to generate proofs, and outputs the proving time, verification time, and proof size.

   - **Run CNN Networks**:
     ```bash
     ./script.sh -c -A  # Run CNN network A and generate proofs
     ./script.sh -c -B  # For CNN network B and generate proofs
     ./script.sh -c -C  # For CNN network C and generate proofs
     ./script.sh -c -D  # For CNN network D and generate proofs
     ./script.sh -c -E  # For CNN network E and generate proofs

     # Run all CNN networks sequentially:
     ./script.sh -c -t  # Run all CNN networks sequentially and generate proofs
     ```

     - **Resource Requirements and Duration**:
       - **RAM**: Between **3 to 16 GB**, depending on the network type
       - **Time**: Between **6 to 75 minutes**, depending on the network type
     
   - **Run LeNet**:
     ```bash
     ./script.sh -l  # Run LeNet model and generate proofs
     ```
     - **Resource Requirements and Duration**:
       - **RAM**: **230 GB**
       - **Time**: **4 hours**

   - **Run Convolutional Layers**:
     ```bash
     ./script.sh -d < filter_size: 3|5|7 > < input_size: 32|64|128|256 > | -d -t

     # Run all convolution experiments sequentially:  
     ./script.sh -d -t  # Run all convolution experiments sequentially and generate proofs
     ```

     For example:
     ```bash
     ./script.sh -d 3 32  # Example: Filter size 3, input size 32x32
     ./script.sh -d 5 64  # Example: Filter size 5, input size 64x64
     ```

     - **Resource Requirements and Duration**:
       - **RAM**: Between **xx to xx GB**, depending on the input size and filter size
       - **Time**: Between **xx to xx minutes**, depending on the input size and filter size


4. **Check Accuracy**

   To assess the accuracy, you can run:

   ```bash
   ./script.sh -a
   ```
     - **Resource Requirements and Duration**:
       - **RAM**: < **1GB**
       - **Time**: **50 minutes**

## Acknowledgments

This project uses the [Spartan repository](https://github.com/microsoft/Spartan), located in the `src/proof_generation/spartan/` directory, as a library for generating proofs for our witnesses. We have modified certain files within the Spartan repository to integrate it with the vPIN framework. We extend our sincere gratitude to the contributors of the Spartan project for providing a robust foundation upon which we could build our proof generation system.

Additionally, we acknowledge the [ezDPS implementation](https://github.com/vt-asaplab/ezDPS) for providing the `commitment_test.rs` file, located in `src/proof_generation/vPIN_proof_generation/src/`. This file has been essential in creating commitments to auxiliary witnesses in our project.

