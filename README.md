# Privacy-Preserving Verifiable Neural Network Inference Service

This repository contains the full implementation of [**vPIN** paper](https://arxiv.org/pdf/2411.07468) (accepted to ACSAC 2024).

**Warning**: This code is a proof-of-concept prototype and is not ready for production use.

## Awards
This work has been awarded all three available badges—Artifact Available, Reviewed, and Reproducible—by the ACSAC 2024 reviewers.

<img width="458" alt="Image" src="https://github.com/user-attachments/assets/9ee47992-d04c-4906-85e7-b5633b2df132" />

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

## Installation

### Installing Python:

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

## Configuration

### Port Configuration

- Default ports:
  - 35000–35006: for individual tests
  - 36000–36004: for batch CNN experiments
  - 37000–37011: for batch convolution experiments
- Modify lines 29–39 in the `script.sh` to customize these ports.

### Output Configuration

- `QUIET_MODE` controls the script's output verbosity:
  - **QUIET_MODE=1** (default): Suppresses terminal output, directing all output to log files.
  - **QUIET_MODE=0**: Enables verbose terminal output for detailed real-time monitoring.
- To switch modes, modify `QUIET_MODE` in line 46 of the `script.sh`.

## How to Run

1. **Generate Pre-computed Table**

     *Note: If you already have the pre-computed table, you can skip this step.*

    To generate table, run the following command:
     ```bash
     ./script.sh -b
     ```
    - **Resource Requirements and Duration**:
      - **RAM**: Approximately ~**3 GB**
      - **Time**: Approximately ~**50 minutes**
      - **Storage**: Approximately ~**230 MB**     

3. **Run Experiments and Generate Proofs**

   Choose between running CNN networks, the LeNet model, or convolutional layers. Each script generates witnesses in the `vPIN/src/rust_files` directory, executes Rust code to generate proofs, and outputs the proving time, verification time, and proof size.

   - **Run CNN Networks**:
     ```bash
     ./script.sh -c -A  # Run CNN network A and generate proofs (default port: 35000)
     ./script.sh -c -B  # For CNN network B and generate proofs (default port: 35001)
     ./script.sh -c -C  # For CNN network C and generate proofs (default port: 35002)
     ./script.sh -c -D  # For CNN network D and generate proofs (default port: 35003)
     ./script.sh -c -E  # For CNN network E and generate proofs (default port: 35004)

     # Run all CNN networks sequentially:
     ./script.sh -c -t  # Run all CNN networks sequentially and generate proofs (default ports: 36000-36004)
     ```

     - **Resource Requirements and Duration**:
       - **RAM**: Between **3 to 16 GB**, depending on the network type
       - **Time**: Between **6 to 75 minutes**, depending on the network type
     
   - **Run LeNet**:
     ```bash
     ./script.sh -l  # Run LeNet model and generate proofs (default port: 35005)
     ```
     - **Resource Requirements and Duration**:
       - **RAM**: ~**230 GB**
       - **Time**: ~**4 hours**

   - **Run Convolutional Layers**:
     ```bash
     ./script.sh -d < filter_size: 3|5|7 > < input_size: 32|64|128|256 > | -d -t

     # Run all convolution experiments sequentially:  
     ./script.sh -d -t  # Run all convolution experiments sequentially and generate proofs (default ports: 37000-37011)
     ```

     For example:
     ```bash
     ./script.sh -d 3 32  # Example: Filter size 3, input size 32x32 (default port: 35006)
     ./script.sh -d 5 64  # Example: Filter size 5, input size 64x64 (default port: 35006)
     ```

     - **Resource Requirements and Duration**:
       - **RAM**: Between **2 to 5 GB**, depending on the input size and filter size
       - **Time**: Between **2 minutes to 4 hours**, depending on the input size and filter size


4. **Check Accuracy**

   To assess the accuracy, you can run:

   ```bash
   ./script.sh -a
   ```
     - **Resource Requirements and Duration**:
       - **RAM**: < **1GB**
       - **Time**: ~**50 minutes**

## Artifact Documentation

For a detailed description of how to reproduce the results presented in the paper, please refer to our [Artifact Documentation](/Documents/ACSAC_2024_Artifact_Documentation_Privacy-Preserving_Verifiable_Neural_Network_Inference_Service.pdf). 

## Acknowledgments

This project uses the [Spartan repository](https://github.com/microsoft/Spartan), located in the `src/proof_generation/spartan/` directory, as a library for generating proofs for our witnesses. We have modified certain files within the Spartan repository to integrate it with the vPIN framework. We extend our sincere gratitude to the contributors of the Spartan project for providing a robust foundation upon which we could build our proof generation system.

Additionally, we acknowledge the [ezDPS implementation](https://github.com/vt-asaplab/ezDPS) for providing the `commitment_test.rs` file, located in `src/proof_generation/vPIN_proof_generation/src/`. This file has been essential in creating commitments to auxiliary witnesses in our project.

## Citing

If you use this repository or build upon our work, we would appreciate it if you cite our paper using the following BibTeX entry:

```bibtex
@inproceedings{vPIN2024,
  author={Riasi, Arman and Guajardo, Jorge and Hoang, Thang},
  booktitle={2024 Annual Computer Security Applications Conference (ACSAC)}, 
  title={Privacy-Preserving Verifiable Neural Network Inference Service}, 
  year={2024},
  volume={},
  number={},
  pages={683-698},
  doi={10.1109/ACSAC63791.2024.00063}}
}

