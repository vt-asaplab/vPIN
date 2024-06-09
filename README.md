# Implementation of "Privacy-Preserving Verifiable Neural Network Inference Service"

## Code Structure
- **src/cnn_networks/Pre_computed_table/**: Contains `baby-step-giant-step.py` for generating the pre-computed table.
- **src/cnn_networks/**: Contains `server.py` and `client.py` files for five different CNN networks used to generate results for Figure 2.
- **src/convolution/**: Contains `server.py` and `client.py` files for convolutional layer with different filter sizes and input sizes to generate results for Figure 3.
- **src/LeNet/**: Contains `server.py` and `client.py` files for LeNet model used to generate results for Table 2.
- **src/accuracy/**: Contains `train_test_lenet5.py` for assessing accuracy.
- **src/proof_generation/vPIN_proof_generation/src/**: Contains files for generating proof for point additions and point multiplications.

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

   After generating the witnesses, you can run the Rust code to generate proofs for point additions and point multiplications:

   ```bash
   cargo run main.rs
   ```

4. **Check Accuracy**

   To assess the accuracy, you can run:

   ```bash
   ./script.sh -a
   ```
