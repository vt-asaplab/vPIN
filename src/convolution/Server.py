import socket
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from ecdsa.ellipticcurve import CurveFp, Point
import random
import numpy as np
import pickle
from decimal import Decimal, getcontext
import hmac
import hashlib
import os
import sys
import multiprocessing
import json

MultiCoreFeature = 1
num_processes = 8

getcontext().prec = 256

points_mult = []
weights_array = []
point_one_Add = []
point_two_Add = []

IP = socket.gethostbyname("")

script_dir = os.path.dirname(os.path.abspath(__file__))

PORT = int(sys.argv[2])
ADDR = (IP, PORT)
FORMAT = "utf-8"
SIZE = 256000

def receiveParameters(conn):
    message1 = conn.recv(SIZE)
    conn.send("curveGenerator received successfully".encode(FORMAT))
    message2 = conn.recv(SIZE)
    conn.send("h received successfully".encode(FORMAT))
    message3 = conn.recv(SIZE)
    conn.send("curveOrder_data received successfully".encode(FORMAT))

    curveGenerator = pickle.loads(message1)
    h = pickle.loads(message2)
    curveOrder = pickle.loads(message3)
    
    return curveGenerator, h, curveOrder

def startServer():
    print("\n[STARTING] Server is starting")
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.bind(ADDR)
    server.listen()
    conn, address = server.accept()
    print(f"Server: [NEW CLIENT CONNECTION] {address} connected.")

    return server, conn, address

def receive_data_in_chunks(conn):
    message = []

    msgLength = conn.recv(SIZE)
    conn.send("length received".encode(FORMAT))
    totalSize = int(msgLength.decode(FORMAT))

    chunkSize = 30000
    numChunks = (totalSize + chunkSize - 1) // chunkSize

    for i in range (0, numChunks):
        msg = conn.recv(SIZE)
        conn.send(f"encryptedValue_c1 part {i} received successfully".encode(FORMAT))
        message.append(msg)

    finalMsg = b''.join(message)

    return finalMsg

def pf(secret_key, message):
    counter = str(message).encode('utf-8')
    h = hmac.new(secret_key, counter, hashlib.sha256)
    result = h.digest()[:16]
    integer_result = int(result.hex(), 16)

    return integer_result

def compute_range(start, end, secret_key, B_prime, lock, identityPoint, final_result):
    temp_sum = identityPoint

    for i in range(start, end):
        random_number = pf(secret_key, i)
        temp = random_number * B_prime[i]
        temp_sum += temp

    final_result.append(temp_sum)

def receiveEncryptedImage(conn, type):

    finalMsg = receive_data_in_chunks(conn)
    encryptedValue_c1 = pickle.loads(finalMsg)

    finalMsg = receive_data_in_chunks(conn)
    encryptedValue_c2 = pickle.loads(finalMsg)

    return encryptedValue_c1, encryptedValue_c2

def rLCL(input, secret_key, identityPoint, curveBaseField, type):
    result_left = identityPoint

    if type == 0:
        length = len(input)
        result_left = identityPoint

        for i in range(len(input)):
            random_number = pf(secret_key, i)
            result_left += random_number * input[i]

    else: 
        length = len(input[0])
        
        for i in range(length):
            random_number = pf(secret_key, i)
            temp = random_number * input[0][i]
            result_left = result_left + temp

    return result_left

def rLCR(input_data, weight_matrix, secret_key, identityPoint, curveBaseField, type):
    if type == 0:
        B_prime = weight_matrix

        kernel = input_data
        result = identityPoint

        if MultiCoreFeature == 0: # single core
            for i in range(B_prime.shape[0]):
                random_number = pf(secret_key, i)
                temp = random_number * B_prime[i]
                result = result + temp

        else: # MultiCoreFeature == 1
            lock = multiprocessing.Lock()
            final_result = multiprocessing.Manager().list()

            indices_per_process = B_prime.shape[0] // num_processes
            processes = []

            for i in range(num_processes):
                start_index = i * indices_per_process
                end_index = (i + 1) * indices_per_process if i < num_processes - 1 else B_prime.shape[0]
                process = multiprocessing.Process(target=compute_range, args=(start_index, end_index, secret_key, B_prime, lock, identityPoint, final_result))
                processes.append(process)
                process.start()

            for process in processes:
                process.join()

            result = identityPoint
            for temp_sum in final_result:
                result += temp_sum

        result_right = identityPoint
        
        for i in range(kernel.shape[0]):
            points_mult.append(result.T[i])
            weights_array.append(kernel[i])

            temp = kernel[i] * result.T[i]

            if i == 0:
                result_right = temp
            else:            
                point_one_Add.append(result_right)
                point_two_Add.append(temp)

                result_right = result_right + temp

    else:
        weight_matrix_transpose = weight_matrix.T
        for i in range(weight_matrix_transpose.shape[0]):
            random_number = pf(secret_key, i)
            temp = []
            for k in range(weight_matrix_transpose.shape[1]):

                mul_out = Decimal(str(random_number)) * Decimal(str(weight_matrix_transpose[i][k]))
                if mul_out > Decimal(str(curveBaseField)):
                    mul_out = mul_out % Decimal(str(curveBaseField))

                temp.append(mul_out)
                assert mul_out == temp[k], "The values are not equal"
            
            if i == 0:
                result = temp   
            else:
                for ii in range(weight_matrix_transpose.shape[1]):
                    add_out = result[ii] + temp[ii]
                    if add_out > Decimal(str(curveBaseField)):
                        add_out = add_out % Decimal(str(curveBaseField))

                    result[ii] = add_out

        result = np.array(result)
        result_right = identityPoint

        for i in range(input_data.shape[1]):
            points_mult.append(input_data[0][i])
            weights_array.append(result.T[i])

            temp = input_data[0][i] * int(result.T[i])

            if i == 0:
                result_right = temp
            else:
                point_one_Add.append(result_right)
                point_two_Add.append(temp)
                result_right = result_right + temp 

    return result_right

def myConv2d(input_data, filter_weights, identityPoint, curveBaseField, type, padding_size=0, stride=1):
    # type == 0 -> original data
    # type == 1 -> encrypted data
    input_height, input_width = input_data.shape
    filter_height, filter_width = filter_weights.shape
    
    if type == 0:
        padded_input_data = np.pad(input_data, padding_size, mode='constant')
    else:
        padded_input_data = np.pad(input_data, padding_size, mode='constant', constant_values=identityPoint)

    output_height = (input_height + 2*padding_size - filter_height) // stride + 1
    output_width = (input_width + 2*padding_size - filter_width) // stride + 1
    
    if type == 0:
        output_data = np.zeros((output_height, output_width))
    else:
        output_data = np.empty((output_height, output_width), dtype=object)

    if type == 0:
        for i in range(output_height):
            for j in range(output_width):
                output_data[i,j] = np.sum(padded_input_data[i*stride:i*stride+filter_height, j*stride:j*stride+filter_width] * filter_weights)
    else:
        window_list = []

        for i in range(output_height):
            for j in range(output_width):
                temp_list = []
                for ii in range(filter_height):
                    for jj in range(filter_width):
                        if ii == 0 and jj == 0:
                            sum_value = padded_input_data[i*stride+ii, j*stride+jj] * filter_weights[ii, jj]
                        else:
                            tempValue = (padded_input_data[i*stride+ii, j*stride+jj] * filter_weights[ii, jj])
                            sum_value = sum_value + tempValue

                        window_value = padded_input_data[i*stride+ii, j*stride+jj]
                        temp_list.append(window_value)

                window_list.append(temp_list)

                output_data[i, j] = sum_value

        window_array = np.array(window_list)
        secret_key = os.urandom(32)
        output_data_flatten = output_data.flatten()

        result_left = rLCL(output_data_flatten, secret_key, identityPoint, curveBaseField, 0)

        result_right = rLCR(filter_weights.flatten(), window_array, secret_key, identityPoint, curveBaseField, 0)

        assert result_left == result_right, "The values are not equal"

    return output_data

def callConv2_ciphertext(images, identityPoint, curveBaseField, filter_weights):
    batchSize = images.shape[0]
    numChannels = images.shape[1]
    height = images.shape[2]
    width = images.shape[3]

    filter_size = filter_weights.shape[0]
    output_height = height - filter_size + 3
    output_width = width - filter_size + 3

    output_numpy = np.empty((batchSize, numChannels, output_height, output_width), dtype=object)

    for i in range(batchSize):
        for j in range(numChannels): 
            output_data = myConv2d(images[i][j], filter_weights, identityPoint, curveBaseField, 1, padding_size=1, stride=1)
            output_numpy[i][j] = output_data

    return output_numpy

def conv2_ciphertext(encryptedValue_c1, encryptedValue_c2,identityPoint, curveBaseField, filter_weights):
    print("\n**************************************************")
    print("Server: First conv. layer started!")

    outputConv2Ciphertext_c1 = callConv2_ciphertext(encryptedValue_c1, identityPoint, curveBaseField, filter_weights)
    outputConv2Ciphertext_c2 = callConv2_ciphertext(encryptedValue_c2, identityPoint, curveBaseField, filter_weights)

    print("Server: First conv. layer finished!")
    print("**************************************************")

    return outputConv2Ciphertext_c1, outputConv2Ciphertext_c2

def intToByte(integer):
    byte_array = []
    while integer > 0:
        byte_array.append(integer & 255)
        integer >>= 8

    byte_array.extend([0] * (32 - len(byte_array)))

    return byte_array

def convertFormatForRust_pointMult():
    point_mult_px_byte = []
    point_mult_py_byte = []

    for item in points_mult:
        point_mult_px_byte.append(intToByte(item.x()))
        point_mult_py_byte.append(intToByte(item.y()))

    # save in JSON format for RUST
    my_array = np.array(weights_array, dtype=object)
    my_array = my_array.tolist()
    my_array = [str(x) for x in my_array]
    file_path = os.path.join(script_dir, "rust_files", "pointMult", "weight.json")
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, 'w') as file:
        json.dump(my_array, file)

    my_array1 = np.array(point_mult_px_byte, dtype=np.int64)
    my_array1 = my_array1.tolist()        
    file_path = os.path.join(script_dir, "rust_files", "pointMult", "point_mult_px_byte.json")
    os.makedirs(os.path.dirname(file_path), exist_ok=True)    
    with open(file_path, 'w') as file:
        json.dump(my_array1, file)

    my_array2 = np.array(point_mult_py_byte, dtype=np.int64)
    my_array2 = my_array2.tolist()   
    file_path = os.path.join(script_dir, "rust_files", "pointMult", "point_mult_py_byte.json")
    os.makedirs(os.path.dirname(file_path), exist_ok=True)             
    with open(file_path, 'w') as file:
        json.dump(my_array2, file)

def convertFormatForRust_pointAdd():
    point_add_px_byte = []
    point_add_py_byte = []    
    point_add_rx_byte = []
    point_add_ry_byte = []
    point_add_rz_byte = [] # 0 or 1 -> indicating infinity point    

    for item in point_one_Add:
        point_add_px_byte.append(intToByte(item.x()))
        point_add_py_byte.append(intToByte(item.y()))

    infinityPoint = point_one_Add[0] * 0
    for item in point_two_Add:    
        if item == infinityPoint:
            point_add_rz_byte.append(1)
            point_add_rx_byte.append(intToByte(0))
            point_add_ry_byte.append(intToByte(0))
        else:
            point_add_rz_byte.append(0)
            point_add_rx_byte.append(intToByte(item.x()))
            point_add_ry_byte.append(intToByte(item.y()))

    # save in JSON format for RUST
    my_array3 = np.array(point_add_px_byte, dtype=np.int64)
    my_array3 = my_array3.tolist()    
    file_path = os.path.join(script_dir, "rust_files", "pointAdd", "point_add_px_byte.json")
    os.makedirs(os.path.dirname(file_path), exist_ok=True)                  
    with open(file_path, 'w') as file:
        json.dump(my_array3, file)

    my_array4 = np.array(point_add_py_byte, dtype=np.int64)
    my_array4 = my_array4.tolist() 
    file_path = os.path.join(script_dir, "rust_files", "pointAdd", "point_add_py_byte.json")
    os.makedirs(os.path.dirname(file_path), exist_ok=True)             
    with open(file_path, 'w') as file:
        json.dump(my_array4, file)

    my_array5 = np.array(point_add_rx_byte, dtype=np.int64)
    my_array5 = my_array5.tolist()   
    file_path = os.path.join(script_dir, "rust_files", "pointAdd", "point_add_rx_byte.json")
    os.makedirs(os.path.dirname(file_path), exist_ok=True)            
    with open(file_path, 'w') as file:
        json.dump(my_array5, file)

    my_array6 = np.array(point_add_ry_byte, dtype=np.int64)
    my_array6 = my_array6.tolist()        
    file_path = os.path.join(script_dir, "rust_files", "pointAdd", "point_add_ry_byte.json")
    os.makedirs(os.path.dirname(file_path), exist_ok=True)        
    with open(file_path, 'w') as file:
        json.dump(my_array6, file)

    my_array7 = np.array(point_add_rz_byte, dtype=np.int64)
    my_array7 = my_array7.tolist()      
    file_path = os.path.join(script_dir, "rust_files", "pointAdd", "point_add_rz_byte.json")
    os.makedirs(os.path.dirname(file_path), exist_ok=True)              
    with open(file_path, 'w') as file:
        json.dump(my_array7, file)

def inferenceCNN(curveBaseField, curveGenerator, curveOrder, h, identityPoint, server, conn, filter_weights):

    encryptedValue_c1, encryptedValue_c2 = receiveEncryptedImage(conn, 0)

    print("\n**************************************************")
    print("Server: Encrypted data sample received.")
    print("Server: Performing inference on encrypted data...")
    print("Server: Filter size:", filter_weights.shape)
    print("**************************************************")

    outputConv2Ciphertext_c1, outputConv2Ciphertext_c2 = conv2_ciphertext(encryptedValue_c1, encryptedValue_c2, identityPoint, curveBaseField, filter_weights)

    server.close()

    convertFormatForRust_pointMult()
    convertFormatForRust_pointAdd()
    print("Server: The witnesses are saved in a file for generating proof with Rust")


def main():
    filter_size = int(sys.argv[1])
    server, conn, address = startServer()
    conn.send("Welcome".encode(FORMAT))

    curveBaseField = 7237005577332262213973186563042994240857116359379907606001950938285454250989
    curveGenerator, h, curveOrder = receiveParameters(conn)
    identityPoint = 0 * h

    #load filter and filter size base on the version = int(sys.argv[1])
    if filter_size == 3:
        filter_weights = np.array([[1, 0, 1], 
                               [2, 0, 2], 
                               [1, 0, 1]]) # pre-trained conv. filter
    elif filter_size == 5:
        filter_weights = np.array([[1, 0, 0, 0, 1],
                                [0, 0, 0, 0, 0],
                                [2, 0, 0, 0, 2], 
                                [0, 0, 0, 0, 0], 
                                [1, 0, 0, 0, 1]]) # pre-trained conv. filter
    elif filter_size == 7:
        filter_weights = np.array([[1, 0, 0, 2, 0, 0, 1], 
                                [0, 0, 0, 0, 0, 0, 0], 
                                [0, 0, 0, 0, 0, 0, 0], 
                                [0, 0, 0, 0, 0, 0, 0], 
                                [0, 0, 0, 0, 0, 0, 0], 
                                [0, 0, 0, 0, 0, 0, 0], 
                                [1, 0, 0, 2, 0, 0, 1]]) # pre-trained conv. filter
    else:
        raise ValueError("Invalid filter size. Please choose 3, 5, or 7.")

    inferenceCNN(curveBaseField, curveGenerator, curveOrder, h, identityPoint, server, conn, filter_weights)


if __name__ == "__main__":
    main()