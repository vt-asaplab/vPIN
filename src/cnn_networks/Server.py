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

MODEL_PATHS = {
    1: {
        "weight_fc1": script_dir+"/Pre_trained_model/weight_fc1_64_16.npy",
        "bias_fc1": script_dir+"/Pre_trained_model/bias_fc1_16.npy",
        "weight_fc2": script_dir+"/Pre_trained_model/weight_fc2_16_10.npy"
    },
    2: {
        "weight_fc1": script_dir+"/Pre_trained_model/weight_fc1_64_32.npy",
        "bias_fc1": script_dir+"/Pre_trained_model/bias_fc1_32.npy",
        "weight_fc2": script_dir+"/Pre_trained_model/weight_fc2_32_10.npy"
    },
    3: {
        "weight_fc1": script_dir+"/Pre_trained_model/weight_fc1_256_16.npy",
        "bias_fc1": script_dir+"/Pre_trained_model/bias_fc1_16.npy",
        "weight_fc2": script_dir+"/Pre_trained_model/weight_fc2_16_10.npy"
    },
    4: {
        "weight_fc1": script_dir+"/Pre_trained_model/weight_fc1_256_32.npy",
        "bias_fc1": script_dir+"/Pre_trained_model/bias_fc1_32.npy",
        "weight_fc2": script_dir+"/Pre_trained_model/weight_fc2_32_10.npy"
    },
    5: {
        "weight_fc1": script_dir+"/Pre_trained_model/weight_fc1_256_64.npy",
        "bias_fc1": script_dir+"/Pre_trained_model/bias_fc1_64.npy",
        "weight_fc2": script_dir+"/Pre_trained_model/weight_fc2_64_10.npy"
    }
}

KERNEL_STRIDE = {
    1: (4, 4),
    2: (4, 4),
    3: (2, 2),
    4: (2, 2),
    5: (2, 2)
}

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

def min_max_scaling(images):
    
    min_val = np.min(images)
    max_val = np.max(images)
    normalized_image = (images - min_val) / (max_val - min_val)
    normalized_image = np.clip(normalized_image, a_min=0.001, a_max=0.9999999)

    return normalized_image

def realNumbersToFixedPointRepresentation(Input, type, bits):
    if type == 1:
        scale_factor = 2 ** bits  # x bits for the fractional part
        fixed_point = (Input * scale_factor).astype(np.int32)
    else:
        scale_factor = 2 ** bits  # x bits for the fractional part
        fixed_point = (Input) * scale_factor

    return fixed_point

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

def compute_range(start, end, secret_key, B_prime, lock, identityPoint, final_result):
    temp_sum = identityPoint

    for i in range(start, end):
        random_number = pf(secret_key, i)
        temp = random_number * B_prime[i]
        temp_sum += temp

    final_result.append(temp_sum)

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

def callConv2_ciphertext(images, identityPoint, curveBaseField):
    filter_weights = np.array([[1, 0, 1], [2, 0, 2], [1, 0, 1]]) # pre-trained conv. filter

    batchSize = images.shape[0]
    numChannels = images.shape[1]
    height = images.shape[2]
    width = images.shape[3]

    output_numpy = np.empty((batchSize, numChannels, height, width), dtype=object)

    for i in range(batchSize):
        for j in range(numChannels): 
            output_data = myConv2d(images[i][j], filter_weights, identityPoint, curveBaseField, 1, padding_size=1, stride=1)
            output_numpy[i][j] = output_data

    return output_numpy

def realNumbersToFixedPointRepresentation(Input, type, bits):
    if type == 1:
        scale_factor = 2 ** bits  # x bits for the fractional part
        fixed_point = (Input * scale_factor).astype(np.int32)
    else:
        scale_factor = 2 ** bits  # x bits for the fractional part
        fixed_point = (Input) * scale_factor

    return fixed_point

def myAvgPool2d(flag , input_data, identityPoint, type1, type2, kernel_size, stride):
    input_height, input_width = input_data.shape
    
    output_height = (input_height - kernel_size) // stride + 1
    output_width = (input_width - kernel_size) // stride + 1
    
    if type1 == 0:
        output_data = np.zeros((output_height, output_width))
    else:
        output_data = np.empty((output_height, output_width), dtype=object)

    if type1 == 0 and type2 == 0:
        for i in range(output_height):
            for j in range(output_width):
                output_data[i,j] = np.mean(input_data[i*stride:i*stride+kernel_size, j*stride:j*stride+kernel_size])
    else:
        for i in range(output_height):
            for j in range(output_width):
                sum_value = 0
                for ii in range(kernel_size):
                    for jj in range(kernel_size):
                        if ii == 0 and jj == 0:
                            sum_value = input_data[i*stride+ii, j*stride+jj]
                            if (sum_value == 0):
                                sum_value = identityPoint   
                        else:
                            tempValue = input_data[i*stride+ii, j*stride+jj]
                            if (tempValue == 0):
                                tempValue = identityPoint

                            if flag == 1:
                                point_one_Add.append(sum_value)
                                point_two_Add.append(tempValue)

                            sum_value = sum_value + tempValue 
                
                denominator = kernel_size ** 2

                fixed_point = int(realNumbersToFixedPointRepresentation((1/denominator), 2, 10))

                if flag == 1:
                    None
                output_data[i,j] = sum_value * fixed_point

    return output_data

def pf(secret_key, message):
    counter = str(message).encode('utf-8')
    h = hmac.new(secret_key, counter, hashlib.sha256)
    result = h.digest()[:14]
    integer_result = int(result.hex(), 16)

    return integer_result

def callAvgPool2d_ciphertext(image, identityPoint, kernelSize, stride):
    batch_size, num_channels, input_height, input_width = image.shape
    output_height = input_height // kernelSize
    output_width = input_width // kernelSize

    output_numpy = np.zeros((batch_size, num_channels, output_height, output_width), dtype=object)

    for i in range(batch_size):
        for j in range(num_channels):
            input_data = image[i][j]
            output_data = myAvgPool2d(1, input_data, identityPoint, 1, 0, kernelSize, stride)
            output_numpy[i][j] = output_data

    return output_numpy

def flatten(x):
    input_size = x.shape[1] * x.shape[2] * x.shape[3]
    x = x.reshape(-1, input_size)
    return x

def FCLayer(input_data, weight_matrix, bias_vector, flag, identityPoint, curveBaseField):
            
    if flag == 0:
        output_data = np.zeros((input_data.shape[0], weight_matrix.shape[1]))

        for i in range(input_data.shape[0]):
            for j in range(weight_matrix.shape[1]):
                for k in range(input_data.shape[1]):
                    temp = input_data[i, k] * weight_matrix[k, j]
                    output_data[i, j] = output_data[i, j] + temp
                output_data[i, j] = output_data[i, j] + bias_vector[j]
    else:
        C = np.matmul(input_data, weight_matrix)

        output_data = np.empty((input_data.shape[0], weight_matrix.shape[1]), dtype=object)

        for j in range(weight_matrix.shape[1]):
            point_one_Add.append(C[0][j])
            point_two_Add.append(bias_vector[j])
            output_data[0, j] = C[0][j] + bias_vector[j]

        secret_key = os.urandom(32)

        result_left = rLCL(C, secret_key, identityPoint, curveBaseField, 1)
        
        result_right = rLCR(input_data, weight_matrix, secret_key, identityPoint, curveBaseField, 1)

        assert result_left == result_right, "The values are not equal"

    return output_data

def encrypt(tensorInputPlaintext, curveOrder, curveGenerator, h, randomValueRList):
    #---Encryption
    randomValueR = random.randrange(1, curveOrder-1) #r
    message = int(tensorInputPlaintext)
    c1 = randomValueR * curveGenerator
    c2_1 = message * curveGenerator
    c2_2 = randomValueR * h
    c2 = c2_1 + c2_2

    return c1, c2

def encryptBias(Input, curveOrder, curveGenerator, h):
    randomValueRList2 = []
    encryptBias_c1 = np.empty((Input.shape[0]), dtype=object)
    encryptBias_c2 = np.empty((Input.shape[0]), dtype=object)
    for i in range(Input.shape[0]):
        c1, c2 = encrypt(Input[i], curveOrder, curveGenerator, h, randomValueRList2)
        encryptBias_c1[i] = c1 #c1
        encryptBias_c2[i] = c2 #c2

    return encryptBias_c1, encryptBias_c2

def send_data_in_chunks(data, chunkSize, conn, size):
    conn.sendall(str(len(data)).encode(FORMAT))
    msg = conn.recv(size).decode(FORMAT)

    for i in range(0, len(data), chunkSize):
        chunk = data[i:i + chunkSize]
        conn.sendall(chunk)
        msg = conn.recv(size).decode(FORMAT)

def interactionClient(conn, c1, c2, type):
    encryptedValue_c1 = pickle.dumps(c1)
    encryptedValue_c2 = pickle.dumps(c2)

    chunk_size = 30000
    send_data_in_chunks(encryptedValue_c1, chunk_size, conn, SIZE)
    send_data_in_chunks(encryptedValue_c2, chunk_size, conn, SIZE)

def load_model_parameters(version):
    """Load model parameters based on the specified version."""
    try:
        params = MODEL_PATHS[version]
        weight_fc1 = np.load(params["weight_fc1"])
        bias_fc1 = np.load(params["bias_fc1"])
        weight_fc2 = np.load(params["weight_fc2"])
        bias_fc2 = np.load(script_dir+"/Pre_trained_model/bias_fc2_10.npy")
        return weight_fc1, bias_fc1, weight_fc2, bias_fc2
    except FileNotFoundError as e:
        raise RuntimeError(f"Model file not found: {e.filename}")

def conv2_ciphertext(encryptedValue_c1, encryptedValue_c2,identityPoint, curveBaseField):
    print("\n**************************************************")
    print("Server: First conv. layer started!")

    outputConv2Ciphertext_c1 = callConv2_ciphertext(encryptedValue_c1, identityPoint, curveBaseField)
    outputConv2Ciphertext_c2 = callConv2_ciphertext(encryptedValue_c2, identityPoint, curveBaseField)

    print("Server: First conv. layer finished!")
    print("**************************************************")

    return outputConv2Ciphertext_c1, outputConv2Ciphertext_c2

def avgPool_ciphertext(encryptedValue_c1, encryptedValue_c2, identityPoint, kernelSize, stride):
    print("\n**************************************************")
    print("Server: First AvgPooling started!")

    outputAvgPool2dCiphertext_c1 = callAvgPool2d_ciphertext(encryptedValue_c1, identityPoint, kernelSize, stride)
    outputAvgPool2dCiphertext_c2 = callAvgPool2d_ciphertext(encryptedValue_c2, identityPoint, kernelSize, stride)

    print("Server: First AvgPooling finished!")
    print("**************************************************")

    return outputAvgPool2dCiphertext_c1, outputAvgPool2dCiphertext_c2

def flattening(outputAvgPool2dCiphertext_c1, outputAvgPool2dCiphertext_c2):
    print("\n**************************************************")
    print("Server: Flattening started!")    
    outputCiphertext_c1_flat = flatten(outputAvgPool2dCiphertext_c1)
    outputCiphertext_c2_flat = flatten(outputAvgPool2dCiphertext_c2)
    
    print("Server: Flattening finished!")             
    print("**************************************************")

    return outputCiphertext_c1_flat, outputCiphertext_c2_flat

def FC1(weight_fc1, bias_fc1, curveOrder, curveGenerator, h, encryptedValue_c1, encryptedValue_c2, identityPoint, curveBaseField):

    print("\n**************************************************")
    print("Server: FC1 started!")

    weight_fc1_FixedPoint = realNumbersToFixedPointRepresentation(weight_fc1, 1, 16)
    bias_fc1_FixedPoint = realNumbersToFixedPointRepresentation(bias_fc1, 1, 16)

    outputBias_Fc1_c1, outputBias_Fc1_c2 = encryptBias(bias_fc1_FixedPoint, curveOrder, curveGenerator, h)
    outputCiphertext_c1_FC1 = FCLayer(encryptedValue_c1, weight_fc1_FixedPoint, outputBias_Fc1_c1, 1, identityPoint, curveBaseField)
    outputCiphertext_c2_FC1 = FCLayer(encryptedValue_c2, weight_fc1_FixedPoint, outputBias_Fc1_c2, 1, identityPoint, curveBaseField)

    print("Server: FC1 finished!")
    print("**************************************************")

    return outputCiphertext_c1_FC1, outputCiphertext_c2_FC1

def FC2(weight_fc2, bias_fc2, curveOrder, curveGenerator, h, encryptedValue_c1, encryptedValue_c2, identityPoint, curveBaseField):

    print("\n**************************************************")
    print("Server: FC2 started!")

    weight_fc2_FixedPoint = realNumbersToFixedPointRepresentation(weight_fc2, 1, 16)
    bias_fc2_FixedPoint = realNumbersToFixedPointRepresentation(bias_fc2, 1, 16)

    outputBias_Fc2_c1, outputBias_Fc2_c2 = encryptBias(bias_fc2_FixedPoint, curveOrder, curveGenerator, h)
    outputCiphertext_c1_FC2 = FCLayer(encryptedValue_c1, weight_fc2_FixedPoint, outputBias_Fc2_c1, 1, identityPoint, curveBaseField)    
    outputCiphertext_c2_FC2 = FCLayer(encryptedValue_c2, weight_fc2_FixedPoint, outputBias_Fc2_c2, 1, identityPoint, curveBaseField)
            
    print("Server: FC2 finished!")
    print("Server: Number of EC point multiplications:", len(points_mult))
    print("Server: Number of EC point additions:", len(point_one_Add))
    print("**************************************************")

    return outputCiphertext_c1_FC2, outputCiphertext_c2_FC2

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
    
def inferenceCNN(curveBaseField, curveGenerator, curveOrder, h, identityPoint, weight_fc1, bias_fc1, weight_fc2, bias_fc2, server, conn, kernelSize, stride):

    encryptedValue_c1, encryptedValue_c2 = receiveEncryptedImage(conn, 0)

    print("\n**************************************************")
    print("Server: Encrypted data sample received.")
    print("Server: Performing inference on encrypted data...")
    print("**************************************************")

    outputConv2Ciphertext_c1, outputConv2Ciphertext_c2 = conv2_ciphertext(encryptedValue_c1, encryptedValue_c2, identityPoint, curveBaseField)

    print("\n**************************************************")
    print("Server: First Activation layer started!")

    interactionClient(conn, outputConv2Ciphertext_c1, outputConv2Ciphertext_c2, 0)
    encryptedValue_c1, encryptedValue_c2 = receiveEncryptedImage(conn, 0)

    print("Server: First Activation layer finished!")
    print("**************************************************")

    outputAvgPool2dCiphertext_c1, outputAvgPool2dCiphertext_c2 = avgPool_ciphertext(encryptedValue_c1, encryptedValue_c2, identityPoint, kernelSize, stride)

    outputCiphertext_c1_flat, outputCiphertext_c2_flat = flattening(outputAvgPool2dCiphertext_c1, outputAvgPool2dCiphertext_c2)


    interactionClient(conn, outputCiphertext_c1_flat, outputCiphertext_c2_flat, 1)
    encryptedValue_c1, encryptedValue_c2 = receiveEncryptedImage(conn, 1)

    outputCiphertext_c1_FC1, outputCiphertext_c2_FC1 = FC1(weight_fc1, bias_fc1, curveOrder, curveGenerator, h, encryptedValue_c1, encryptedValue_c2, identityPoint, curveBaseField)

    print("\n**************************************************")
    print("Server: Second Activation layer started!")

    interactionClient(conn, outputCiphertext_c1_FC1, outputCiphertext_c2_FC1, 2)
    encryptedValue_c1, encryptedValue_c2 = receiveEncryptedImage(conn, 2)

    print("Server: Second Activation layer finished!")
    print("**************************************************")

    outputCiphertext_c1_FC2, outputCiphertext_c2_FC2 = FC2(weight_fc2, bias_fc2, curveOrder, curveGenerator, h, encryptedValue_c1, encryptedValue_c2, identityPoint, curveBaseField)

    interactionClient(conn, outputCiphertext_c1_FC2, outputCiphertext_c2_FC2, 3)
    
    server.close()

    convertFormatForRust_pointMult()
    convertFormatForRust_pointAdd()
    print("Server: The witnesses are saved in a file for generating proof with Rust")


def main():
    version = int(sys.argv[1])
    server, conn, address = startServer()
    conn.send("Welcome".encode(FORMAT))

    curveBaseField = 7237005577332262213973186563042994240857116359379907606001950938285454250989
    curveGenerator, h, curveOrder = receiveParameters(conn)
    identityPoint = 0 * h

    weight_fc1, bias_fc1, weight_fc2, bias_fc2 = load_model_parameters(version)
    kernelSize, stride = KERNEL_STRIDE[version]
    
    #print(f"Loaded model version {version} with kernel size {kernelSize} and stride {stride}")
    #print(f"weight_fc1 shape: {weight_fc1.shape}")
    #print(f"bias_fc1 shape: {bias_fc1.shape}")
    #print(f"weight_fc2 shape: {weight_fc2.shape}")
    #print(f"bias_fc2 shape: {bias_fc2.shape}")

    inferenceCNN(curveBaseField, curveGenerator, curveOrder, h, identityPoint, weight_fc1, bias_fc1, weight_fc2, bias_fc2, server, conn, kernelSize, stride)

if __name__ == "__main__":
    main()