import socket
import random
import numpy as np
import json
import pickle
from decimal import Decimal, getcontext
import hmac
import hashlib
import os
import sys
import multiprocessing

# Configuration
MultiCoreFeature = 1
num_processes = 8
getcontext().prec = 256

# Global Variables
points_mult = []
weights_array = []
point_one_Add = []
point_two_Add = []

# Get the directory path of the script and its parent directory
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(os.path.dirname(script_dir))

IP = socket.gethostbyname("")
PORT = int(sys.argv[1])
ADDR = (IP, PORT) # Server address tuple (IP, Port)
FORMAT = "utf-8" # Encoding format
SIZE = 256000 # Buffer size for socket communication

def receiveParameters(conn):
    """
    Receive curve parameters from the client.
    """    
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
    """       
    Initialize the server, bind to the address, and listen for incoming connections.
    """    
    print("\n[STARTING] Server is starting")
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.bind(ADDR)
    server.listen()
    conn, address = server.accept()
    print(f"Server: [NEW CLIENT CONNECTION] {address} connected.")
    return server, conn

def realNumbersToFixedPointRepresentation(Input, type, bits):
    """
    Convert real numbers to fixed-point representation.
    """
    if type == 1:
        scale_factor = 2 ** bits  # x bits for the fractional part
        fixed_point = (Input * scale_factor).astype(np.int32)
    else:
        scale_factor = 2 ** bits  # x bits for the fractional part
        fixed_point = (Input) * scale_factor

    return fixed_point

def receive_data_in_chunks(conn):
    """
    Receive data in chunks from the client.
    """    
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

def receiveEncryptedImage(conn):
    """
    Receive encrypted image data from the client.
    """
    finalMsg = receive_data_in_chunks(conn)
    encryptedValue_c1 = pickle.loads(finalMsg)
    finalMsg = receive_data_in_chunks(conn)
    encryptedValue_c2 = pickle.loads(finalMsg)
    return encryptedValue_c1, encryptedValue_c2

def compute_range(start, end, secret_key, B_prime, lock, identityPoint, final_result):
    temp_sum = identityPoint

    for i in range(start, end):
        random_number = pf(secret_key, i)
        temp = random_number * B_prime[i]
        temp_sum += temp

    final_result.append(temp_sum)

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

        if MultiCoreFeature == 0: # Single core
            print("HERE IN SINGLE CORE")
            for i in range(B_prime.shape[0]):
                random_number = pf(secret_key, i)
                temp = random_number * B_prime[i]
                result = result + temp

        else: # Multi-core
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
    """
    Apply 2D convolution operation.
    """    
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

def pf(secret_key, message):
    counter = str(message).encode('utf-8')
    h = hmac.new(secret_key, counter, hashlib.sha256)
    result = h.digest()[:13]
    integer_result = int(result.hex(), 16)
    return integer_result

def callConv2_ciphertext(images, identityPoint, curveBaseField, padding_size, stride):
    """
    Call the myConv2d function to perform the convolution.
    """
    batchSize = images.shape[0]
    numChannels = images.shape[1]
    height = images.shape[2]
    width = images.shape[3]

    filter_weights = np.array([[2, 0, 0, 0, 0], [0, 2, 0, 0, 0], [0, 0, 1, 0, 0], [0, 0, 0, 2, 0], [0, 0, 0, 0, 2]])

    filter_height, filter_width = filter_weights.shape

    output_height = (height + 2*padding_size - filter_height) // stride + 1
    output_width = (width + 2*padding_size - filter_width) // stride + 1    

    output_numpy = np.empty((batchSize, numChannels, output_height, output_width), dtype=object)

    for i in range(batchSize):
        for j in range(numChannels):
            output_data = myConv2d(images[i][j], filter_weights, identityPoint, curveBaseField, 1, padding_size, stride)
            output_numpy[i][j] = output_data

    return output_numpy

def send_data_in_chunks(data, chunkSize, conn, size):
    conn.sendall(str(len(data)).encode(FORMAT))
    msg = conn.recv(size).decode(FORMAT)

    for i in range(0, len(data), chunkSize):
        chunk = data[i:i + chunkSize]
        conn.sendall(chunk)
        msg = conn.recv(size).decode(FORMAT)

def interactionClient(conn, c1, c2):
    """
    Send encrypted result to the client.
    """    
    encryptedValue_c1 = pickle.dumps(c1)
    encryptedValue_c2 = pickle.dumps(c2)

    chunk_size = 30000
    send_data_in_chunks(encryptedValue_c1, chunk_size, conn, SIZE)
    send_data_in_chunks(encryptedValue_c2, chunk_size, conn, SIZE)

def myAvgPool2d(flag , input_data, identityPoint, type1, type2, kernel_size, stride):
    """
    Apply average pooling operation.
    """
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

def callAvgPool2d_ciphertext(image, identityPoint, kernelSize, stride):
    """
    Call the myAvgPool2d function to perform the average pooling.
    """    
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

def encrypt(tensorInputPlaintext, curveOrder, curveGenerator, h, randomValueRList):
    """
    Perform the Exponential ElGamal encryption.
    """
    randomValueR = random.randrange(1, curveOrder-1) #r
    message = int(tensorInputPlaintext)
    c1 = randomValueR * curveGenerator
    c2_1 = message * curveGenerator
    c2_2 = randomValueR * h
    c2 = c2_1 + c2_2
    return c1, c2

def encryptBias(Input, curveOrder, curveGenerator, h):
    """
    Call the encrypt function to encrypt bias parameters.
    """    
    randomValueRList2 = []
    encryptBias_c1 = np.empty((Input.shape[0]), dtype=object)
    encryptBias_c2 = np.empty((Input.shape[0]), dtype=object)
    for i in range(Input.shape[0]):
        c1, c2 = encrypt(Input[i], curveOrder, curveGenerator, h, randomValueRList2)
        encryptBias_c1[i] = c1 #c1
        encryptBias_c2[i] = c2 #c2
    return encryptBias_c1, encryptBias_c2

def FCLayer(input_data, weight_matrix, bias_vector, flag, identityPoint, curveBaseField):
    """
    Perform the fully connected layer operation.
    """            
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

def firstConv(num_kernels_conv1, encryptedValue_c1, encryptedValue_c2, identityPoint, curveBaseField):
    print("\n**************************************************")
    print("Server: First conv. layer started!")

    outputConv2Ciphertext_c1 = []
    outputConv2Ciphertext_c2 = []

    for i in range (0,num_kernels_conv1):
        outputConv2Ciphertext_c1.append(callConv2_ciphertext(encryptedValue_c1, identityPoint, curveBaseField, 0, 1))
        outputConv2Ciphertext_c2.append(callConv2_ciphertext(encryptedValue_c2, identityPoint, curveBaseField, 0, 1))
        progress = (i + 1) / num_kernels_conv1 * 100
        print(f"Progress: {progress:.2f}%")

    print("Server: First conv. layer finished!")
    print("**************************************************")

    return outputConv2Ciphertext_c1, outputConv2Ciphertext_c2

def firstAct(num_kernels_conv1, conn, outputConv2Ciphertext_c1, outputConv2Ciphertext_c2):
    print("\n**************************************************")
    print("Server: First Activation layer started!")
    encryptedValue_c1 = []
    encryptedValue_c2 = []
    for i in range (0,num_kernels_conv1):
        interactionClient(conn, outputConv2Ciphertext_c1[i], outputConv2Ciphertext_c2[i])
        result1, result2 = receiveEncryptedImage(conn)
        encryptedValue_c1.append(result1)
        encryptedValue_c2.append(result2)
        progress = (i + 1) / num_kernels_conv1 * 100
        print(f"Progress: {progress:.2f}%")

    print("Server: First Activation layer finished!")
    print("**************************************************")

    return encryptedValue_c1, encryptedValue_c2

def firstAvgPool(num_kernels_conv1, identityPoint, kernelSize, stride, encryptedValue_c1, encryptedValue_c2):
    print("\n**************************************************")
    print("Server: First AvgPooling started!")

    outputAvgPool2dCiphertext_c1 = []
    outputAvgPool2dCiphertext_c2 = []

    for i in range (0,num_kernels_conv1):
        outputAvgPool2dCiphertext_c1.append(callAvgPool2d_ciphertext(encryptedValue_c1[i], identityPoint, kernelSize, stride))
        outputAvgPool2dCiphertext_c2.append(callAvgPool2d_ciphertext(encryptedValue_c2[i], identityPoint, kernelSize, stride))
        progress = (i + 1) / num_kernels_conv1 * 100
        print(f"Progress: {progress:.2f}%")

    print("Server: First AvgPooling finished!")
    print("**************************************************")

    return outputAvgPool2dCiphertext_c1, outputAvgPool2dCiphertext_c2

def secondConv(num_kernels_conv1, num_kernels_conv2, identityPoint, curveBaseField, encryptedValue_c1, encryptedValue_c2):
    print("\n**************************************************")
    print("Server: Second conv. layer started!")

    connection_table = [
        [1, 1, 1, 0, 0, 0],  
        [0, 1, 1, 1, 0, 0],  
        [0, 0, 1, 1, 1, 0],  
        [0, 0, 0, 1, 1, 1],  
        [1, 0, 0, 0, 1, 1],  
        [1, 1, 0, 0, 0, 1],  
        [1, 1, 1, 1, 0, 0],  
        [0, 1, 1, 1, 1, 0],  
        [0, 0, 1, 1, 1, 1],  
        [1, 0, 0, 1, 1, 1],  
        [1, 1, 0, 0, 1, 1],  
        [1, 1, 1, 0, 0, 1],  
        [1, 1, 0, 1, 1, 0],  
        [0, 1, 1, 0, 1, 1],  
        [1, 0, 1, 1, 0, 1],  
        [1, 1, 1, 1, 1, 1], 
    ]

    temp_outputConv2Ciphertext_c1_2 = []
    temp_outputConv2Ciphertext_c2_2 = []
    
    outputConv2Ciphertext_c1_2 = []
    outputConv2Ciphertext_c2_2 = []

    for i in range (0, num_kernels_conv2): # 16
        for j in range (0, num_kernels_conv1): # 6
            if connection_table[i][j] == 1:
                temp_outputConv2Ciphertext_c1_2.append(encryptedValue_c1[j])
                temp_outputConv2Ciphertext_c2_2.append(encryptedValue_c2[j])
        
        outputConv2Ciphertext_c1_2.append(callConv2_ciphertext(np.sum(temp_outputConv2Ciphertext_c1_2, axis=0), identityPoint, curveBaseField, 0, 1))
        outputConv2Ciphertext_c2_2.append(callConv2_ciphertext(np.sum(temp_outputConv2Ciphertext_c2_2, axis=0), identityPoint, curveBaseField, 0, 1))
        progress = (i + 1) / num_kernels_conv2 * 100
        print(f"Progress: {progress:.2f}%")
        temp_outputConv2Ciphertext_c1_2 = []
        temp_outputConv2Ciphertext_c2_2 = []        
        
    for i in range(0,1):
        print(outputConv2Ciphertext_c1_2[i].shape)
        print(outputConv2Ciphertext_c2_2[i].shape)

    print("Server: Second conv. layer finished!")
    print("**************************************************")

    return outputConv2Ciphertext_c1_2, outputConv2Ciphertext_c2_2 

def secondAct(num_kernels_conv2, conn, outputConv2Ciphertext_c1_2, outputConv2Ciphertext_c2_2):
    print("\n**************************************************")
    print("Server: Second Activation layer started!")
    encryptedValue_c1 = []
    encryptedValue_c2 = []
    for i in range (0,num_kernels_conv2):
        interactionClient(conn, outputConv2Ciphertext_c1_2[i], outputConv2Ciphertext_c2_2[i])
        result1, result2 = receiveEncryptedImage(conn)
        encryptedValue_c1.append(result1)
        encryptedValue_c2.append(result2)
        progress = (i + 1) / num_kernels_conv2 * 100
        print(f"Progress: {progress:.2f}%")

    print(len(encryptedValue_c2))
    print(encryptedValue_c2[0].shape)
    print("Server: Second Activation layer finished!")
    print("**************************************************")

    return encryptedValue_c1, encryptedValue_c2

def secondAvgPool(num_kernels_conv2, identityPoint, kernelSize, stride, encryptedValue_c1, encryptedValue_c2):
    print("\n**************************************************")
    print("Server: Second AvgPooling started!")

    outputAvgPool2dCiphertext_c1_2 = []
    outputAvgPool2dCiphertext_c2_2 = []

    for i in range (0,num_kernels_conv2): #16
        outputAvgPool2dCiphertext_c1_2.append(callAvgPool2d_ciphertext(encryptedValue_c1[i], identityPoint, kernelSize, stride))
        outputAvgPool2dCiphertext_c2_2.append(callAvgPool2d_ciphertext(encryptedValue_c2[i], identityPoint, kernelSize, stride))
        progress = (i + 1) / num_kernels_conv2 * 100
        print(f"Progress: {progress:.2f}%")

    print("Server: Second AvgPooling finished!")
    print("**************************************************")

    return outputAvgPool2dCiphertext_c1_2, outputAvgPool2dCiphertext_c2_2

def thirdConv(num_kernels_conv2, num_kernels_conv3, encryptedValue_c1, encryptedValue_c2, identityPoint, curveBaseField):
    print("\n**************************************************")
    print("Server: Third conv. layer started!")

    temp_outputConv3Ciphertext_c1_2 = []
    temp_outputConv3Ciphertext_c2_2 = []
    
    outputConv3Ciphertext_c1_2 = []
    outputConv3Ciphertext_c2_2 = []

    for i in range (0, num_kernels_conv3): # 120
        for j in range (0, num_kernels_conv2): # 16
            temp_outputConv3Ciphertext_c1_2.append(encryptedValue_c1[j])
            temp_outputConv3Ciphertext_c2_2.append(encryptedValue_c2[j])
        
        outputConv3Ciphertext_c1_2.append(callConv2_ciphertext(np.sum(temp_outputConv3Ciphertext_c1_2, axis=0), identityPoint, curveBaseField, 0, 1))
        outputConv3Ciphertext_c2_2.append(callConv2_ciphertext(np.sum(temp_outputConv3Ciphertext_c2_2, axis=0), identityPoint, curveBaseField, 0, 1))
        progress = (i + 1) / num_kernels_conv3 * 100
        print(f"Progress: {progress:.2f}%")
        temp_outputConv3Ciphertext_c1_2 = []
        temp_outputConv3Ciphertext_c2_2 = []     

    print("Server: Third conv. layer finished!")
    print("**************************************************")

    output_array_c1 = np.array(outputConv3Ciphertext_c1_2)
    output_array_c2 = np.array(outputConv3Ciphertext_c2_2)
    
    outputConv3Ciphertext_c1 = output_array_c1.reshape(1, 120)
    outputConv3Ciphertext_c2 = output_array_c2.reshape(1, 120)

    return outputConv3Ciphertext_c1, outputConv3Ciphertext_c2

def FC1(weight_fc1, bias_fc1, curveOrder, curveGenerator, h, identityPoint, curveBaseField, encryptedValue_c1_1, encryptedValue_c2_1):
    print("\n**************************************************")
    print("Server: FC1 started!")
    weight_fc1_FixedPoint = realNumbersToFixedPointRepresentation(weight_fc1, 1, 16)
    bias_fc1_FixedPoint = realNumbersToFixedPointRepresentation(bias_fc1, 1, 16)
    outputBias_Fc1_c1, outputBias_Fc1_c2 = encryptBias(bias_fc1_FixedPoint, curveOrder, curveGenerator, h)
    outputCiphertext_c1_FC1 = FCLayer(encryptedValue_c1_1, weight_fc1_FixedPoint, outputBias_Fc1_c1, 1, identityPoint, curveBaseField)
    outputCiphertext_c2_FC1 = FCLayer(encryptedValue_c2_1, weight_fc1_FixedPoint, outputBias_Fc1_c2, 1, identityPoint, curveBaseField)

    print("Server: FC1 finished!")
    print("len pointmult", len(points_mult))
    print("len weights_array", len(weights_array))
    print("len point_one_Add", len(point_one_Add))
    print("len point_two_Add", len(point_two_Add))    
    print("**************************************************")

    return outputCiphertext_c1_FC1, outputCiphertext_c2_FC1

def FC2(weight_fc2, bias_fc2, curveOrder, curveGenerator, h, identityPoint, curveBaseField, encryptedValue_c1_1, encryptedValue_c2_1):
    print("\n**************************************************")
    print("Server: FC2 started!")
    
    weight_fc2_FixedPoint = realNumbersToFixedPointRepresentation(weight_fc2, 1, 16)
    bias_fc2_FixedPoint = realNumbersToFixedPointRepresentation(bias_fc2, 1, 16)
    outputBias_Fc2_c1, outputBias_Fc2_c2 = encryptBias(bias_fc2_FixedPoint, curveOrder, curveGenerator, h)
    outputCiphertext_c1_FC2 = FCLayer(encryptedValue_c1_1, weight_fc2_FixedPoint, outputBias_Fc2_c1, 1, identityPoint, curveBaseField)    
    outputCiphertext_c2_FC2 = FCLayer(encryptedValue_c2_1, weight_fc2_FixedPoint, outputBias_Fc2_c2, 1, identityPoint, curveBaseField)

    print("Server: FC2 finished!")
    print("Server: Number of EC point multiplications:", len(points_mult))
    print("Server: Number of EC point additions:", len(point_one_Add))  
    print("**************************************************")

    return outputCiphertext_c1_FC2, outputCiphertext_c2_FC2

def intToByte(integer):
    """
    Converts an integer into a 32-byte list representation.
    """    
    byte_array = []
    while integer > 0:
        byte_array.append(integer & 255)
        integer >>= 8

    byte_array.extend([0] * (32 - len(byte_array)))

    return byte_array

def convertFormatForRust_pointMult():
    """
    Save the EC point multiplication and witnesses.
    """    
    point_mult_px_byte = []
    point_mult_py_byte = []

    for item in points_mult:
        point_mult_px_byte.append(intToByte(item.x()))
        point_mult_py_byte.append(intToByte(item.y()))

    # save in JSON format for RUST
    my_array = np.array(weights_array, dtype=object)
    my_array = my_array.tolist()
    my_array = [str(x) for x in my_array]
    file_path = os.path.join(parent_dir, "src", "rust_files", "pointMult", "weight.json")
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, 'w') as file:
        json.dump(my_array, file)

    my_array1 = np.array(point_mult_px_byte, dtype=np.int64)
    my_array1 = my_array1.tolist()        
    file_path = os.path.join(parent_dir, "src", "rust_files", "pointMult", "point_mult_px_byte.json")
    os.makedirs(os.path.dirname(file_path), exist_ok=True)    
    with open(file_path, 'w') as file:
        json.dump(my_array1, file)

    my_array2 = np.array(point_mult_py_byte, dtype=np.int64)
    my_array2 = my_array2.tolist()   
    file_path = os.path.join(parent_dir, "src", "rust_files", "pointMult", "point_mult_py_byte.json")
    os.makedirs(os.path.dirname(file_path), exist_ok=True)             
    with open(file_path, 'w') as file:
        json.dump(my_array2, file)

def convertFormatForRust_pointAdd():
    """
    Save the EC point multiplication and witnesses.
    """    
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
    file_path = os.path.join(parent_dir, "src", "rust_files", "pointAdd", "point_add_px_byte.json")
    os.makedirs(os.path.dirname(file_path), exist_ok=True)                  
    with open(file_path, 'w') as file:
        json.dump(my_array3, file)

    my_array4 = np.array(point_add_py_byte, dtype=np.int64)
    my_array4 = my_array4.tolist() 
    file_path = os.path.join(parent_dir, "src", "rust_files", "pointAdd", "point_add_py_byte.json")
    os.makedirs(os.path.dirname(file_path), exist_ok=True)             
    with open(file_path, 'w') as file:
        json.dump(my_array4, file)

    my_array5 = np.array(point_add_rx_byte, dtype=np.int64)
    my_array5 = my_array5.tolist()   
    file_path = os.path.join(parent_dir, "src", "rust_files", "pointAdd", "point_add_rx_byte.json")
    os.makedirs(os.path.dirname(file_path), exist_ok=True)            
    with open(file_path, 'w') as file:
        json.dump(my_array5, file)

    my_array6 = np.array(point_add_ry_byte, dtype=np.int64)
    my_array6 = my_array6.tolist()        
    file_path = os.path.join(parent_dir, "src", "rust_files", "pointAdd", "point_add_ry_byte.json")
    os.makedirs(os.path.dirname(file_path), exist_ok=True)        
    with open(file_path, 'w') as file:
        json.dump(my_array6, file)

    my_array7 = np.array(point_add_rz_byte, dtype=np.int64)
    my_array7 = my_array7.tolist()      
    file_path = os.path.join(parent_dir, "src", "rust_files", "pointAdd", "point_add_rz_byte.json")
    os.makedirs(os.path.dirname(file_path), exist_ok=True)              
    with open(file_path, 'w') as file:
        json.dump(my_array7, file)

def curveInfo(conn):
    curveBaseField = 7237005577332262213973186563042994240857116359379907606001950938285454250989
    curveGenerator, h, curveOrder = receiveParameters(conn)
    identityPoint = 0 * h
    return curveBaseField, curveGenerator, curveOrder, identityPoint, h

def inferenceCNN(curveBaseField, curveGenerator, curveOrder, h, identityPoint, weight_fc1, bias_fc1, weight_fc2, bias_fc2, server, conn, kernelSize, stride):
    """
    Performs CNN inference on encrypted data received from the client.
    """
    num_kernels_conv1 = 6 #6
    num_kernels_conv2 = 16 #16
    num_kernels_conv3 = 120 #120

    encryptedValue_c1, encryptedValue_c2 = receiveEncryptedImage(conn)

    print("\n**************************************************")
    print("Server: Encrypted data sample received.")
    print("Server: Performing inference on encrypted data...")
    print("**************************************************")

    outputConv2Ciphertext_c1, outputConv2Ciphertext_c2 = firstConv(num_kernels_conv1, encryptedValue_c1, encryptedValue_c2, identityPoint, curveBaseField)

    encryptedValue_c1, encryptedValue_c2 = firstAct(num_kernels_conv1, conn, outputConv2Ciphertext_c1, outputConv2Ciphertext_c2)

    outputAvgPool2dCiphertext_c1, outputAvgPool2dCiphertext_c2 = firstAvgPool(num_kernels_conv1, identityPoint, kernelSize, stride, encryptedValue_c1, encryptedValue_c2)

    encryptedValue_c1 = []
    encryptedValue_c2 = []
    for i in range (0,num_kernels_conv1):
        interactionClient(conn, outputAvgPool2dCiphertext_c1[i], outputAvgPool2dCiphertext_c2[i])
        result1, result2 = receiveEncryptedImage(conn)
        encryptedValue_c1.append(result1)
        encryptedValue_c2.append(result2)

    outputConv2Ciphertext_c1_2, outputConv2Ciphertext_c2_2 = secondConv(num_kernels_conv1, num_kernels_conv2, identityPoint, curveBaseField, encryptedValue_c1, encryptedValue_c2)

    encryptedValue_c1, encryptedValue_c2 = secondAct(num_kernels_conv2, conn, outputConv2Ciphertext_c1_2, outputConv2Ciphertext_c2_2)

    outputAvgPool2dCiphertext_c1_2, outputAvgPool2dCiphertext_c2_2 = secondAvgPool(num_kernels_conv2, identityPoint, kernelSize, stride, encryptedValue_c1, encryptedValue_c2)

    # Interaction with client
    encryptedValue_c1 = []
    encryptedValue_c2 = []
    for i in range (0,num_kernels_conv2):
        interactionClient(conn, outputAvgPool2dCiphertext_c1_2[i], outputAvgPool2dCiphertext_c2_2[i])
        result1, result2 = receiveEncryptedImage(conn)
        encryptedValue_c1.append(result1)
        encryptedValue_c2.append(result2)

    outputConv3Ciphertext_c1, outputConv3Ciphertext_c2 = thirdConv(num_kernels_conv2, num_kernels_conv3, encryptedValue_c1, encryptedValue_c2, identityPoint, curveBaseField)

    print("\n**************************************************")
    print("Server: Third Activation layer started!")
    interactionClient(conn, outputConv3Ciphertext_c1, outputConv3Ciphertext_c2)
    encryptedValue_c1_1, encryptedValue_c2_1 = receiveEncryptedImage(conn)
    print("Server: Third Activation layer finished!")
    print("**************************************************")

    outputCiphertext_c1_FC1, outputCiphertext_c2_FC1 = FC1(weight_fc1, bias_fc1, curveOrder, curveGenerator, h, identityPoint, curveBaseField, encryptedValue_c1_1, encryptedValue_c2_1)

    print("\n**************************************************")
    print("Server: Forth Activation layer started!")
    interactionClient(conn, outputCiphertext_c1_FC1, outputCiphertext_c2_FC1)
    encryptedValue_c1_1, encryptedValue_c2_1 = receiveEncryptedImage(conn)
    print("Server: Forth Activation layer finished!")
    print("**************************************************")

    outputCiphertext_c1_FC2, outputCiphertext_c2_FC2 = FC2(weight_fc2, bias_fc2, curveOrder, curveGenerator, h, identityPoint, curveBaseField, encryptedValue_c1_1, encryptedValue_c2_1)

    interactionClient(conn, outputCiphertext_c1_FC2, outputCiphertext_c2_FC2)

    server.close()

    convertFormatForRust_pointMult()
    convertFormatForRust_pointAdd()
    print("Server: The witnesses are saved in a file for generating proof with Rust")

def main():
    server, conn = startServer()
    conn.send("Welcome".encode(FORMAT))

    curveBaseField, curveGenerator, curveOrder, identityPoint, h = curveInfo(conn)

    weight_fc1 = np.load(script_dir+"/Pre_trained_model/weight_fc1_120_84.npy")
    bias_fc1 = np.load(script_dir+"/Pre_trained_model/bias_fc1_84.npy")
    weight_fc2 = np.load(script_dir+"/Pre_trained_model/weight_fc2_84_10.npy")
    bias_fc2 = np.load(script_dir+"/Pre_trained_model/bias_fc2_10.npy")            

    kernelSize = 2
    stride = 2

    inferenceCNN(curveBaseField, curveGenerator, curveOrder, h, identityPoint, weight_fc1, bias_fc1, weight_fc2, bias_fc2, server, conn, kernelSize, stride)

if __name__ == "__main__":
    main()