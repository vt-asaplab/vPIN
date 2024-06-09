import socket
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from ecdsa.ellipticcurve import CurveFp, Point
import random
import numpy as np
import pickle
import sys
import os

IP = socket.gethostbyname("")

script_dir = os.path.dirname(os.path.abspath(__file__))

PORT = int(sys.argv[1])
ADDR = (IP, PORT)
FORMAT = "utf-8"
SIZE = 256000

def encrypt(tensorInputPlaintext, curve, curveBaseField, curveOrder, curveGenerator, h, randomValueRList):
    #---Encryption
    randomValueR = random.randrange(1, curveOrder-1) #r
    message = int(tensorInputPlaintext)
    c1 = randomValueR * curveGenerator
    c2_1 = message * curveGenerator
    c2_2 = randomValueR * h
    c2 = c2_1 + c2_2

    return c1, c2

def encryptFixedPointValue(input, curve, curveBaseField, curveOrder, curveGenerator, h, type):
    randomValueRList = []

    if type == 0:
        batchSize = input.shape[0]
        numChannels = input.shape[1]
        
        encryptFixedPointValue_c1 = np.empty((batchSize, numChannels, input.shape[2], input.shape[3]), dtype=object)
        encryptFixedPointValue_c2 = np.empty((batchSize, numChannels, input.shape[2], input.shape[3]), dtype=object)

        for i in range(batchSize):
            for j in range(input.shape[1]):
                for k in range(input.shape[2]):
                    for l in range(input.shape[3]):
                        c1, c2 = encrypt(input[i][j][k][l], curve, curveBaseField, curveOrder, curveGenerator, h, randomValueRList)
                        encryptFixedPointValue_c1[i][j][k][l] = c1 #c1
                        encryptFixedPointValue_c2[i][j][k][l] = c2 #c2
    else:
        row = input.shape[0]
        col = input.shape[1]
        
        encryptFixedPointValue_c1 = np.empty((row, col), dtype=object)
        encryptFixedPointValue_c2 = np.empty((row, col), dtype=object)

        for i in range(row):
            for j in range(col):
                c1, c2 = encrypt(input[i][j], curve, curveBaseField, curveOrder, curveGenerator, h, randomValueRList)
                encryptFixedPointValue_c1[i][j] = c1 #c1
                encryptFixedPointValue_c2[i][j] = c2 #c2

    return encryptFixedPointValue_c1, encryptFixedPointValue_c2

def fixedPointRepresentationToRealNumbers(fixed_point, bits):
    scale_factor = 2 ** bits  # x bits for the fractional part
    floating_point = np.array(fixed_point, dtype=np.float32) / scale_factor
    return floating_point

def realNumbersToFixedPointRepresentation(Input, type, bits):
    if type == 1:
        scale_factor = 2 ** bits  # x bits for the fractional part
        fixed_point = (Input * scale_factor).astype(np.int32)
    else:
        scale_factor = 2 ** bits  # x bits for the fractional part
        fixed_point = (Input) * scale_factor

    return fixed_point

def send_data_in_chunks(data, chunkSize, client, size):
    client.sendall(str(len(data)).encode(FORMAT))
    msg = client.recv(size).decode(FORMAT)

    for i in range(0, len(data), chunkSize):
        chunk = data[i:i + chunkSize]
        client.sendall(chunk)
        msg = client.recv(size).decode(FORMAT)

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

def encryptInputImage_send(fixedPointValue, client, curve, curveBaseField, curveOrder, curveGenerator, h, type):
    
    encryptedValue_c1, encryptedValue_c2 = encryptFixedPointValue(fixedPointValue, curve, curveBaseField, curveOrder, curveGenerator, h, type)

    encryptedValue_c1_data = pickle.dumps(encryptedValue_c1)
    encryptedValue_c2_data = pickle.dumps(encryptedValue_c2)

    chunk_size = 30000
    send_data_in_chunks(encryptedValue_c1_data, chunk_size, client, SIZE)
    send_data_in_chunks(encryptedValue_c2_data, chunk_size, client, SIZE)

def min_max_scaling(images):
    
    min_val = np.min(images)
    max_val = np.max(images)
    normalized_image = (images - min_val) / (max_val - min_val)
    normalized_image = np.clip(normalized_image, a_min=0.001, a_max=0.9999999)

    return normalized_image

def sendParameters(client, curveGenerator, h, curveOrder):
    curveBaseField_data = pickle.dumps(curveGenerator)    
    h_data = pickle.dumps(h)
    curveOrder_data = pickle.dumps(curveOrder)

    client.sendall(curveBaseField_data)
    msg = client.recv(SIZE).decode(FORMAT)

    client.sendall(h_data)
    msg = client.recv(SIZE).decode(FORMAT)

    client.sendall(curveOrder_data)
    msg = client.recv(SIZE).decode(FORMAT)

def curveE2Info():
    curveBaseField = 7237005577332262213973186563042994240857116359379907606001950938285454250989 #base_field
    a = 3491403595575449084947959021303599933011749826127899762162894550148391771037
    b = 3633908682298454119909199192149978293706667958442512986315258451820769071958
    x = 4561981307020378385254256586024830594940985765081274686120783167106442831732
    y = 684120277165286233470758410892647831027470652988879249692043589061244861334
    curveOrder = 7237005577332262213973186563042994240704759454384003648147593987722918659549 #prime_order

    curve = CurveFp(curveBaseField, a, b)
    curveGenerator = Point(curve, x, y)
    identityPoint = curveGenerator * 0

    return curve, curveBaseField, curveOrder, curveGenerator, identityPoint

def keyGen():
    curve, curveBaseField, curveOrder, curveGenerator, identityPoint = curveE2Info()
    randomValueX = random.randrange(1, curveOrder-1) #x
    h = randomValueX * curveGenerator

    return curve, curveBaseField, curveOrder, curveGenerator, h, randomValueX, identityPoint

def load_table(filename):
    with open(filename, 'rb') as file:
        table = pickle.load(file)
    return table

def connectToServer():
    client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client.connect(ADDR)
    
    return client

def giant_step(alpha, beta, output2, table):
    n = 34359738368
    m = int(n ** 0.5) + 1  # Ceiling(sqrt(n)) #127 bits
    m = 3200000 #int(n ** 0.5) + 1, 400000, 800000, 1600000, 3200000

    # Giant step phase
    inv_alpha_m = -m * alpha #pow(alpha, -m, n)

    gamma = beta
    gamma2 = output2

    for i in range(m):
        if (gamma.x(), gamma.y()) in table:
            result = i * m + table[(gamma.x(), gamma.y())]
            break
        elif (gamma2.x(), gamma2.y()) in table:
            result = i * m + table[(gamma2.x(), gamma2.y())]
            result = -result
            break
        gamma = (gamma + inv_alpha_m) # (gamma * inv_alpha_m)
        gamma2 = (gamma2 + inv_alpha_m) # (gamma * inv_alpha_m)
        
    return result

def decrypt_c1_c2(randomValueX, encrypted_trained_c1, encrypted_trained_c2, curveGenerator, table, type):

    if type == 0:
        row, col = encrypted_trained_c1.shape[2], encrypted_trained_c1.shape[3]
        decryptedItems = np.zeros((encrypted_trained_c1.shape[0],encrypted_trained_c1.shape[1], row, col))

        for i in range(row):
            for j in range(col):
                s = randomValueX * encrypted_trained_c1[0][0][i][j]
                output = encrypted_trained_c2[0][0][i][j] + ((-1) * s)

                s2 = randomValueX * (-1 * encrypted_trained_c1[0][0][i][j])
                output2 = (-1 * encrypted_trained_c2[0][0][i][j]) + ((-1) * s2)

                result = giant_step(curveGenerator, output, output2, table)
                decryptedItems[0][0][i][j] = result

    else:
        row, col = encrypted_trained_c1.shape[0], encrypted_trained_c1.shape[1]
        decryptedItems = np.zeros((row, col))

        for i in range(row):
            for j in range(col):
                s = randomValueX * encrypted_trained_c1[i][j]
                output = encrypted_trained_c2[i][j] + ((-1) * s)

                s2 = randomValueX * (-1 * encrypted_trained_c1[i][j])
                output2 = (-1 * encrypted_trained_c2[i][j]) + ((-1) * s2)

                result = giant_step(curveGenerator, output, output2, table)
                decryptedItems[i][j] = result

    return decryptedItems

def receive_decrypt(client, randomValueX, curveGenerator, table, type):
    finalMsg = receive_data_in_chunks(client)
    encrypted_trained_c1 = pickle.loads(finalMsg)

    finalMsg = receive_data_in_chunks(client)
    encrypted_trained_c2 = pickle.loads(finalMsg)

    #decrypt c1 and c2
    decryptedItems = decrypt_c1_c2(randomValueX, encrypted_trained_c1, encrypted_trained_c2, curveGenerator, table, type)

    return decryptedItems

def relu(decryptedItems):
    
    relu_output = np.maximum(0, decryptedItems)
    return relu_output

def shifting(decryptedItems, bits):
    outputDecryptedRealNumber = fixedPointRepresentationToRealNumbers(decryptedItems, bits)     

    fixedPointValue = realNumbersToFixedPointRepresentation(outputDecryptedRealNumber, 1, 16)

    return fixedPointValue

def main():
    client = connectToServer()
    msg = client.recv(SIZE).decode(FORMAT)
    print("\n**************************************************")
    print("Client: Connection established.")

    table = load_table(script_dir+"/Pre_computed_table/table.pickle")

    print("Client: Generating public-private keys...")
    curve, curveBaseField, curveOrder, curveGenerator, h, randomValueX, identityPoint = keyGen()

    #Send curveGenerator, h, and curveOrder 
    sendParameters(client, curveGenerator, h, curveOrder)

    #load image
    images = np.load(script_dir+"/image_mnist_32_32.npy")
    images = min_max_scaling(images)

    #Encrypt image
    print("Client: Encrypting data sample...")
    print("**************************************************")

    fixedPointValue = realNumbersToFixedPointRepresentation(images, 1, 16)
    encryptInputImage_send(fixedPointValue, client, curve, curveBaseField, curveOrder, curveGenerator, h, 0)

    #Relu 
    decryptedItems = receive_decrypt(client, randomValueX, curveGenerator, table, 0)
    relu_output = relu(decryptedItems)
    encryptInputImage_send(relu_output, client, curve, curveBaseField, curveOrder, curveGenerator, h, 0)

    #interaction with server
    decryptedItems = receive_decrypt(client, randomValueX, curveGenerator, table, 1)
    shifted_output = shifting(decryptedItems, 26)    
    encryptInputImage_send(shifted_output, client, curve, curveBaseField, curveOrder, curveGenerator, h, 1)

    #interaction with server
    decryptedItems = receive_decrypt(client, randomValueX, curveGenerator, table, 2)
    relu_output = relu(decryptedItems)
    shifted_output = shifting(relu_output, 32)  
    encryptInputImage_send(shifted_output, client, curve, curveBaseField, curveOrder, curveGenerator, h, 2)

    #interaction with server
    decryptedItems = receive_decrypt(client, randomValueX, curveGenerator, table, 3)
    relu_output = relu(decryptedItems)

    client.close()

if __name__ == "__main__":
    main()
