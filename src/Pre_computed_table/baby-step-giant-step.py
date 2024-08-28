import time
from ecdsa.ellipticcurve import CurveFp, Point
import pickle
import os

# Get the directory of the current script
script_dir = os.path.dirname(os.path.abspath(__file__))

def curveE2Info():
    """
    Initialize the elliptic curve and its parameters.
    """    
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

def save_table(table, filename):
    """
    Save the precomputed table to a file using pickle.
    """
    with open(filename, 'wb') as file:
        pickle.dump(table, file, protocol=pickle.HIGHEST_PROTOCOL)

def load_table(filename):
    """
    Load the precomputed table from a file using pickle.
    """
    with open(filename, 'rb') as file:
        table = pickle.load(file)
    return table

def baby_step(alpha, beta, n):
    """
    Perform the baby-step phase of the baby-step giant-step algorithm.
    """
    n = 34359738368 # 2^35
    m = int(n ** 0.5) + 1  # Ceiling(sqrt(n)) # 127 bits
    
    m = 3200000 # Size of the precomputed table (enlarged for performance)

    # Baby step phase
    table = {}
    for j in range(m):
        temp = j * alpha
        table[(temp.x(), temp.y())] = j # pow(alpha, j, n)
        if j % 50000 == 0:
            print("Progress: {}% - Generating the table...".format((j / m) * 100))

    save_table(table, script_dir+"/table.pickle")
    print("Table generated")

def giant_step(alpha, beta, n):
    """
    Perform the giant-step phase of the baby-step giant-step algorithm.
    """
    n = 34359738368 # 2^35
    m = int(n ** 0.5) + 1  # Ceiling(sqrt(n)) #127 bits

    m = 3200000 # Size of the precomputed table (consistent with the baby step phase)

    # Read table from file
    table = load_table(script_dir+"/table.pickle")

    # Giant step phase
    inv_alpha_m = -m * alpha #pow(alpha, -m, n)
    gamma = beta
    start = time.time()
    for i in range(m):
        if (gamma.x(), gamma.y()) in table:
            result = i * m + table[(gamma.x(), gamma.y())]
            break
        gamma = (gamma + inv_alpha_m) # (gamma * inv_alpha_m)
    end = time.time()

    return result, end-start

def main():
    """
    Main function to initialize the curve, compute the baby step table, and find the discrete logarithm using the giant step algorithm.
    """
    curve, curveBaseField, curveOrder, curveGenerator, identityPoint = curveE2Info()

    group_order = curveOrder
    generator = curveGenerator
    ciphertext = 34359738367 * generator

    baby_step(generator, ciphertext, group_order)

    flag = 1 # Flag for testing the precomputed table
    if flag == 1:
        print("Testing the precomputed table for discrete logarithm calculations...")
        result, time = giant_step(generator, ciphertext, group_order)
        if result is not None:
            print("Solution found: x =", result)
        else:
            print("No solution found.")

if __name__ == "__main__":
    main()