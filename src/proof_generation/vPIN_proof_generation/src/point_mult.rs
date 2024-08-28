use libspartan::{Instance, VarsAssignment, InputsAssignment};
use curve25519_dalek::scalar::Scalar;
use std::thread;
use std::time::Duration;
use crate::load_data::load_data;

pub fn point_mult() -> (
    usize,
    usize,
    usize,
    usize,
    Instance,
    VarsAssignment,
    VarsAssignment,
    VarsAssignment,
    InputsAssignment,
){

    // Read the JSON file
    let (weights_len, weight_list, point_mult_x_byte, point_mult_y_byte, n) = load_data();

    println!("Point Multiplication Gadget...");

    println!("Number of Point Multiplications: {}", weights_len);
    let number_of_point_multiplications: usize = weights_len;

    let (param_1, param_2, param_3);

    if number_of_point_multiplications == 50 {
        param_1 = 100;
        param_2 = 2;
        param_3 = 80;
    }
    else if number_of_point_multiplications == 210 {
        param_1 = 300;
        param_2 = 2;
        param_3 = 20;      
    }
    else if number_of_point_multiplications < 660 {
        param_1 = 100;
        param_2 = 2;
        param_3 = 40;
    } else {
        param_1 = 350;
        param_2 = 2;
        param_3 = 20;
    } 

    println!("Generating Proof...");

    //parameters of the R1CS instance
    let one_num_cons = 27 * n + 8; // n=35 -> 953
    let one_num_vars = n + 10 + (n * 26); // n=35 -> 955 + 1 

    let num_cons_1 = one_num_cons * number_of_point_multiplications;
    let num_vars_1 = (one_num_vars * number_of_point_multiplications) + 1;
    let num_inputs_1 = 1;
    let num_non_zero_entries_1 = param_1 * ((param_2*n) + (param_3 * number_of_point_multiplications));
    
    //encode the above constraints into three matrices A, B, C
    let mut A: Vec<(usize, usize, [u8; 32])> = Vec::new();
    let mut B: Vec<(usize, usize, [u8; 32])> = Vec::new();
    let mut C: Vec<(usize, usize, [u8; 32])> = Vec::new();
    
    // a variable that holds a byte representation of 1
    let mut two_base = Scalar::one();
    let zero = Scalar::zero().to_bytes();
    let one = Scalar::one().to_bytes();
    let two = (Scalar::one() + Scalar::one()).to_bytes();
    let three = (Scalar::one() + Scalar::one() + Scalar::one()).to_bytes();

    let minus_one = (Scalar::zero() - Scalar::one()).to_bytes();
    let minus_two = (Scalar::zero() - Scalar::one() - Scalar::one()).to_bytes();
    
    //set the matrices. Suppose the order of the variables is (b0, b1, ..., b255, x)
    for j in 0..number_of_point_multiplications{

        for i in 0..n as i32 {
            A.push((0 + (one_num_cons * j), i as usize + (one_num_vars * j as usize), two_base.to_bytes()));
            //println!("two base is {:?} {}", two_base, i);
            two_base = two_base * Scalar::from(2u64);
        }
        two_base = Scalar::one();
        B.push((0 + (one_num_cons * j), num_vars_1, one));
        C.push((0 + (one_num_cons * j), n + (one_num_vars * j), one));

        for i in 1..(n+1) {
            A.push((i + (one_num_cons * j), i-1 + (one_num_vars * j), one));
            B.push((i + (one_num_cons * j), i-1 + (one_num_vars * j), one));
            C.push((i + (one_num_cons * j), i-1 + (one_num_vars * j), one));
        }

        // (Ax0 - Px) * 1 = 0
        A.push((n+1 + (one_num_cons * j), n+1 + (one_num_vars * j), one)); //n+1 = 36 
        A.push((n+1 + (one_num_cons * j), 10*n+8 + (one_num_vars * j), minus_one));
        B.push((n+1 + (one_num_cons * j), num_vars_1, one));

        // (Ay0 - Py) * 1 = 0
        A.push((n+2 + (one_num_cons * j), 2*n+2 + (one_num_vars * j), one)); //n+2 = 37
        A.push((n+2 + (one_num_cons * j), 10*n+9 + (one_num_vars * j), minus_one));
        B.push((n+2 + (one_num_cons * j), num_vars_1, one));

        // Bx0 * 1 = 0
        A.push((n+3 + (one_num_cons * j), 3*n+3 + (one_num_vars * j), one)); //n+3 = 38 
        B.push((n+3 + (one_num_cons * j), num_vars_1, one));

        // By0 * 1 = 0
        A.push((n+4 + (one_num_cons * j), 4*n+4 + (one_num_vars * j), one)); //n+4 = 39 
        B.push((n+4 + (one_num_cons * j), num_vars_1, one));

        // (Bz0 - 1) * 1 = 0
        A.push((n+5 + (one_num_cons * j), 5*n+5 + (one_num_vars * j), one)); //n+5 = 40 
        A.push((n+5 + (one_num_cons * j), num_vars_1, minus_one));
        B.push((n+5 + (one_num_cons * j), num_vars_1, one));

        // ****************
        //PA
        // ****************
        
        for i in 0..n {
            // constraint 0 in PA entries in (A,B,C)
            // c * (Rx - Px) = 1
            A.push((n+6+(i*26) + (one_num_cons * j), 10*n+10+i + (one_num_vars * j), one)); //n+6 = 41, 67
            B.push((n+6+(i*26) + (one_num_cons * j), 3*n+3+i + (one_num_vars * j), one));
            B.push((n+6+(i*26) + (one_num_cons * j), n+1+i + (one_num_vars * j), minus_one));
            C.push((n+6+(i*26) + (one_num_cons * j), num_vars_1, one));

            // constraint 1 in PA entries in (A,B,C)
            // (Ry - Py) * c = s1
            A.push((n+7+(i*26) + (one_num_cons * j), 4*n+4+i + (one_num_vars * j), one)); //n+7 = 42
            A.push((n+7+(i*26) + (one_num_cons * j), 2*n+2+i + (one_num_vars * j), minus_one));
            B.push((n+7+(i*26) + (one_num_cons * j), 10*n+10+i + (one_num_vars * j), one));
            C.push((n+7+(i*26) + (one_num_cons * j), 11*n+10+i + (one_num_vars * j), one));

            // constraint 2 in PA entries in (A,B,C)
            // s1 * s1 = s2
            A.push((n+8+(i*26) + (one_num_cons * j), 11*n+10+i + (one_num_vars * j), one)); //n+8 = 43 
            B.push((n+8+(i*26) + (one_num_cons * j), 11*n+10+i + (one_num_vars * j), one));
            C.push((n+8+(i*26) + (one_num_cons * j), 12*n+10+i + (one_num_vars * j), one));

            // constraint 3 in PA entries in (A,B,C)
            // (s2 - Px - Rx) * (1 - Rz) = t1
            A.push((n+9+(i*26) + (one_num_cons * j), 12*n+10+i + (one_num_vars * j), one)); //n+9 = 44
            A.push((n+9+(i*26) + (one_num_cons * j), n+1+i + (one_num_vars * j), minus_one));
            A.push((n+9+(i*26) + (one_num_cons * j), 3*n+3+i + (one_num_vars * j), minus_one));
            B.push((n+9+(i*26) + (one_num_cons * j), num_vars_1, one));
            B.push((n+9+(i*26) + (one_num_cons * j), 5*n+5+i + (one_num_vars * j), minus_one));
            C.push((n+9+(i*26) + (one_num_cons * j), 14*n+10+i + (one_num_vars * j), one));

            // constraint 4 in PA entries in (A,B,C)
            // px * Rz = t2
            A.push((n+10+(i*26) + (one_num_cons * j), n+1+i + (one_num_vars * j), one)); //n+10 = 45
            B.push((n+10+(i*26) + (one_num_cons * j), 5*n+5+i + (one_num_vars * j), one));
            C.push((n+10+(i*26) + (one_num_cons * j), 15*n+10+i + (one_num_vars * j), one));
        
            // constraint 5 in PA entries in (A,B,C)
            // (t1 + t2) * 1 = x3
            A.push((n+11+(i*26) + (one_num_cons * j), 14*n+10+i + (one_num_vars * j), one)); //n+11 = 46
            A.push((n+11+(i*26) + (one_num_cons * j), 15*n+10+i + (one_num_vars * j), one));
            B.push((n+11+(i*26) + (one_num_cons * j), num_vars_1, one));
            C.push((n+11+(i*26) + (one_num_cons * j), 6*n+6+i + (one_num_vars * j), one));

            // constraint 6 in PA entries in (A,B,C)
            // s1 * (Px - x3) = s3
            A.push((n+12+(i*26) + (one_num_cons * j), 11*n+10+i + (one_num_vars * j), one)); //n+12 = 47
            B.push((n+12+(i*26) + (one_num_cons * j), n+1+i + (one_num_vars * j), one));
            B.push((n+12+(i*26) + (one_num_cons * j), 6*n+6+i + (one_num_vars * j), minus_one));
            C.push((n+12+(i*26) + (one_num_cons * j), 13*n+10+i + (one_num_vars * j), one));

            // constraint 7 in PA entries in (A,B,C)
            // (s3 - Py) * (1 - Rz) = t3 
            A.push((n+13+(i*26) + (one_num_cons * j), 13*n+10+i + (one_num_vars * j), one)); //n+13 = 48
            A.push((n+13+(i*26) + (one_num_cons * j), 2*n+2+i + (one_num_vars * j), minus_one));
            B.push((n+13+(i*26) + (one_num_cons * j), num_vars_1, one));
            B.push((n+13+(i*26) + (one_num_cons * j), 5*n+5+i + (one_num_vars * j), minus_one));    
            C.push((n+13+(i*26) + (one_num_cons * j), 16*n+10+i + (one_num_vars * j), one));

            // constraint 8 in PA entries in (A,B,C)
            // Py * Rz = t4
            A.push((n+14+(i*26) + (one_num_cons * j), 2*n+2+i + (one_num_vars * j), one)); //n+14 = 49
            B.push((n+14+(i*26) + (one_num_cons * j), 5*n+5+i + (one_num_vars * j), one));
            C.push((n+14+(i*26) + (one_num_cons * j), 17*n+10+i + (one_num_vars * j), one)); 
        
            // constraint 9 in PA entries in (A,B,C)
            // (t3 + t4) * 1 = y3
            A.push((n+15+(i*26) + (one_num_cons * j), 16*n+10+i + (one_num_vars * j), one)); //n+15 = 50
            A.push((n+15+(i*26) + (one_num_cons * j), 17*n+10+i + (one_num_vars * j), one));
            B.push((n+15+(i*26) + (one_num_cons * j), num_vars_1, one));
            C.push((n+15+(i*26) + (one_num_cons * j), 7*n+6+i + (one_num_vars * j), one)); 
            
            // last constraint: n+15 + (31 * 24) = 791

            // ****************
            //PD
            // ****************

            // constraint 0 in PD entries in (A,B,C)
            // c * 2Py = 1 
            A.push((n+16+(i*26) + (one_num_cons * j), 18*n+10+i + (one_num_vars * j), one)); //n+16 = 51
            B.push((n+16+(i*26) + (one_num_cons * j), 2*n+2+i + (one_num_vars * j), two));
            C.push((n+16+(i*26) + (one_num_cons * j), num_vars_1, one));

            // constraint 1 in PD entries in (A,B,C)
            // Px × Px = t1
            A.push((n+17+(i*26) + (one_num_cons * j), n+1+i + (one_num_vars * j), one)); //n+17 = 52
            B.push((n+17+(i*26) + (one_num_cons * j), n+1+i + (one_num_vars * j), one));
            C.push((n+17+(i*26) + (one_num_cons * j), 19*n+10+i + (one_num_vars * j), one));

            // constraint 2 in PD entries in (A,B,C)
            // ((3t1) + a) * c = s1
            A.push((n+18+(i*26) + (one_num_cons * j), 19*n+10+i + (one_num_vars * j), three)); //n+18 = 53
            A.push((n+18+(i*26) + (one_num_cons * j), num_vars_1 + 1, one));
            B.push((n+18+(i*26) + (one_num_cons * j), 18*n+10+i + (one_num_vars * j), one));
            C.push((n+18+(i*26) + (one_num_cons * j), 20*n+10+i + (one_num_vars * j), one));

            // constraint 3 in PD entries in (A,B,C)
            // s1 * s1 = s2
            A.push((n+19+(i*26) + (one_num_cons * j), 20*n+10+i + (one_num_vars * j), one)); //n+19 = 54
            B.push((n+19+(i*26) + (one_num_cons * j), 20*n+10+i + (one_num_vars * j), one));
            C.push((n+19+(i*26) + (one_num_cons * j), 21*n+10+i + (one_num_vars * j), one));

            // constraint 4 in PD entries in (A,B,C)
            // (s2 − 2Px) * 1 = x3
            A.push((n+20+(i*26) + (one_num_cons * j), 21*n+10+i + (one_num_vars * j), one)); //n+20 = 55
            A.push((n+20+(i*26) + (one_num_cons * j), n+1+i + (one_num_vars * j), minus_two));
            B.push((n+20+(i*26) + (one_num_cons * j), num_vars_1, one));
            C.push((n+20+(i*26) + (one_num_cons * j), 8*n+6+i + (one_num_vars * j), one));
            
            // constraint 5 in PD entries in (A,B,C)
            // s1 × (Px − x3) = t2
            A.push((n+21+(i*26) + (one_num_cons * j), 20*n+10+i + (one_num_vars * j), one)); //n+21 = 56
            B.push((n+21+(i*26) + (one_num_cons * j), n+1+i + (one_num_vars * j), one));
            B.push((n+21+(i*26) + (one_num_cons * j), 8*n+6+i + (one_num_vars * j), minus_one));
            C.push((n+21+(i*26) + (one_num_cons * j), 22*n+10+i + (one_num_vars * j), one));

            // constraint 6 in PD entries in (A,B,C)
            // (t2 − Py) * 1 = y3
            A.push((n+22+(i*26) + (one_num_cons * j), 22*n+10+i + (one_num_vars * j), one)); //n+22 = 57
            A.push((n+22+(i*26) + (one_num_cons * j), 2*n+2+i + (one_num_vars * j), minus_one));
            B.push((n+22+(i*26) + (one_num_cons * j), num_vars_1, one));
            C.push((n+22+(i*26) + (one_num_cons * j), 9*n+6+i + (one_num_vars * j), one));

            // last constraint: 54 + (31 * 24) = 798

            // ****************

            // Cx1 * v[1] = z1
            A.push((n+23+(i*26) + (one_num_cons * j), 6*n+6+i + (one_num_vars * j), one)); //n+23 = 58
            B.push((n+23+(i*26) + (one_num_cons * j), 0+i + (one_num_vars * j), one));
            C.push((n+23+(i*26) + (one_num_cons * j), 23*n+10+i + (one_num_vars * j), one));

            // Bx0 * (1 - v[1]) = z2
            A.push((n+24+(i*26) + (one_num_cons * j), 3*n+3+i + (one_num_vars * j), one)); //n+24 = 59
            B.push((n+24+(i*26) + (one_num_cons * j), num_vars_1, one));
            B.push((n+24+(i*26) + (one_num_cons * j), 0+i + (one_num_vars * j), minus_one));
            C.push((n+24+(i*26) + (one_num_cons * j), 24*n+10+i + (one_num_vars * j), one));

            // (z1 + z2) * 1 = Bx1
            A.push((n+25+(i*26) + (one_num_cons * j), 23*n+10+i + (one_num_vars * j), one)); //n+25 = 60
            A.push((n+25+(i*26) + (one_num_cons * j), 24*n+10+i + (one_num_vars * j), one));
            B.push((n+25+(i*26) + (one_num_cons * j), num_vars_1, one));
            C.push((n+25+(i*26) + (one_num_cons * j), 3*n+4 +i + (one_num_vars * j), one));

            // Cy1 * v[1] = z3
            A.push((n+26+(i*26) + (one_num_cons * j), 7*n+6+i + (one_num_vars * j), one)); //n+26 = 61
            B.push((n+26+(i*26) + (one_num_cons * j), 0+i + (one_num_vars * j), one));
            C.push((n+26+(i*26) + (one_num_cons * j), 25*n+10+i + (one_num_vars * j), one));

            // By0 * (1 - v[1]) = z4
            A.push((n+27+(i*26) + (one_num_cons * j), 4*n+4+i + (one_num_vars * j), one)); //n+27 = 62
            B.push((n+27+(i*26) + (one_num_cons * j), num_vars_1, one));
            B.push((n+27+(i*26) + (one_num_cons * j), 0+i + (one_num_vars * j), minus_one));   
            C.push((n+27+(i*26) + (one_num_cons * j), 26*n+10+i + (one_num_vars * j), one));

            // (z3 + z4) * 1 = By1
            A.push((n+28+(i*26) + (one_num_cons * j), 25*n+10+i + (one_num_vars * j), one)); //n+28 = 63
            A.push((n+28+(i*26) + (one_num_cons * j), 26*n+10+i + (one_num_vars * j), one));
            B.push((n+28+(i*26) + (one_num_cons * j), num_vars_1, one));
            C.push((n+28+(i*26) + (one_num_cons * j), 4*n+5+i + (one_num_vars * j), one));

            // Bz0 * (1 - v[1]) = Bz1
            A.push((n+29+(i*26) + (one_num_cons * j), 5*n+5+i + (one_num_vars * j), one)); //n+29 = 64
            B.push((n+29+(i*26) + (one_num_cons * j), num_vars_1, one));   
            B.push((n+29+(i*26) + (one_num_cons * j), 0+i + (one_num_vars * j), minus_one));   
            C.push((n+29+(i*26) + (one_num_cons * j), 5*n+6+i + (one_num_vars * j), one));

            // (Ax1 - dx1) * 1 = 0
            A.push((n+30+(i*26) + (one_num_cons * j), n+2+i + (one_num_vars * j), one)); //n+30 = 65
            A.push((n+30+(i*26) + (one_num_cons * j), 8*n+6+i + (one_num_vars * j), minus_one));
            B.push((n+30+(i*26) + (one_num_cons * j), num_vars_1, one));

            // (Ay1 - dy1) * 1 = 0
            A.push((n+31+(i*26) + (one_num_cons * j), 2*n+3+i + (one_num_vars * j), one)); //n+31 = 66
            A.push((n+31+(i*26) + (one_num_cons * j), 9*n+6+i + (one_num_vars * j), minus_one));
            B.push((n+31+(i*26) + (one_num_cons * j), num_vars_1, one));

            // last constraint: 66 + (34 * 26) = 950
        }
        
        // ****************
        // constraint 870
        // (Qx - Bx35) * 1 = 0 // Bx35
        A.push((one_num_cons-2 + (one_num_cons * j), 10*n+6 + (one_num_vars * j), one)); //num_cons_1 - 2
        A.push((one_num_cons-2 + (one_num_cons * j), 3*n+3+n + (one_num_vars * j), minus_one));
        B.push((one_num_cons-2 + (one_num_cons * j), num_vars_1, one));

        // constraint 871
        // (Qy - By35) * 1 = 0 //By25
        
        A.push((one_num_cons-1 + (one_num_cons * j), 10*n+7 + (one_num_vars * j), one)); //num_cons_1 - 1
        A.push((one_num_cons-1 + (one_num_cons * j), 4*n+4+n + (one_num_vars * j), minus_one));
        B.push((one_num_cons-1 + (one_num_cons * j), num_vars_1, one));
    }


    let inst_1 = Instance::new(num_cons_1, num_vars_1, num_inputs_1, &A, &B, &C).unwrap();

    
    let mut number: u128 = 0;

    let mut a = vec![Scalar::from(0u8); number_of_point_multiplications];
    let mut bit_string: Vec<String> = Vec::new();
    let mut bits: Vec<Vec<Scalar>> = vec![vec![Scalar::from_bytes_mod_order(zero); n]; number_of_point_multiplications];

    for j in 0..number_of_point_multiplications{
        number = weight_list[j] as u128;
        a[j] = Scalar::from(number);
        bit_string.push(String::from(u128_to_128_bit_string(number, n)));
        bits[j] = process_bit_string(&bit_string[j]);
    }
    
    let a_pd_byte = [157, 27, 50, 101, 63, 42, 38, 142, 68, 159, 245, 15, 16, 47, 75, 58, 203, 87, 15, 3, 219, 183, 77, 94, 64, 118, 147, 233, 124, 16, 184, 7];
    let a_pd = Scalar::from_bytes_mod_order(a_pd_byte);

    let mut ax0 = vec![Scalar::from_bytes_mod_order(zero); number_of_point_multiplications];
    let mut ay0 = vec![Scalar::from_bytes_mod_order(zero); number_of_point_multiplications];

    let mut px = vec![Scalar::from_bytes_mod_order(zero); number_of_point_multiplications];
    let mut py = vec![Scalar::from_bytes_mod_order(zero); number_of_point_multiplications];

    for j in 0..number_of_point_multiplications{
        let mut px_byte: [u8; 32] = [0; 32];
        let mut py_byte: [u8; 32] = [0; 32];

        for (i, &value) in point_mult_x_byte[j].iter().enumerate() {
            px_byte[i] = value as u8;
        }
    
        for (i, &value) in point_mult_y_byte[j].iter().enumerate() {
            py_byte[i] = value as u8;
        }

        //println!("point_mult_x_byte: {:?}", px_byte);
        //println!("point_mult_y_byte: {:?}", py_byte);
        
        px[j] = Scalar::from_bytes_mod_order(px_byte);
        py[j] = Scalar::from_bytes_mod_order(py_byte);
    
        ax0[j] = px[j];
        ay0[j] = py[j];

    }

    let mut bx0 = vec![Scalar::from_bytes_mod_order(zero); number_of_point_multiplications];
    let mut by0 = vec![Scalar::from_bytes_mod_order(zero); number_of_point_multiplications];
    let mut bz0 = vec![Scalar::from_bytes_mod_order(one); number_of_point_multiplications];


    let mut ax: Vec<Vec<Scalar>> = vec![vec![Scalar::from_bytes_mod_order(zero); n]; number_of_point_multiplications];
    let mut ay: Vec<Vec<Scalar>> = vec![vec![Scalar::from_bytes_mod_order(zero); n]; number_of_point_multiplications];
    let mut bx: Vec<Vec<Scalar>> = vec![vec![Scalar::from_bytes_mod_order(zero); n]; number_of_point_multiplications];
    let mut by: Vec<Vec<Scalar>> = vec![vec![Scalar::from_bytes_mod_order(zero); n]; number_of_point_multiplications];
    let mut bz: Vec<Vec<Scalar>> = vec![vec![Scalar::from_bytes_mod_order(zero); n]; number_of_point_multiplications];
    let mut cx: Vec<Vec<Scalar>> = vec![vec![Scalar::from_bytes_mod_order(zero); n]; number_of_point_multiplications];
    let mut cy: Vec<Vec<Scalar>> = vec![vec![Scalar::from_bytes_mod_order(zero); n]; number_of_point_multiplications];
    let mut dx: Vec<Vec<Scalar>> = vec![vec![Scalar::from_bytes_mod_order(zero); n]; number_of_point_multiplications];
    let mut dy: Vec<Vec<Scalar>> = vec![vec![Scalar::from_bytes_mod_order(zero); n]; number_of_point_multiplications];

    let mut c_pa: Vec<Vec<Scalar>> = vec![vec![Scalar::from_bytes_mod_order(zero); n]; number_of_point_multiplications];    
    let mut s1_pa: Vec<Vec<Scalar>> = vec![vec![Scalar::from_bytes_mod_order(zero); n]; number_of_point_multiplications];    
    let mut s2_pa: Vec<Vec<Scalar>> = vec![vec![Scalar::from_bytes_mod_order(zero); n]; number_of_point_multiplications];    
    let mut s3_pa: Vec<Vec<Scalar>> = vec![vec![Scalar::from_bytes_mod_order(zero); n]; number_of_point_multiplications];    
    let mut t1_pa: Vec<Vec<Scalar>> = vec![vec![Scalar::from_bytes_mod_order(zero); n]; number_of_point_multiplications];    
    let mut t2_pa: Vec<Vec<Scalar>> = vec![vec![Scalar::from_bytes_mod_order(zero); n]; number_of_point_multiplications];    
    let mut t3_pa: Vec<Vec<Scalar>> = vec![vec![Scalar::from_bytes_mod_order(zero); n]; number_of_point_multiplications];    
    let mut t4_pa: Vec<Vec<Scalar>> = vec![vec![Scalar::from_bytes_mod_order(zero); n]; number_of_point_multiplications];
    
    let mut c_pd: Vec<Vec<Scalar>> = vec![vec![Scalar::from_bytes_mod_order(zero); n]; number_of_point_multiplications];
    let mut s1_pd: Vec<Vec<Scalar>> = vec![vec![Scalar::from_bytes_mod_order(zero); n]; number_of_point_multiplications];
    let mut s2_pd: Vec<Vec<Scalar>> = vec![vec![Scalar::from_bytes_mod_order(zero); n]; number_of_point_multiplications];
    let mut t1_pd: Vec<Vec<Scalar>> = vec![vec![Scalar::from_bytes_mod_order(zero); n]; number_of_point_multiplications];
    let mut t2_pd: Vec<Vec<Scalar>> = vec![vec![Scalar::from_bytes_mod_order(zero); n]; number_of_point_multiplications];

    let mut z1: Vec<Vec<Scalar>> = vec![vec![Scalar::from_bytes_mod_order(zero); n]; number_of_point_multiplications];
    let mut z2: Vec<Vec<Scalar>> = vec![vec![Scalar::from_bytes_mod_order(zero); n]; number_of_point_multiplications];
    let mut z3: Vec<Vec<Scalar>> = vec![vec![Scalar::from_bytes_mod_order(zero); n]; number_of_point_multiplications];
    let mut z4: Vec<Vec<Scalar>> = vec![vec![Scalar::from_bytes_mod_order(zero); n]; number_of_point_multiplications];

    let mut qx = vec![Scalar::from_bytes_mod_order(zero); number_of_point_multiplications];
    let mut qy = vec![Scalar::from_bytes_mod_order(zero); number_of_point_multiplications];

    let one = Scalar::one().to_bytes();
    let one_scalar = Scalar::from_bytes_mod_order(one);

    for j in 0..(number_of_point_multiplications){

        for i in 0..n{
            if i == 0{    
                let result_pa = pa(bx0[j], by0[j], bz0[j], ax0[j], ay0[j]);
                //cx[0] is cx1
                cx[j][0] = result_pa.0;
                cy[j][0] = result_pa.1;
                c_pa[j][0] = result_pa.2;
                s1_pa[j][0] = result_pa.3;
                s2_pa[j][0] = result_pa.4;
                s3_pa[j][0] = result_pa.5;
                t1_pa[j][0] = result_pa.6;
                t2_pa[j][0] = result_pa.7;
                t3_pa[j][0] = result_pa.8;
                t4_pa[j][0] = result_pa.9;

                let result_pd = pd(ax0[j], ay0[j], a_pd);
                dx[j][0] = result_pd.0;
                dy[j][0] = result_pd.1;
                t1_pd[j][0] = result_pd.2;
                t2_pd[j][0] = result_pd.3;
                s1_pd[j][0] = result_pd.4;
                s2_pd[j][0] = result_pd.5;
                c_pd[j][0] = result_pd.6;

                z1[j][0] = (cx[j][0] * bits[j][0]);
                z2[j][0] = (bx0[j] * (one_scalar - bits[j][0]));
                bx[j][0] = z1[j][0] + z2[j][0];
                z3[j][0] = (cy[j][0] * bits[j][0]);
                z4[j][0] = (by0[j] * (one_scalar - bits[j][0]));
                by[j][0] = z3[j][0] + z4[j][0];
                bz[j][0] = bz0[j] * (one_scalar - bits[j][0]);

                ax[j][0] = dx[j][0];
                ay[j][0] = dy[j][0];

            }else{
                let result_pa = pa(bx[j][i-1], by[j][i-1], bz[j][i-1], ax[j][i-1], ay[j][i-1]);
                //cx[1] is cx2
                cx[j][i] = result_pa.0;
                cy[j][i] = result_pa.1;
                c_pa[j][i] = result_pa.2;
                s1_pa[j][i] = result_pa.3;
                s2_pa[j][i] = result_pa.4;
                s3_pa[j][i] = result_pa.5;
                t1_pa[j][i] = result_pa.6;
                t2_pa[j][i] = result_pa.7;
                t3_pa[j][i] = result_pa.8;
                t4_pa[j][i] = result_pa.9;

                let result_pd = pd(ax[j][i-1], ay[j][i-1], a_pd);
                dx[j][i] = result_pd.0;
                dy[j][i] = result_pd.1;
                t1_pd[j][i] = result_pd.2;
                t2_pd[j][i] = result_pd.3;
                s1_pd[j][i] = result_pd.4;
                s2_pd[j][i] = result_pd.5;
                c_pd[j][i] = result_pd.6;

                z1[j][i] = (cx[j][i] * bits[j][i]);
                z2[j][i] = (bx[j][i-1] * (one_scalar - bits[j][i]));
                bx[j][i] = z1[j][i] + z2[j][i];
                z3[j][i] = (cy[j][i] * bits[j][i]);
                z4[j][i] = (by[j][i-1] * (one_scalar - bits[j][i]));
                by[j][i] = z3[j][i] + z4[j][i];
                bz[j][i] = bz[j][i-1] * (one_scalar - bits[j][i]);

                ax[j][i] = dx[j][i];
                ay[j][i] = dy[j][i];
            }
        }

        // Qx , Qy
        qx[j] = bx[j][n-1];
        qy[j] = by[j][n-1];

        /*
        if weight_list[j] == 0{
            println!("infinity")
        } else{
            println!("qx {:?}", qx[j]);
            println!("qy {:?}", qy[j]);
        }
        */

    }

    println!("Still working on...");

    // model paramters
    let mut vars_para = vec![Scalar::zero().to_bytes(); num_vars_1];
 
    // input & output & auxiliary witnesses 
    let mut vars_input = vec![Scalar::zero().to_bytes(); num_vars_1];
    
    //create a VarsAssignment
    let mut vars = vec![Scalar::zero().to_bytes(); num_vars_1];


    for j in 0..(number_of_point_multiplications){
        
        // vars_para
        vars_para[n + (one_num_vars * j)] = a[j].to_bytes();

        // vars_input
        vars_input[n+1 + (one_num_vars * j)] = ax0[j].to_bytes();
        vars_input[2*n+2 + (one_num_vars * j)] = ay0[j].to_bytes();
        vars_input[3*n+3 + (one_num_vars * j)] = bx0[j].to_bytes();
        vars_input[4*n+4 + (one_num_vars * j)] = by0[j].to_bytes();
        vars_input[5*n+5 + (one_num_vars * j)] = bz0[j].to_bytes();
        vars_input[10*n+6 + (one_num_vars * j)] = qx[j].to_bytes();
        vars_input[10*n+7 + (one_num_vars * j)] = qy[j].to_bytes();
        vars_input[10*n+8 + (one_num_vars * j)] = px[j].to_bytes();
        vars_input[10*n+9 + (one_num_vars * j)] = py[j].to_bytes();

        // vars
        vars[n + (one_num_vars * j)] = a[j].to_bytes();
        vars[n+1 + (one_num_vars * j)] = ax0[j].to_bytes();
        vars[2*n+2 + (one_num_vars * j)] = ay0[j].to_bytes();
        vars[3*n+3 + (one_num_vars * j)] = bx0[j].to_bytes();
        vars[4*n+4 + (one_num_vars * j)] = by0[j].to_bytes();
        vars[5*n+5 + (one_num_vars * j)] = bz0[j].to_bytes();
        vars[10*n+6 + (one_num_vars * j)] = qx[j].to_bytes();
        vars[10*n+7 + (one_num_vars * j)] = qy[j].to_bytes();
        vars[10*n+8 + (one_num_vars * j)] = px[j].to_bytes();
        vars[10*n+9 + (one_num_vars * j)] = py[j].to_bytes();
 

        for i in 0..n {
            vars_input[i + (one_num_vars * j)] = bits[j][i].to_bytes();
            vars_input[n+2 +i + (one_num_vars * j)] = ax[j][i].to_bytes();
            vars_input[2*n+3 +i + (one_num_vars * j)] = ay[j][i].to_bytes();
            vars_input[3*n+4 +i + (one_num_vars * j)] = bx[j][i].to_bytes();
            vars_input[4*n+5 +i + (one_num_vars * j)] = by[j][i].to_bytes();
            vars_input[5*n+6 +i + (one_num_vars * j)] = bz[j][i].to_bytes();
            vars_input[6*n+6 +i + (one_num_vars * j)] = cx[j][i].to_bytes();
            vars_input[7*n+6 +i + (one_num_vars * j)] = cy[j][i].to_bytes();
            vars_input[8*n+6 +i + (one_num_vars * j)] = dx[j][i].to_bytes();
            vars_input[9*n+6 +i + (one_num_vars * j)] = dy[j][i].to_bytes();
            vars_input[10*n+10 +i + (one_num_vars * j)] = c_pa[j][i].to_bytes();
            vars_input[11*n+10 +i + (one_num_vars * j)] = s1_pa[j][i].to_bytes();
            vars_input[12*n+10 +i + (one_num_vars * j)] = s2_pa[j][i].to_bytes();
            vars_input[13*n+10 +i + (one_num_vars * j)] = s3_pa[j][i].to_bytes();
            vars_input[14*n+10 +i + (one_num_vars * j)] = t1_pa[j][i].to_bytes();
            vars_input[15*n+10 +i + (one_num_vars * j)] = t2_pa[j][i].to_bytes();
            vars_input[16*n+10 +i + (one_num_vars * j)] = t3_pa[j][i].to_bytes();
            vars_input[17*n+10 +i + (one_num_vars * j)] = t4_pa[j][i].to_bytes();
            vars_input[18*n+10 +i + (one_num_vars * j)] = c_pd[j][i].to_bytes();
            vars_input[19*n+10 +i + (one_num_vars * j)] = t1_pd[j][i].to_bytes();
            vars_input[20*n+10 +i + (one_num_vars * j)] = s1_pd[j][i].to_bytes();
            vars_input[21*n+10 +i + (one_num_vars * j)] = s2_pd[j][i].to_bytes();
            vars_input[22*n+10 +i + (one_num_vars * j)] = t2_pd[j][i].to_bytes();
            vars_input[23*n+10 +i + (one_num_vars * j)] = z1[j][i].to_bytes();
            vars_input[24*n+10 +i + (one_num_vars * j)] = z2[j][i].to_bytes();
            vars_input[25*n+10 +i + (one_num_vars * j)] = z3[j][i].to_bytes();
            vars_input[26*n+10 +i + (one_num_vars * j)] = z4[j][i].to_bytes();
        
        
            // vars
            vars[i + (one_num_vars * j)] = bits[j][i].to_bytes();
            vars[n+2 +i + (one_num_vars * j)] = ax[j][i].to_bytes();
            vars[2*n+3 +i + (one_num_vars * j)] = ay[j][i].to_bytes();
            vars[3*n+4 +i + (one_num_vars * j)] = bx[j][i].to_bytes();
            vars[4*n+5 +i + (one_num_vars * j)] = by[j][i].to_bytes();
            vars[5*n+6 +i + (one_num_vars * j)] = bz[j][i].to_bytes();
            vars[6*n+6 +i + (one_num_vars * j)] = cx[j][i].to_bytes();
            vars[7*n+6 +i + (one_num_vars * j)] = cy[j][i].to_bytes();
            vars[8*n+6 +i + (one_num_vars * j)] = dx[j][i].to_bytes();
            vars[9*n+6 +i + (one_num_vars * j)] = dy[j][i].to_bytes();
            vars[10*n+10 +i + (one_num_vars * j)] = c_pa[j][i].to_bytes();
            vars[11*n+10 +i + (one_num_vars * j)] = s1_pa[j][i].to_bytes();
            vars[12*n+10 +i + (one_num_vars * j)] = s2_pa[j][i].to_bytes();
            vars[13*n+10 +i + (one_num_vars * j)] = s3_pa[j][i].to_bytes();
            vars[14*n+10 +i + (one_num_vars * j)] = t1_pa[j][i].to_bytes();
            vars[15*n+10 +i + (one_num_vars * j)] = t2_pa[j][i].to_bytes();
            vars[16*n+10 +i + (one_num_vars * j)] = t3_pa[j][i].to_bytes();
            vars[17*n+10 +i + (one_num_vars * j)] = t4_pa[j][i].to_bytes();
            vars[18*n+10 +i + (one_num_vars * j)] = c_pd[j][i].to_bytes();
            vars[19*n+10 +i + (one_num_vars * j)] = t1_pd[j][i].to_bytes();
            vars[20*n+10 +i + (one_num_vars * j)] = s1_pd[j][i].to_bytes();
            vars[21*n+10 +i + (one_num_vars * j)] = s2_pd[j][i].to_bytes();
            vars[22*n+10 +i + (one_num_vars * j)] = t2_pd[j][i].to_bytes();
            vars[23*n+10 +i + (one_num_vars * j)] = z1[j][i].to_bytes();
            vars[24*n+10 +i + (one_num_vars * j)] = z2[j][i].to_bytes();
            vars[25*n+10 +i + (one_num_vars * j)] = z3[j][i].to_bytes();
            vars[26*n+10 +i + (one_num_vars * j)] = z4[j][i].to_bytes();
        } 
    }
    

    let assignment_vars_para = VarsAssignment::new(&vars_para).unwrap();
    let padded_vars_para = {
        let num_padded_vars = inst_1.inst.get_num_vars();
        let num_vars = assignment_vars_para.assignment.len();
        let padded_vars = if num_padded_vars > num_vars {
            assignment_vars_para.pad(num_padded_vars)
        } else {
            assignment_vars_para
        };
        padded_vars
    };
    
    let assignment_vars_input = VarsAssignment::new(&vars_input).unwrap();
    let padded_vars_input = {
        let num_padded_vars = inst_1.inst.get_num_vars();
        let num_vars = assignment_vars_input.assignment.len();
        let padded_vars = if num_padded_vars > num_vars {
            assignment_vars_input.pad(num_padded_vars)
        } else {
            assignment_vars_input
        };
        padded_vars
    };

    let assignment_vars = VarsAssignment::new(&vars).unwrap();

    let padded_vars = {
        let num_padded_vars = inst_1.inst.get_num_vars();
        let num_vars = assignment_vars.assignment.len();
        let padded_vars = if num_padded_vars > num_vars {
            assignment_vars.pad(num_padded_vars)
        } else {
            assignment_vars.clone()
        };
        padded_vars
    };

    // create an InputAssignment
    let mut inputs = vec![Scalar::zero().to_bytes(); num_inputs_1];

    inputs[0] = a_pd.to_bytes();
    
    let assignment_inputs = InputsAssignment::new(&inputs).unwrap();

    // check if the instance we created is satisfiable
    let res = inst_1.is_sat(&assignment_vars, &assignment_inputs);
    assert_eq!(res.unwrap(), true);

    (
        num_cons_1,
        num_vars_1,
        num_inputs_1,
        num_non_zero_entries_1,
        inst_1,
        padded_vars_para,
        padded_vars_input,
        padded_vars,
        assignment_inputs
        )
}


fn pa(bx: Scalar, by: Scalar, bz: Scalar, ax: Scalar, ay: Scalar) -> (Scalar, Scalar, Scalar, Scalar, Scalar, Scalar, Scalar, Scalar, Scalar, Scalar) {
    let one = Scalar::one().to_bytes();
    let one_scalar = Scalar::from_bytes_mod_order(one);

    let c = (bx - ax).invert();
    let s1 = (by - ay) * c;
    let s2 = s1 * s1;
    let t1 = (s2 - ax - bx) * (one_scalar - bz);
    let t2 = ax * bz;
    let cx = (t1 + t2) * one_scalar;
    let s3 = s1 * (ax - cx);
    let t3 = (s3 - ay) * (one_scalar - bz);
    let t4 = ay * bz;
    let cy = (t3 + t4) * one_scalar;

    let result = (cx, cy, c, s1, s2, s3, t1, t2, t3, t4);

    result
}

fn pd(ax: Scalar, ay: Scalar, a: Scalar) -> (Scalar, Scalar, Scalar, Scalar, Scalar, Scalar, Scalar) {
    let two = (Scalar::one() + Scalar::one()).to_bytes();
    let three = (Scalar::one() + Scalar::one() + Scalar::one()).to_bytes();
    let two_scalar = Scalar::from_bytes_mod_order(two);
    let three_scalar = Scalar::from_bytes_mod_order(three);

    let c = (two_scalar * ay).invert();
    let t1 = (ax * ax);
    let s1 = ((three_scalar * t1) + a) * c;
    let s2 = s1 * s1;
    let dx = (s2 - (two_scalar * ax));
    let t2 = s1 * (ax - dx);
    let dy = t2 - ay;

    let result = (dx, dy, t1, t2, s1, s2, c);

    result
}

fn u128_to_128_bit_string(num: u128, n: usize) -> String {
    let mut result = String::new();

    for i in (0..n).rev() {
        let bit = (num >> i) & 1;
        result.push_str(&bit.to_string());
    }

    result
}

fn process_bit_string(bit_string: &str) -> Vec<Scalar> {
    let mut bits: Vec<Scalar> = Vec::new();

    for bit_char in bit_string.chars().rev() {
        match bit_char {
            '1' => bits.push(Scalar::one()),
            '0' => bits.push(Scalar::zero()),
            _ => panic!("Invalid character in bit string!"),
        }
    }

    bits
}
