use libspartan::{Instance, VarsAssignment, InputsAssignment};
use curve25519_dalek::scalar::Scalar;
use crate::load_data_add::load_data_add;

pub fn point_addition(network: &str) -> (
    usize,
    usize,
    usize,
    usize,
    Instance,
    VarsAssignment,
    VarsAssignment,
    VarsAssignment,
    InputsAssignment,
   ) {

    // Read the JSON file
    let (len, point_add_px_byte, point_add_py_byte, point_add_rx_byte, point_add_ry_byte, point_add_rz_byte) = load_data_add(network);

    println!("Point Addition Gadget...");
    println!("Number of Point Additions: {}", len);

    // PA (Point Addition): Point P + Point R

    // c * (Rx - Px) = 1
    // (Ry - Py) * c = s1
    // s1 * s1 = s2
    // (s2 - Px - Rx) * (1 - Rz) = t1
    // px * Rz = t2
    // (t1 + t2) * 1 = x3
    // s1 * (Px - x3) = s3
    // (s3 - Py) * (1 - Rz) = t3 
    // Py * Rz = t4
    // (t3 + t4) * 1 = y3

    let number_of_point_additions = len;

    let (param_1, param_2, param_3);

    if number_of_point_additions < 780 {
        param_1 = 2;
        param_2 = 25;
        param_3 = 3;
    }
    else if number_of_point_additions >= 500 && number_of_point_additions < 780 {
        param_1 = 2;
        param_2 = 25;
        param_3 = 3;
    } else if number_of_point_additions > 2130 && number_of_point_additions < 2150 {
        param_1 = 5;
        param_2 = 30;
        param_3 = 5;
    } else if number_of_point_additions > 2149 && number_of_point_additions < 2450 {
        param_1 = 3;
        param_2 = 30;
        param_3 = 5;
    } else if number_of_point_additions > 5000 && number_of_point_additions < 8000 {
        param_1 = 3;
        param_2 = 20;
        param_3 = 5;        
    } else {
        param_1 = 5;
        param_2 = 30;
        param_3 = 5;
    } 

    let num_cons_1 = 10 * number_of_point_additions; 
    let num_vars_1 = (15 * number_of_point_additions) + 1; //c, Rx, Px, Ry, Py, Rz, s1, s2, s3, t1, t2, t3, t4, x3, y3  
    let num_inputs_1 = 0; //
    let num_non_zero_entries_1 = param_1 * (param_2/param_3) * number_of_point_additions; //c, Rx, Px, Ry, Py, Rz, s1, s2, s3, t1, t2, t3, t4, x3

    let mut A1: Vec<(usize, usize, [u8; 32])> = Vec::new();
    let mut B1: Vec<(usize, usize, [u8; 32])> = Vec::new();
    let mut C1: Vec<(usize, usize, [u8; 32])> = Vec::new();

    // a variable that holds a byte representation of 1
    let zero = Scalar::zero().to_bytes();
    let one = Scalar::one().to_bytes();
    let minus_one = (Scalar::zero() - Scalar::one()).to_bytes();

    for i in 0..number_of_point_additions{
        // constraint 0 entries in (A,B,C)
        // c * (Rx - Px) = 1
        A1.push((0 + (10 * i), 0 + (15 * i), one));
        B1.push((0 + (10 * i), 1 + (15 * i), one));
        B1.push((0 + (10 * i), 2 + (15 * i), minus_one));
        C1.push((0 + (10 * i), num_vars_1, one));

        // constraint 1 entries in (A,B,C)
        // (Ry - Py) * c = s1
        A1.push((1 + (10 * i), 3 + (15 * i), one));
        A1.push((1 + (10 * i), 4 + (15 * i), minus_one));
        B1.push((1 + (10 * i), 0 + (15 * i), one));
        C1.push((1 + (10 * i), 6 + (15 * i), one));

        // constraint 2 entries in (A,B,C)
        // s1 * s1 = s2
        A1.push((2 + (10 * i), 6 + (15 * i), one));
        B1.push((2 + (10 * i), 6 + (15 * i), one));
        C1.push((2 + (10 * i), 7 + (15 * i), one));

        // constraint 3 entries in (A,B,C)
        // (s2 - Px - Rx) * (1 - Rz) = t1
        A1.push((3 + (10 * i), 7 + (15 * i), one));
        A1.push((3 + (10 * i), 2 + (15 * i), minus_one));
        A1.push((3 + (10 * i), 1 + (15 * i), minus_one));
        B1.push((3 + (10 * i), num_vars_1, one));
        B1.push((3 + (10 * i), 5 + (15 * i), minus_one));
        C1.push((3 + (10 * i), 9 + (15 * i), one));

        // constraint 4 entries in (A,B,C)
        // px * Rz = t2
        A1.push((4 + (10 * i), 2 + (15 * i), one));
        B1.push((4 + (10 * i), 5 + (15 * i), one));
        C1.push((4 + (10 * i), 10 + (15 * i), one));

        // constraint 5 entries in (A,B,C)
        // (t1 + t2) * 1 = x3
        A1.push((5 + (10 * i), 9 + (15 * i), one));
        A1.push((5 + (10 * i), 10 + (15 * i), one));
        B1.push((5 + (10 * i), num_vars_1, one));
        C1.push((5 + (10 * i), 13 + (15 * i), one));

        // constraint 6 entries in (A,B,C)
        // s1 * (Px - x3) = s3
        A1.push((6 + (10 * i), 6 + (15 * i), one));
        B1.push((6 + (10 * i), 2 + (15 * i), one));
        B1.push((6 + (10 * i), 13 + (15 * i), minus_one));
        C1.push((6 + (10 * i), 8 + (15 * i), one));

        // constraint 7 entries in (A,B,C)
        // (s3 - Py) * (1 - Rz) = t3 
        A1.push((7 + (10 * i), 8 + (15 * i), one));
        A1.push((7 + (10 * i), 4 + (15 * i), minus_one));
        B1.push((7 + (10 * i), num_vars_1, one));
        B1.push((7 + (10 * i), 5 + (15 * i), minus_one));    
        C1.push((7 + (10 * i), 11 + (15 * i), one));

        // constraint 8 entries in (A,B,C)
        // Py * Rz = t4
        A1.push((8 + (10 * i), 4 + (15 * i), one));
        B1.push((8 + (10 * i), 5 + (15 * i), one));
        C1.push((8 + (10 * i), 12 + (15 * i), one)); 

        // constraint 9 entries in (A,B,C)
        // (t3 + t4) * 1 = y3
        A1.push((9 + (10 * i), 11 + (15 * i), one));
        A1.push((9 + (10 * i), 12 + (15 * i), one));
        B1.push((9 + (10 * i), num_vars_1, one));
        C1.push((9 + (10 * i), 14 + (15 * i), one));
    }

    let inst_1 = Instance::new(num_cons_1, num_vars_1, num_inputs_1, &A1, &B1, &C1).unwrap();

    // compute a satisfying assignment

    let mut px = vec![Scalar::from_bytes_mod_order(zero); number_of_point_additions];
    let mut py = vec![Scalar::from_bytes_mod_order(zero); number_of_point_additions];
    let pz = Scalar::from_bytes_mod_order(zero); // infinity point has rz = 1;
    let mut rx = vec![Scalar::from_bytes_mod_order(zero); number_of_point_additions];
    let mut ry = vec![Scalar::from_bytes_mod_order(zero); number_of_point_additions];
    let mut rz = vec![Scalar::from_bytes_mod_order(zero); number_of_point_additions];

    for i in 0..number_of_point_additions{
        let mut px_byte: [u8; 32] = [0; 32];
        let mut py_byte: [u8; 32] = [0; 32];
        let mut rx_byte: [u8; 32] = [0; 32];
        let mut ry_byte: [u8; 32] = [0; 32];

        for (j, &value) in point_add_px_byte[i].iter().enumerate() {
            px_byte[j] = value as u8;
        }
        for (j, &value) in point_add_py_byte[i].iter().enumerate() {
            py_byte[j] = value as u8;
        }
        for (j, &value) in point_add_rx_byte[i].iter().enumerate() {
            rx_byte[j] = value as u8;
        }
        for (j, &value) in point_add_ry_byte[i].iter().enumerate() {
            ry_byte[j] = value as u8;
        }

        px[i] = Scalar::from_bytes_mod_order(px_byte);
        py[i] = Scalar::from_bytes_mod_order(py_byte);
        rx[i] = Scalar::from_bytes_mod_order(rx_byte);
        ry[i] = Scalar::from_bytes_mod_order(ry_byte);

        if point_add_rz_byte[i] == 0{
            rz[i] = Scalar::from_bytes_mod_order(zero);
        }else{
            rz[i] = Scalar::from_bytes_mod_order(one); // infinity point has rz = 1; 
        }
    
    }

    let mut c = vec![Scalar::from_bytes_mod_order(zero); number_of_point_additions];
    let mut s1 = vec![Scalar::from_bytes_mod_order(zero); number_of_point_additions];
    let mut s2 = vec![Scalar::from_bytes_mod_order(zero); number_of_point_additions];
    let mut s3 = vec![Scalar::from_bytes_mod_order(zero); number_of_point_additions];
    let mut t1 = vec![Scalar::from_bytes_mod_order(zero); number_of_point_additions];
    let mut t2 = vec![Scalar::from_bytes_mod_order(zero); number_of_point_additions];
    let mut t3 = vec![Scalar::from_bytes_mod_order(zero); number_of_point_additions];
    let mut t4 = vec![Scalar::from_bytes_mod_order(zero); number_of_point_additions];
    let mut x3 = vec![Scalar::from_bytes_mod_order(zero); number_of_point_additions];
    let mut y3 = vec![Scalar::from_bytes_mod_order(zero); number_of_point_additions];

    let one_scalar = Scalar::from_bytes_mod_order(one);

    for i in 0..(number_of_point_additions){
        c[i] = (rx[i] - px[i]).invert();
        s1[i] = (ry[i] - py[i]) * c[i];
        s2[i] = s1[i] * s1[i];
        t1[i] = (s2[i] - px[i] - rx[i]) * (one_scalar - rz[i]);
        t2[i] = px[i] * rz[i];
        x3[i] = (t1[i] + t2[i]) * one_scalar;
        s3[i] = s1[i] * (px[i] - x3[i]);
        t3[i] = (s3[i] - py[i]) * (one_scalar - rz[i]);
        t4[i] = py[i] * rz[i];
        y3[i] = (t3[i] + t4[i]) * one_scalar;
    }
    
    // model paramters
    let mut vars_para = vec![Scalar::zero().to_bytes(); num_vars_1];
    //vars_para[0] = c.to_bytes();

    // input & output & auxiliary witnesses 
    let mut vars_input = vec![Scalar::zero().to_bytes(); num_vars_1];
    
    // create a VarsAssignment
    let mut vars_1 = vec![Scalar::zero().to_bytes(); num_vars_1];


    for i in 0..(number_of_point_additions){

        vars_input[0 + (15 * i)] = c[i].to_bytes();
        vars_input[1 + (15 * i)] = rx[i].to_bytes();
        vars_input[2 + (15 * i)] = px[i].to_bytes();
        vars_input[3 + (15 * i)] = ry[i].to_bytes();
        vars_input[4 + (15 * i)] = py[i].to_bytes();
        vars_input[5 + (15 * i)] = rz[i].to_bytes();
        vars_input[6 + (15 * i)] = s1[i].to_bytes();
        vars_input[7 + (15 * i)] = s2[i].to_bytes();
        vars_input[8 + (15 * i)] = s3[i].to_bytes();
        vars_input[9 + (15 * i)] = t1[i].to_bytes();
        vars_input[10 + (15 * i)] = t2[i].to_bytes();
        vars_input[11 + (15 * i)] = t3[i].to_bytes();   
        vars_input[12 + (15 * i)] = t4[i].to_bytes();    
        vars_input[13 + (15 * i)] = x3[i].to_bytes();
        vars_input[14 + (15 * i)] = y3[i].to_bytes();
    

        vars_1[0 + (15 * i)] = c[i].to_bytes();
        vars_1[1 + (15 * i)] = rx[i].to_bytes();
        vars_1[2 + (15 * i)] = px[i].to_bytes();
        vars_1[3 + (15 * i)] = ry[i].to_bytes();
        vars_1[4 + (15 * i)] = py[i].to_bytes();
        vars_1[5 + (15 * i)] = rz[i].to_bytes();
        vars_1[6 + (15 * i)] = s1[i].to_bytes();
        vars_1[7 + (15 * i)] = s2[i].to_bytes();
        vars_1[8 + (15 * i)] = s3[i].to_bytes();
        vars_1[9 + (15 * i)] = t1[i].to_bytes();
        vars_1[10 + (15 * i)] = t2[i].to_bytes();
        vars_1[11 + (15 * i)] = t3[i].to_bytes();   
        vars_1[12 + (15 * i)] = t4[i].to_bytes();    
        vars_1[13 + (15 * i)] = x3[i].to_bytes();
        vars_1[14 + (15 * i)] = y3[i].to_bytes();
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


    let assignment_vars_1 = VarsAssignment::new(&vars_1).unwrap();
    let padded_vars = {
        let num_padded_vars = inst_1.inst.get_num_vars();
        let num_vars = assignment_vars_1.assignment.len();
        let padded_vars = if num_padded_vars > num_vars {
            assignment_vars_1.pad(num_padded_vars)
        } else {
            assignment_vars_1.clone()
        };
        padded_vars
    };


    // create an InputsAssignment
    let mut inputs_1 = vec![Scalar::zero().to_bytes(); num_inputs_1];
    let assignment_inputs_1 = InputsAssignment::new(&inputs_1).unwrap();

    //check if the instance we created is satisfiable
    let res_1 = inst_1.is_sat(&assignment_vars_1, &assignment_inputs_1);
    assert_eq!(res_1.unwrap(), true);

    (
    num_cons_1,
    num_vars_1,
    num_inputs_1,
    num_non_zero_entries_1,
    inst_1,
    padded_vars_para,
    padded_vars_input,
    padded_vars,
    assignment_inputs_1,
    )
}

