from scipy.sparse import csc_matrix, eye, kron, isspmatrix
from scipy.sparse.linalg import expm_multiply, lsqr
from scipy.linalg import pinv
import numpy as np
cimport numpy as cnp
#################################################################

# Efficient sparse tensor product computation
cpdef object generate_sparse_tensor_product_iterative(list operators):
    cdef object result = eye(1, format="csc", dtype=np.complex128)  # Updated to np.complex128
    for op in operators:
        result = kron(result, op, format="csc")  # Sparse Kronecker product
    return result


# Get sparse Pauli matrices
def get_sparse_pauli_matrices():
    cdef object I = eye(2, format="csc", dtype=np.complex128)  # Updated to np.complex128
    cdef object X = csc_matrix([[0, 1], [1, 0]], dtype=np.complex128)
    cdef object Y = csc_matrix([[0, -1j], [1j, 0]], dtype=np.complex128)
    cdef object Z = csc_matrix([[1, 0], [0, -1]], dtype=np.complex128)
    return I, X, Y, Z


# Create sparse operators
cpdef list create_sparse_operators(
    int n_qubits,
    int i,
    object target,  # Allow `None`
    object operator1,
    object operator2
):
    cdef list ops = [eye(2, format="csc", dtype=np.complex128)] * n_qubits  # Updated to np.complex128
    ops[i] = operator1
    if target is not None:
        ops[int(target) % n_qubits] = operator2
    return ops


# Generate sparse basis
cpdef list basis(int n_qubits):
    cdef list matrices = []  # Final list of sparse matrices
    cdef int i, offset
    cdef object matrix
    I, X, Y, Z = get_sparse_pauli_matrices()

    # Part 1: Single-qubit Y gates
    for i in range(n_qubits):
        ops = create_sparse_operators(n_qubits, i, None, Y, None)  # Y on qubit i
        matrix = generate_sparse_tensor_product_iterative(ops)
        matrices.append(matrix)

    # Part 2: Pairs of operators
    for i in range(n_qubits):
        for offset in range(1, (n_qubits // 2) + 1):
            if offset < n_qubits // 2:
                configs = [
                    (i, (i + offset) % n_qubits, Y, Z),
                    (i, (i + offset) % n_qubits, Y, X),
                    ((i + offset) % n_qubits, i, Y, Z),
                    ((i + offset) % n_qubits, i, Y, X),
                ]
            else:
                configs = [
                    (i, (i + offset) % n_qubits, Y, Z),
                    (i, (i + offset) % n_qubits, Y, X),
                ]

            for config in configs:
                ops = create_sparse_operators(n_qubits, *config)
                matrix = generate_sparse_tensor_product_iterative(ops)
                matrices.append(matrix)

    return matrices
################################################################################


cpdef list b_matrix(
    object hamiltonian,  # Sparse Hamiltonian
    list matrices,       # List of sparse matrices
    object psi           # Sparse |psi| (csc_matrix)
):
    """
    Compute the b vector using the commutator [H, A] for each matrix A in matrices.

    Parameters:
        hamiltonian (csc_matrix): The Hamiltonian matrix (sparse format).
        matrices (list of csc_matrix): List of sparse tensor product matrices.
        psi (csc_matrix): The state vector |psi| (sparse column matrix).

    Returns:
        list: The b vector as a list of complex numbers.
    """
    cdef list b = []
    cdef object commutator  # Sparse commutator
    cdef object result      # Result of dot product
    cdef cnp.complex128_t expectation_value

    for matrix in matrices:
        # Compute the commutator [H, A] = H @ A - A @ H
        commutator = hamiltonian @ matrix - matrix @ hamiltonian

        # Compute the expectation value: psi.H @ commutator @ psi
        result = psi.getH() @ commutator @ psi  # Sparse matrix multiplications
        if result.nnz > 0:  # If the result is non-empty
            expectation_value = result.data[0]  # Extract the first value
        else:
            expectation_value = 0

        b.append(-1j * expectation_value)

    return b

#####################################################################################


cpdef object compute_s_matrices(list matrices, object psi):
    """
    Compute the S matrix for the system.

    Parameters:
        matrices (list of csc_matrix): List of sparse tensor product matrices.
        psi (csc_matrix): The state vector |psi| as a sparse column matrix.

    Returns:
        csc_matrix: The S matrix as a sparse matrix.
    """
    from scipy.sparse import csc_matrix
    cdef int n_matrices = len(matrices)
    cdef object s_matrices = csc_matrix((n_matrices, n_matrices), dtype=np.complex128)
    cdef object matrix_i, matrix_j, intermediate
    cdef int i, j
    cdef cnp.complex128_t value

    for i in range(n_matrices):
        for j in range(n_matrices):
            # Compute <psi| matrix_i† matrix_j |psi>
            matrix_i = matrices[i].getH()  # Hermitian conjugate of matrix_i
            matrix_j = matrices[j]
            intermediate = matrix_i @ matrix_j @ psi  # Sparse multiplication
            
            # Check if intermediate is non-empty
            if intermediate.nnz > 0:
                result = psi.getH() @ intermediate  # psi† @ intermediate
                if result.nnz > 0:
                    value = result.data[0]  # Extract the scalar
                else:
                    value = 0
            else:
                value = 0

            s_matrices[i, j] = value  # Populate the S-matrix

    # Ensure S-matrix is symmetric
    return s_matrices + s_matrices.getH()



######################################################################################


from time import time  # Import the time module for runtime tracking

cpdef tuple iterative_solver(
    object hamiltonian,
    list matrices,
    object psi,  # Sparse |psi| (csc_matrix)
    double tau,
    int num_iterations,
    double a,
    double r
):
    """
    Perform an iterative update on the state vector |psi> and return results.

    Parameters:
        hamiltonian (csc_matrix): Sparse Hamiltonian matrix.
        matrices (list): List of sparse matrices for the iterative process.
        psi (csc_matrix): Initial state vector (sparse column matrix).
        tau (float): Time step for the exponential operator.
        num_iterations (int): Number of iterations to perform.
        my_rcond (float): Cutoff for pseudoinverse computation.

    Returns:
        tuple: (psi_list, coefficients_list)
    """
    cdef list psi_list = [psi]
    cdef list coefficients_list = []
    cdef object s_matrix, b_vector, operator
    cdef cnp.ndarray[cnp.complex128_t, ndim=1] coefficients
    cdef int i, iteration

    

    for iteration in range(num_iterations):
        # Start tracking the runtime
        start_time = time()
        # Step 1: Compute S-matrix
        s_matrix = compute_s_matrices(matrices, psi)

        # Step 2: Compute b-matrix
        b_vector = b_matrix(hamiltonian, matrices, psi)
        b_array = np.array(b_vector, dtype=np.complex128).reshape(-1, 1)

        # Step 3: Compute pseudoinverse of S-matrix and coefficients
        if isinstance(s_matrix, csc_matrix):
            coefficients = -lsqr(s_matrix, b_array)[0]
        else:
            s_dense = s_matrix.toarray()
            s_pinv = pinv(s_dense, atol=a, rtol=r).astype(np.complex128)
            coefficients = -s_pinv @ b_array

        coefficients = coefficients.flatten()
        coefficients_list.append(coefficients.tolist())

        # Step 4: Construct operator from coefficients
        operator = construct_operator(matrices, coefficients)

        # Step 5: Update psi using the matrix exponential
        psi = expm_multiply(-1j * tau * operator, psi)
        psi_list.append(psi)

        # Log iteration completion
        print(f"Iteration {iteration + 1}/{num_iterations} completed.", flush=True)
        # End tracking runtime
        end_time = time()
        runtime = end_time - start_time
        # Print runtime
        print(f"Iterative solver completed in {runtime:.2f} seconds.", flush=True)
    

    return psi_list, coefficients_list


######################################################################################################


cpdef object construct_operator(list matrices, cnp.ndarray[cnp.complex128_t, ndim=1] coefficients):
    """
    Construct the operator from the given matrices and coefficients.

    Parameters:
        matrices (list): List of sparse matrices (csc_matrix).
        coefficients (cnp.ndarray[cnp.complex128_t, ndim=1]): Array of coefficients.

    Returns:
        object: Sparse operator (csc_matrix).
    """
    cdef object operator  # Declare as a general Python object
    cdef int i

    # Initialize the operator as a zero sparse matrix with the same shape as the first matrix
    operator = csc_matrix(matrices[0].shape, dtype=np.complex128)

    # Sum the coefficients * matrices
    for i in range(len(matrices)):
        operator += coefficients[i] * matrices[i]

    return operator
















