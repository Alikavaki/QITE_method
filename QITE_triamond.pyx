from scipy.sparse import lil_matrix, csc_matrix, eye, kron, isspmatrix
from scipy.sparse.linalg import expm_multiply, lsqr
from scipy.linalg import pinv
import numpy as np
cimport numpy as cnp
from time import time  # Import the time module for runtime tracking

#################################################################
# Efficient sparse tensor product computation (now uses LIL during construction)
cpdef object generate_sparse_tensor_product_iterative(list operators):
    cdef object result = eye(1, format="lil", dtype=np.complex128)  # Start with LIL
    for op in operators:
        result = kron(result, op, format="lil")  # Build with LIL
    return result.tocsc()  # Convert to CSC for storage/operations

# Get sparse Pauli matrices (now LIL-based)
def get_sparse_pauli_matrices():
    cdef object I = eye(2, format="lil", dtype=np.complex128)
    cdef object X = lil_matrix([[0, 1], [1, 0]], dtype=np.complex128)
    cdef object Y = lil_matrix([[0, -1j], [1j, 0]], dtype=np.complex128)
    cdef object Z = lil_matrix([[1, 0], [0, -1]], dtype=np.complex128)
    return I, X, Y, Z  # Return LIL matrices for construction

# Create sparse operators (now uses LIL during initialization)
cpdef list create_sparse_operators(
    int n_qubits,
    int i,
    object target,
    object operator1,
    object operator2
):
    # Initialize with LIL matrices
    cdef list ops = [eye(2, format="lil", dtype=np.complex128)] * n_qubits
    ops[i] = operator1.tolil() if isspmatrix(operator1) else operator1
    if target is not None:
        ops[int(target) % n_qubits] = operator2.tolil() if isspmatrix(operator2) else operator2
    return ops

# Generate sparse basis (unchanged except for underlying matrix format)
cpdef list basis(int n_qubits):
    cdef list matrices = []
    cdef int i, offset
    cdef object matrix
    I, X, Y, Z = get_sparse_pauli_matrices()

    # Single-qubit Y gates
    for i in range(n_qubits):
        ops = create_sparse_operators(n_qubits, i, None, Y, None)
        matrix = generate_sparse_tensor_product_iterative(ops)
        matrices.append(matrix)

    # Pairs of operators
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
# Remaining functions unchanged except type annotations
cpdef list b_matrix(object hamiltonian, list matrices, object psi):
    cdef list b = []
    cdef object commutator
    cdef object result
    cdef cnp.complex128_t expectation_value

    for matrix in matrices:
        commutator = hamiltonian @ matrix - matrix @ hamiltonian
        result = psi.getH() @ commutator @ psi
        b.append(-1j * (result.data[0] if result.nnz > 0 else 0))
    return b
##############################################################################################
cpdef object compute_s_matrices(list matrices, object psi):
    from scipy.sparse import lil_matrix
    cdef int n_matrices = len(matrices)
    cdef object s_matrices = lil_matrix((n_matrices, n_matrices), dtype=np.complex128)
    cdef object matrix_i, matrix_j, intermediate
    cdef int i, j
    cdef cnp.complex128_t value

    for i in range(n_matrices):
        for j in range(n_matrices):
            matrix_i = matrices[i].getH()
            matrix_j = matrices[j]
            intermediate = matrix_i @ matrix_j @ psi
            result = psi.getH() @ intermediate
            value = result.data[0] if intermediate.nnz > 0 and result.nnz > 0 else 0
            s_matrices[i, j] = value

    return (s_matrices + s_matrices.getH()).tocsc()  # Return CSC for symmetry
###################################################################################################
cpdef tuple iterative_solver(object hamiltonian, list matrices, object psi, 
                            double tau, int num_iterations, double a, double r):
    cdef list psi_list = [psi]
    cdef list coefficients_list = []
    cdef object s_matrix, b_vector, operator
    cdef cnp.ndarray[cnp.complex128_t, ndim=1] coefficients

    for iteration in range(num_iterations):
        #Start tracking the runtime
        start_time = time()
        s_matrix = compute_s_matrices(matrices, psi)
        b_vector = b_matrix(hamiltonian, matrices, psi)
        b_array = np.array(b_vector, dtype=np.complex128).reshape(-1, 1)

        if isinstance(s_matrix, csc_matrix):
            coefficients = -lsqr(s_matrix, b_array)[0]
        else:
            s_pinv = pinv(s_matrix.toarray(), atol=a, rtol=r).astype(np.complex128)
            coefficients = -s_pinv @ b_array

        coefficients_list.append(coefficients.flatten().tolist())
        operator = construct_operator(matrices, coefficients)
        psi = expm_multiply(-1j * tau * operator, psi)
        psi_list.append(psi)
        end_time = time()
        runtime = end_time - start_time
        #Log iteration completion
        print(f"Iteration {iteration + 1}/{num_iterations} completed.", flush=True)
        # Print runtime
        print(f"Iterative solver completed in {runtime:.2f} seconds.", flush=True)

    return psi_list, coefficients_list
#####################################################################################################
cpdef object construct_operator(list matrices, cnp.ndarray[cnp.complex128_t, ndim=1] coefficients):
    """
    Construct the operator from the given matrices and coefficients.
    """
    # Changed variable name from 'operator' to 'op' to avoid C++ keyword conflict
    cdef object op = csc_matrix(matrices[0].shape, dtype=np.complex128)
    cdef int i
    
    for i in range(len(matrices)):
        op += coefficients[i] * matrices[i]
    
    return op
















