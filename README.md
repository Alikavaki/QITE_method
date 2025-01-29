# **Quantum Imaginary Time Evolution (QITE) Documentation**

## **Overview**

This repository implements the **Quantum Imaginary Time Evolution (QITE)** method, enabling efficient simulation of quantum systems. The code supports arbitrary qubit pairings and leverages sparse matrix operations for scalable performance.

## **Modules and Functions**

### **1. Sparse Matrix Operations**

#### `generate_sparse_tensor_product_iterative(operators: list) -> object`

Generates the tensor product of sparse matrices iteratively for efficient computation.

- **Parameters:**
  - `operators (list)`: List of sparse matrices to tensor product.
- **Returns:**
  - Sparse matrix representing the tensor product.

#### `get_sparse_pauli_matrices() -> tuple`

Returns sparse representations of the identity and Pauli matrices (I, X, Y, Z).

- **Returns:**
  - Tuple `(I, X, Y, Z)` of sparse matrices.

#### `create_sparse_operators(n_qubits: int, i: int, target: object, operator1: object, operator2: object) -> list`

Creates sparse operators for a given qubit and an optional target qubit.

- **Parameters:**
  - `n_qubits (int)`: Number of qubits.
  - `i (int)`: Index of the primary qubit.
  - `target (object)`: Index of the target qubit (optional).
  - `operator1 (object)`: Operator for the primary qubit.
  - `operator2 (object)`: Operator for the target qubit.
- **Returns:**
  - List of sparse operators for the system.

### **2. Basis Generation**

#### `basis(n_qubits: int) -> list`

Generates a basis set of sparse matrices for QITE.

- **Parameters:**
  - `n_qubits (int)`: Number of qubits.
- **Returns:**
  - List of basis matrices.

### **3. Commutator and Expectation Computation**

#### `b_matrix(hamiltonian: object, matrices: list, psi: object) -> list`

Computes the `b` vector using the commutator `[H, A]` for each A in the basis.

- **Parameters:**
  - `hamiltonian (csc_matrix)`: Sparse Hamiltonian matrix.
  - `matrices (list)`: List of sparse matrices.
  - `psi (csc_matrix)`: Sparse quantum state.
- **Returns:**
  - List of complex numbers representing the `b` vector.

#### `compute_s_matrices(matrices: list, psi: object) -> object`

Computes the `S` matrix, capturing inner products between basis matrices.

- **Parameters:**
  - `matrices (list)`: List of sparse matrices.
  - `psi (csc_matrix)`: Sparse quantum state.
- **Returns:**
  - Sparse `S` matrix.

### **4. Iterative Solver**

#### `iterative_solver(hamiltonian: object, matrices: list, psi: object, tau: float, num_iterations: int, a: float, r: float) -> tuple`

Performs iterative updates using QITE to evolve the quantum state.

- **Parameters:**
  - `hamiltonian (csc_matrix)`: Sparse Hamiltonian matrix.
  - `matrices (list)`: List of sparse matrices.
  - `psi (csc_matrix)`: Initial sparse state.
  - `tau (float)`: Timestep for evolution.
  - `num_iterations (int)`: Number of iterations.
  - `a (float)`: Absolute tolerance for pseudoinverse.
  - `r (float)`: Relative tolerance for pseudoinverse.
- **Returns:**
  - Tuple `(psi_list, coefficients_list)` representing the evolved states and coefficients.

### **5. Operator Construction**

#### `construct_operator(matrices: list, coefficients: array) -> object`

Constructs an operator from a set of matrices and coefficients.

- **Parameters:**
  - `matrices (list)`: List of sparse matrices.
  - `coefficients (array)`: Array of complex coefficients.
- **Returns:**
  - Sparse operator matrix.

## **Usage**

### **Example Workflow**

```python
# Define system size
n_qubits = 4

# Generate basis
basis_matrices = basis(n_qubits)

# Define Hamiltonian (example)
H = csc_matrix(np.random.rand(2**n_qubits, 2**n_qubits))

# Define initial state |psi>
psi = csc_matrix(np.random.rand(2**n_qubits, 1))

# Run iterative solver
psi_list, coefficients_list = iterative_solver(H, basis_matrices, psi, tau=0.1, num_iterations=10, a=1e-5, r=1e-5)
```

## **Performance Considerations**

- Uses **sparse matrix operations** to optimize efficiency.
- Implements **Kronecker products iteratively** for better scalability.
- Utilizes **least squares solutions** for the coefficient matrix computation.

## **Contributing**

Contributions are welcome! Feel free to submit issues or pull requests to improve the implementation.

---

This documentation provides a structured guide for using the QITE implementation effectively. Let me know if you need revisions! 🚀

Sounds good, how can I upload it to my github account, should I put it in the readme file?

