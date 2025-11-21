# Sparse Linear Algebra Examples

This directory contains comprehensive examples demonstrating the Sparse Linear Algebra domain in Kairo. These examples showcase efficient solutions to large-scale linear systems using sparse matrix formats and iterative solvers.

## Overview

The Sparse Linear Algebra domain provides:
- **Sparse matrix formats** (CSR, CSC, COO) for memory-efficient storage
- **Iterative solvers** (CG, BiCGSTAB, GMRES) for large systems
- **Sparse factorizations** (incomplete Cholesky, incomplete LU) for preconditioning
- **Discrete operators** (Laplacian, gradient, divergence) for PDEs

**Key Benefits:**
- Handle systems with 100K+ unknowns efficiently
- Memory usage scales with nonzeros, not matrix size
- Fast iterative solvers (often faster than direct methods)
- Essential for PDEs, circuit simulation, graph algorithms

---

## Examples

### 1. Heat Equation Solver (`01_heat_equation.py`)

Demonstrates solving the heat equation using sparse Laplacian operators and iterative solvers.

**Topics Covered:**
- 1D and 2D heat equation
- Implicit time-stepping for stability
- Conjugate Gradient (CG) solver
- Convergence analysis across grid sizes

**Equations:**
```
∂u/∂t = α ∇²u  (heat equation)
(I - α*dt*∇²) u_{n+1} = u_n  (implicit Euler)
```

**Run:**
```bash
python 01_heat_equation.py
```

**Output:**
- `output_01_heat_1d.png` - 1D temperature evolution
- `output_01_heat_2d.png` - 2D heat diffusion snapshots
- `output_01_convergence.png` - CG convergence analysis

**Use Cases:**
- Heat transfer simulations
- Diffusion processes
- Implicit time-stepping methods

---

### 2. Poisson Equation Solver (`02_poisson_equation.py`)

Demonstrates solving the Poisson equation for various physics applications.

**Topics Covered:**
- Electrostatic potential (point charges)
- Pressure projection (incompressible flow)
- Periodic boundary conditions
- Gradient and divergence operators

**Equations:**
```
∇²φ = f  (Poisson equation)

Applications:
- Electrostatics: ∇²φ = -ρ/ε₀
- Pressure: ∇²p = ∇·v
- Gravity: ∇²φ = 4πGρ
```

**Run:**
```bash
python 02_poisson_equation.py
```

**Output:**
- `output_02_electrostatics.png` - Electric field from point charges
- `output_02_pressure_projection.png` - Velocity field projection
- `output_02_periodic.png` - Solution with periodic BC

**Use Cases:**
- Electrostatic field calculations
- Incompressible fluid flow (pressure projection)
- Gravitational potential
- Steady-state heat conduction

---

### 3. Solver Comparison (`03_solver_comparison.py`)

Compares different iterative solvers and analyzes performance characteristics.

**Topics Covered:**
- CG vs BiCGSTAB vs GMRES comparison
- Symmetric vs nonsymmetric systems
- Automatic solver selection
- Large-scale performance benchmarks (up to 512×512 grids)

**Solvers:**
- **CG**: Optimal for symmetric positive-definite (SPD) matrices
- **BiCGSTAB**: Robust for general nonsymmetric matrices
- **GMRES**: Most general, memory-intensive with restarts

**Run:**
```bash
python 03_solver_comparison.py
```

**Output:**
- `output_03_symmetric_comparison.png` - CG dominance on SPD systems
- `output_03_nonsymmetric_comparison.png` - BiCGSTAB/GMRES on nonsymmetric
- `output_03_large_scale_performance.png` - Scaling to 250K+ unknowns

**Use Cases:**
- Solver selection guidelines
- Performance benchmarking
- Understanding solver behavior

---

## API Quick Reference

### Sparse Matrix Creation

```python
from morphogen.stdlib.sparse_linalg import csr_matrix, csc_matrix, coo_matrix

# From dense array
A_dense = np.array([[1, 0, 2], [0, 3, 0], [4, 0, 5]])
A_csr = csr_matrix(A_dense)

# From coordinates
data = np.array([1, 2, 3, 4, 5])
rows = np.array([0, 0, 1, 2, 2])
cols = np.array([0, 2, 1, 0, 2])
A_coo = coo_matrix((data, (rows, cols)), shape=(3, 3))
```

### Iterative Solvers

```python
from morphogen.stdlib.sparse_linalg import solve_cg, solve_bicgstab, solve_gmres

# Conjugate Gradient (SPD matrices only)
x, iterations, residual = solve_cg(A, b, tol=1e-8)

# BiCGSTAB (general nonsymmetric)
x, iterations, residual = solve_bicgstab(A, b, tol=1e-8)

# GMRES (most general)
x, iterations, residual = solve_gmres(A, b, tol=1e-8, restart=20)

# Automatic selection
from morphogen.stdlib.sparse_linalg import solve_sparse
x = solve_sparse(A, b, method="auto")  # Chooses best method
```

### Discrete Operators

```python
from morphogen.stdlib.sparse_linalg import laplacian_2d, gradient_2d, divergence_2d

# 2D Laplacian operator
lap = laplacian_2d(64, 64, bc="dirichlet")  # ∇² operator

# Gradient operators
Gx, Gy = gradient_2d(64, 64)  # ∂/∂x and ∂/∂y

# Divergence operator
div = divergence_2d(64, 64)  # ∇·
```

### Boundary Conditions

```python
# Dirichlet BC (fixed values at boundary)
lap = laplacian_1d(100, bc="dirichlet")  # u = 0 at boundaries

# Neumann BC (zero derivative at boundary)
lap = laplacian_1d(100, bc="neumann")  # du/dx = 0 at boundaries

# Periodic BC
lap = laplacian_1d(100, bc="periodic")  # u(0) = u(L)
```

---

## Solver Selection Guidelines

| Matrix Type | Recommended Solver | Notes |
|-------------|-------------------|-------|
| Symmetric Positive-Definite | **CG** | Optimal convergence, least memory |
| Nonsymmetric | **BiCGSTAB** | Good general choice, stable |
| Ill-conditioned | **GMRES** | Most robust, uses more memory |
| Small (<1000 unknowns) | Direct solver | Faster than iterative |
| Unknown properties | `solve_sparse(method="auto")` | Automatic selection |

### Performance Characteristics

**Conjugate Gradient (CG):**
- ✅ Best for SPD matrices (Laplacian, FEM stiffness)
- ✅ Minimal memory overhead
- ✅ Guaranteed convergence for SPD systems
- ❌ Fails or slow for nonsymmetric matrices

**BiCGSTAB:**
- ✅ Handles general nonsymmetric matrices
- ✅ Faster than GMRES typically
- ✅ Lower memory than GMRES
- ❌ May fail for some matrices (rare)

**GMRES:**
- ✅ Most robust (works for almost all matrices)
- ✅ Configurable restart for memory control
- ❌ Higher memory usage
- ❌ Restart can slow convergence

---

## Common Applications

### 1. Heat/Diffusion Equations
```python
# ∂u/∂t = α ∇²u
lap = laplacian_2d(64, 64)
A = I - alpha * dt * lap / dx**2  # Implicit Euler
u_new, _, _ = solve_cg(A, u_old)
```

### 2. Poisson Equation (Electrostatics)
```python
# ∇²φ = -ρ/ε₀
lap = laplacian_2d(128, 128)
phi, _, _ = solve_cg(-lap / dx**2, -rho)
```

### 3. Pressure Projection (Fluids)
```python
# ∇²p = ∇·v
lap = laplacian_2d(64, 64)
div_v = # ... compute divergence
p, _, _ = solve_cg(-lap / dx**2, -div_v)
```

### 4. Circuit Simulation
```python
# Nodal analysis: G*V = I
# G = conductance matrix (sparse)
V, _, _ = solve_sparse(G, I, method="bicgstab")
```

---

## Performance Tips

### 1. Choose the Right Format
- **CSR** - Best for row operations, matrix-vector products
- **CSC** - Best for column operations, factorizations
- **COO** - Best for incremental construction

### 2. Preconditioners
```python
from morphogen.stdlib.sparse_linalg import incomplete_cholesky, incomplete_lu

# For SPD matrices
M = incomplete_cholesky(A)
# Use with scipy: x, info = sp_linalg.cg(A, b, M=M)

# For general matrices
M = incomplete_lu(A)
# Use with scipy: x, info = sp_linalg.bicgstab(A, b, M=M)
```

### 3. Tolerance Settings
```python
# High accuracy (slower)
x, _, _ = solve_cg(A, b, tol=1e-10)

# Engineering accuracy (faster)
x, _, _ = solve_cg(A, b, tol=1e-6)

# Quick solve (visualization)
x, _, _ = solve_cg(A, b, tol=1e-4)
```

### 4. Initial Guess
```python
# Use previous solution as initial guess (faster convergence)
x_prev = # ... previous timestep solution
x_new, _, _ = solve_cg(A, b, x0=x_prev)
```

---

## Troubleshooting

### Problem: CG not converging
**Cause:** Matrix is not symmetric positive-definite

**Solution:**
1. Check matrix symmetry: `np.allclose(A.todense(), A.todense().T)`
2. Switch to BiCGSTAB or GMRES
3. Use `solve_sparse(method="auto")`

### Problem: Solver is slow
**Causes:**
- Matrix is ill-conditioned
- Wrong solver for matrix type
- Tolerance too tight

**Solutions:**
1. Use preconditioner (incomplete Cholesky/LU)
2. Relax tolerance if appropriate
3. Try different solver
4. Check matrix conditioning: `np.linalg.cond(A.todense())`

### Problem: Memory issues with GMRES
**Cause:** Large restart parameter

**Solution:**
```python
# Reduce restart parameter (trades memory for iterations)
x, _, _ = solve_gmres(A, b, restart=10)  # Instead of 20+
```

---

## Dependencies

Required packages:
```bash
pip install numpy scipy matplotlib
```

The sparse linear algebra module uses:
- **NumPy** - Array operations
- **SciPy** - Sparse matrix formats and solvers
- **Matplotlib** - Visualization (examples only)

---

## Further Reading

### Theory
- [Conjugate Gradient Method](https://en.wikipedia.org/wiki/Conjugate_gradient_method)
- [Krylov Subspace Methods](https://en.wikipedia.org/wiki/Krylov_subspace)
- [Sparse Matrix Formats](https://docs.scipy.org/doc/scipy/reference/sparse.html)

### Applications
- **PDEs**: "Numerical Methods for Partial Differential Equations" by Ames
- **Circuit Simulation**: Modified Nodal Analysis (MNA)
- **Graph Algorithms**: Laplacian matrix for spectral clustering
- **Optimization**: Interior point methods, SQP

### Performance
- **Preconditioning**: "Iterative Methods for Sparse Linear Systems" by Saad
- **Multigrid**: Geometric and algebraic multigrid methods
- **GPU Acceleration**: cuSPARSE, MAGMA libraries

---

## Impact

The Sparse Linear Algebra domain unlocks:
- ✅ **Large-scale PDEs** - 1M+ unknowns (Poisson, heat, wave equations)
- ✅ **Circuit simulation** - 1000+ node circuits
- ✅ **Graph algorithms** - Spectral methods, PageRank
- ✅ **Optimization** - Constrained optimization, SQP
- ✅ **Machine learning** - Graph neural networks, kernel methods

**Performance:**
- Solve 250K unknowns in <1 second (2D Poisson equation)
- Memory usage: O(nnz) instead of O(n²)
- Scales to multi-million unknown problems

---

## License

These examples are part of the Kairo project and are licensed under the MIT License.
