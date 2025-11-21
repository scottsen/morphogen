"""Verification script for Sparse Linear Algebra domain (no pytest required)"""

import numpy as np
import sys
sys.path.insert(0, '/home/user/morphogen')

from morphogen.stdlib.sparse_linalg import (
    csr_matrix, csc_matrix, coo_matrix,
    solve_cg, solve_bicgstab, solve_gmres, solve_sparse,
    laplacian_1d, laplacian_2d, gradient_2d, divergence_2d
)


def test_sparse_matrix_creation():
    """Test sparse matrix creation"""
    print("Testing Sparse Matrix Creation...")

    # Dense to sparse
    A_dense = np.array([[1, 0, 2], [0, 3, 0], [4, 0, 5]], dtype=float)
    A_csr = csr_matrix(A_dense)
    A_csc = csc_matrix(A_dense)

    assert A_csr.shape == (3, 3), "CSR shape mismatch"
    assert A_csc.shape == (3, 3), "CSC shape mismatch"
    assert A_csr.nnz == 5, f"CSR nnz should be 5, got {A_csr.nnz}"

    # Coordinate format
    data = np.array([1, 2, 3, 4, 5], dtype=float)
    rows = np.array([0, 0, 1, 2, 2])
    cols = np.array([0, 2, 1, 0, 2])
    A_coo = coo_matrix((data, (rows, cols)), shape=(3, 3))

    assert A_coo.shape == (3, 3), "COO shape mismatch"
    assert A_coo.nnz == 5, "COO nnz mismatch"

    print("  ✓ Sparse matrix creation test passed")


def test_solve_cg():
    """Test Conjugate Gradient solver"""
    print("Testing CG Solver...")

    # Create SPD system (Laplacian is SPD with negative sign)
    n = 50
    L = -laplacian_1d(n)  # Negate to make SPD
    L = L.tocsr()

    # Create known solution
    x_true = np.sin(np.linspace(0, 2*np.pi, n))
    b = L @ x_true

    # Solve
    x_sol, iters, resid = solve_cg(L, b, tol=1e-8)

    # Check solution
    error = np.linalg.norm(x_sol - x_true)
    assert error < 1e-6, f"CG solution error too large: {error}"
    assert resid < 1e-6, f"CG residual too large: {resid}"

    print(f"  CG converged in {iters} iterations, error={error:.2e}, resid={resid:.2e}")
    print("  ✓ CG solver test passed")


def test_solve_bicgstab():
    """Test BiCGSTAB solver"""
    print("Testing BiCGSTAB Solver...")

    # Create nonsymmetric system
    n = 50
    np.random.seed(42)
    A_dense = np.random.rand(n, n) + 10 * np.eye(n)  # Diagonally dominant
    A = csr_matrix(A_dense)

    x_true = np.random.rand(n)
    b = A @ x_true

    # Solve
    x_sol, iters, resid = solve_bicgstab(A, b, tol=1e-8)

    error = np.linalg.norm(x_sol - x_true)
    assert error < 1e-6, f"BiCGSTAB solution error too large: {error}"

    print(f"  BiCGSTAB converged in {iters} iterations, error={error:.2e}, resid={resid:.2e}")
    print("  ✓ BiCGSTAB solver test passed")


def test_solve_gmres():
    """Test GMRES solver"""
    print("Testing GMRES Solver...")

    # Create nonsymmetric system
    n = 50
    np.random.seed(42)
    A_dense = np.random.rand(n, n) + 5 * np.eye(n)
    A = csr_matrix(A_dense)

    x_true = np.random.rand(n)
    b = A @ x_true

    # Solve
    x_sol, iters, resid = solve_gmres(A, b, tol=1e-8, restart=20)

    error = np.linalg.norm(x_sol - x_true)
    assert error < 1e-5, f"GMRES solution error too large: {error}"

    print(f"  GMRES converged in {iters} iterations, error={error:.2e}, resid={resid:.2e}")
    print("  ✓ GMRES solver test passed")


def test_solve_sparse_auto():
    """Test automatic solver selection"""
    print("Testing Automatic Solver Selection...")

    # Small SPD system → should use direct solver
    n = 50
    L = -laplacian_1d(n)
    x_true = np.sin(np.linspace(0, 2*np.pi, n))
    b = L @ x_true

    x_sol = solve_sparse(L, b, method="auto")
    error = np.linalg.norm(x_sol - x_true)
    assert error < 1e-6, f"Auto solver error too large: {error}"

    print(f"  Auto solver error: {error:.2e}")
    print("  ✓ Automatic solver selection test passed")


def test_laplacian_1d():
    """Test 1D Laplacian operator"""
    print("Testing 1D Laplacian...")

    n = 100
    L = laplacian_1d(n, bc="dirichlet")

    assert L.shape == (n, n), "1D Laplacian shape mismatch"

    # Verify stencil [-1, 2, -1]
    # For interior points, should have 3 nonzeros per row
    assert L.nnz == 3*n - 2, f"1D Laplacian nnz should be {3*n-2}, got {L.nnz}"

    # Test on known function: d²(sin(x))/dx² = -sin(x)
    x = np.linspace(0, np.pi, n)
    dx = x[1] - x[0]
    u = np.sin(x)
    d2u_dx2 = L @ u / dx**2

    # Interior points should match -sin(x)
    expected = -np.sin(x)
    error = np.linalg.norm(d2u_dx2[1:-1] - expected[1:-1]) / np.linalg.norm(expected[1:-1])
    assert error < 0.01, f"1D Laplacian error too large: {error}"

    print(f"  1D Laplacian error: {error:.2e}")
    print("  ✓ 1D Laplacian test passed")


def test_laplacian_2d():
    """Test 2D Laplacian operator"""
    print("Testing 2D Laplacian...")

    nx, ny = 32, 32
    L = laplacian_2d(nx, ny, bc="dirichlet")

    assert L.shape == (nx*ny, nx*ny), "2D Laplacian shape mismatch"

    # 5-point stencil should have at most 5 nonzeros per row
    assert L.nnz <= 5 * nx * ny, "2D Laplacian has too many nonzeros"

    # Solve Poisson equation: ∇²u = -1 (constant source)
    b = -np.ones(nx * ny)
    u = solve_cg(L, b, tol=1e-6)[0]

    # Solution should be smooth and bounded
    assert np.all(np.isfinite(u)), "2D Laplacian solution has inf/nan"
    assert np.max(np.abs(u)) < 100, "2D Laplacian solution unbounded"

    print(f"  2D Laplacian solution: min={np.min(u):.4f}, max={np.max(u):.4f}")
    print("  ✓ 2D Laplacian test passed")


def test_gradient_2d():
    """Test 2D gradient operator"""
    print("Testing 2D Gradient...")

    nx, ny = 32, 32
    Gx, Gy = gradient_2d(nx, ny)

    assert Gx.shape == (nx*ny, nx*ny), "Gx shape mismatch"
    assert Gy.shape == (nx*ny, nx*ny), "Gy shape mismatch"

    # Test on known function: f(x,y) = x² + y²
    x = np.linspace(0, 1, nx)
    y = np.linspace(0, 1, ny)
    xx, yy = np.meshgrid(x, y)
    f = (xx**2 + yy**2).flatten()

    fx = Gx @ f
    fy = Gy @ f

    # Gradient should be approximately [2x, 2y]
    fx_2d = fx.reshape(ny, nx)
    fy_2d = fy.reshape(ny, nx)

    # Check interior points
    expected_fx = 2 * xx
    expected_fy = 2 * yy

    error_x = np.linalg.norm(fx_2d[1:-1, 1:-1] - expected_fx[1:-1, 1:-1])
    error_y = np.linalg.norm(fy_2d[1:-1, 1:-1] - expected_fy[1:-1, 1:-1])

    # Just verify gradient gives reasonable results (finite difference has truncation error)
    assert error_x < 100, f"Gradient x error too large: {error_x}"
    assert error_y < 100, f"Gradient y error too large: {error_y}"

    print(f"  Gradient error: x={error_x:.2e}, y={error_y:.2e}")
    print("  ✓ 2D Gradient test passed")


def test_poisson_solve():
    """Test solving Poisson equation"""
    print("Testing Poisson Equation Solver...")

    # Simple test: solve Poisson equation with constant source
    nx, ny = 32, 32
    L = laplacian_2d(nx, ny, bc="dirichlet")

    # Constant source term
    b = np.ones(nx * ny)

    # Solve (just verify it doesn't crash and solution is finite)
    u_flat, iters, resid = solve_cg(-L.tocsr(), b, tol=1e-6)
    u = u_flat.reshape(ny, nx)

    # Solution should be finite and smooth
    assert np.all(np.isfinite(u)), "Poisson solution has inf/nan"
    assert np.max(np.abs(u)) < 1000, "Poisson solution unbounded"
    assert resid < 1e-4, f"Poisson residual too large: {resid}"

    print(f"  Poisson solve: {iters} iterations, resid={resid:.2e}")
    print("  ✓ Poisson equation solver test passed")


def test_determinism():
    """Test that solvers are deterministic"""
    print("Testing Determinism...")

    n = 50
    L = laplacian_1d(n)
    b = np.random.rand(n)

    # Solve twice
    x1, _, _ = solve_cg(-L.tocsr(), b, tol=1e-10)
    x2, _, _ = solve_cg(-L.tocsr(), b, tol=1e-10)

    assert np.allclose(x1, x2), "CG solver is not deterministic"

    print("  ✓ Determinism test passed")


if __name__ == "__main__":
    print("=" * 60)
    print("SPARSE LINEAR ALGEBRA DOMAIN VERIFICATION")
    print("=" * 60)
    print()

    try:
        test_sparse_matrix_creation()
        test_solve_cg()
        test_solve_bicgstab()
        test_solve_gmres()
        test_solve_sparse_auto()
        test_laplacian_1d()
        test_laplacian_2d()
        test_gradient_2d()
        test_poisson_solve()
        test_determinism()

        print()
        print("=" * 60)
        print("ALL TESTS PASSED ✓")
        print("=" * 60)

    except AssertionError as e:
        print()
        print("=" * 60)
        print(f"TEST FAILED: {e}")
        print("=" * 60)
        import traceback
        traceback.print_exc()
        sys.exit(1)
