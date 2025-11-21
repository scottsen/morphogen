"""Example 1: Heat Equation Solver using Sparse Linear Algebra

This example demonstrates solving the 1D and 2D heat equation using:
- Sparse Laplacian operators (laplacian_1d, laplacian_2d)
- Conjugate Gradient (CG) iterative solver
- Implicit time-stepping for numerical stability

The heat equation: ∂u/∂t = α ∇²u

Where:
  u = temperature field
  α = thermal diffusivity
  ∇² = Laplacian operator

Implicit time-stepping:
  (I - α*dt*∇²) u_{n+1} = u_n

This requires solving a sparse linear system at each timestep.
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.insert(0, '/home/user/morphogen')

from morphogen.stdlib.sparse_linalg import (
    laplacian_1d,
    laplacian_2d,
    solve_cg,
    csr_matrix
)
from scipy import sparse


def example_1d_heat_equation():
    """Solve 1D heat equation with implicit time-stepping"""
    print("=" * 70)
    print("Example 1: 1D Heat Equation")
    print("=" * 70)

    # Parameters
    n = 200  # Number of spatial points
    L = 1.0  # Domain length
    alpha = 0.01  # Thermal diffusivity
    dt = 0.001  # Time step
    n_steps = 500  # Number of time steps

    # Spatial grid
    dx = L / (n - 1)
    x = np.linspace(0, L, n)

    # Initial condition: Gaussian temperature distribution
    u = np.exp(-100 * (x - 0.5)**2)

    # Create Laplacian operator
    lap = laplacian_1d(n, bc="dirichlet")

    # System matrix for implicit solve: (I - α*dt*∇²)
    # Note: Laplacian discretization includes 1/dx² scaling
    I = sparse.eye(n, format='csr')
    A = I - alpha * dt * lap / dx**2

    # Storage for solution at different times
    u_history = [u.copy()]
    times = [0]

    print(f"\nSolving 1D heat equation:")
    print(f"  Grid points: {n}")
    print(f"  Time step: {dt}")
    print(f"  Thermal diffusivity: {alpha}")
    print(f"  Total steps: {n_steps}")

    # Time integration with implicit method
    for step in range(n_steps):
        # Solve (I - α*dt*∇²) u_{n+1} = u_n
        u_new, iters, resid = solve_cg(A, u, x0=u, tol=1e-8)
        u = u_new

        # Store solution periodically
        if step % 100 == 0:
            u_history.append(u.copy())
            times.append((step + 1) * dt)
            print(f"  Step {step:4d}: CG iterations={iters:3d}, residual={resid:.2e}")

    # Plot results
    plt.figure(figsize=(12, 5))

    # Plot temperature evolution
    plt.subplot(1, 2, 1)
    for i, (u_snap, t) in enumerate(zip(u_history, times)):
        plt.plot(x, u_snap, label=f't={t:.3f}', linewidth=2)
    plt.xlabel('Position')
    plt.ylabel('Temperature')
    plt.title('1D Heat Equation - Temperature Evolution')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Plot final temperature profile
    plt.subplot(1, 2, 2)
    plt.plot(x, u_history[0], 'r--', label='Initial', linewidth=2)
    plt.plot(x, u_history[-1], 'b-', label=f'Final (t={times[-1]:.3f})', linewidth=2)
    plt.xlabel('Position')
    plt.ylabel('Temperature')
    plt.title('1D Heat Equation - Initial vs Final')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('/home/user/morphogen/examples/sparse_linalg/output_01_heat_1d.png', dpi=150)
    print(f"\n✓ Saved plot to output_01_heat_1d.png")
    plt.close()


def example_2d_heat_equation():
    """Solve 2D heat equation with implicit time-stepping"""
    print("\n" + "=" * 70)
    print("Example 2: 2D Heat Equation")
    print("=" * 70)

    # Parameters
    nx, ny = 64, 64  # Grid size
    L = 1.0  # Domain size
    alpha = 0.01  # Thermal diffusivity
    dt = 0.0005  # Time step
    n_steps = 100  # Number of time steps

    # Spatial grid
    dx = L / (nx - 1)
    dy = L / (ny - 1)
    x = np.linspace(0, L, nx)
    y = np.linspace(0, L, ny)
    xx, yy = np.meshgrid(x, y)

    # Initial condition: Two Gaussian heat sources
    u_2d = (np.exp(-200 * ((xx - 0.3)**2 + (yy - 0.3)**2)) +
            np.exp(-200 * ((xx - 0.7)**2 + (yy - 0.7)**2)))
    u = u_2d.flatten()

    # Create 2D Laplacian operator
    lap = laplacian_2d(nx, ny, bc="dirichlet")

    # System matrix for implicit solve
    # Assuming dx = dy for simplicity
    n = nx * ny
    I = sparse.eye(n, format='csr')
    A = I - alpha * dt * lap / dx**2

    print(f"\nSolving 2D heat equation:")
    print(f"  Grid size: {nx} × {ny} = {n} unknowns")
    print(f"  Matrix nonzeros: {A.nnz}")
    print(f"  Sparsity: {100 * A.nnz / n**2:.2f}%")
    print(f"  Time step: {dt}")
    print(f"  Total steps: {n_steps}")

    # Storage for snapshots
    snapshots = [u_2d.copy()]
    snapshot_times = [0]

    # Time integration
    for step in range(n_steps):
        # Solve sparse system
        u_new, iters, resid = solve_cg(A, u, x0=u, tol=1e-8)
        u = u_new

        # Store snapshots
        if step % 25 == 0 or step == n_steps - 1:
            u_2d = u.reshape(ny, nx)
            snapshots.append(u_2d.copy())
            snapshot_times.append((step + 1) * dt)
            print(f"  Step {step:4d}: CG iterations={iters:3d}, residual={resid:.2e}, "
                  f"T_max={np.max(u_2d):.4f}")

    # Plot results
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    for i, (snap, t) in enumerate(zip(snapshots, snapshot_times)):
        if i >= 6:
            break
        ax = axes[i]
        im = ax.imshow(snap, extent=[0, L, 0, L], origin='lower', cmap='hot',
                      vmin=0, vmax=snapshots[0].max())
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_title(f't = {t:.4f}')
        plt.colorbar(im, ax=ax)

    plt.suptitle('2D Heat Equation - Temperature Evolution', fontsize=14)
    plt.tight_layout()
    plt.savefig('/home/user/morphogen/examples/sparse_linalg/output_01_heat_2d.png', dpi=150)
    print(f"\n✓ Saved plot to output_01_heat_2d.png")
    plt.close()


def example_convergence_analysis():
    """Analyze CG convergence for different grid sizes"""
    print("\n" + "=" * 70)
    print("Example 3: CG Convergence Analysis")
    print("=" * 70)

    grid_sizes = [32, 64, 128, 256]
    iterations = []
    residuals = []
    solve_times = []

    print("\nAnalyzing CG convergence for different grid sizes:")

    import time

    for n in grid_sizes:
        # Create 2D Laplacian
        lap = laplacian_2d(n, n, bc="dirichlet")

        # Make SPD by negating
        A = -lap.tocsr()

        # Create random right-hand side
        np.random.seed(42)
        b = np.random.rand(n * n)

        # Solve
        t0 = time.time()
        x, iters, resid = solve_cg(A, b, tol=1e-8)
        t1 = time.time()

        iterations.append(iters)
        residuals.append(resid)
        solve_times.append(t1 - t0)

        print(f"  {n:3d} × {n:3d} = {n*n:6d} unknowns: "
              f"{iters:4d} iterations, {resid:.2e} residual, {t1-t0:.3f}s")

    # Plot convergence
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 3, 1)
    plt.plot(grid_sizes, iterations, 'o-', linewidth=2, markersize=8)
    plt.xlabel('Grid size (n)')
    plt.ylabel('CG iterations')
    plt.title('Iterations vs Grid Size')
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 3, 2)
    plt.semilogy(grid_sizes, residuals, 's-', linewidth=2, markersize=8, color='orange')
    plt.xlabel('Grid size (n)')
    plt.ylabel('Final residual')
    plt.title('Residual vs Grid Size')
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 3, 3)
    plt.plot(grid_sizes, solve_times, '^-', linewidth=2, markersize=8, color='green')
    plt.xlabel('Grid size (n)')
    plt.ylabel('Solve time (s)')
    plt.title('Solve Time vs Grid Size')
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('/home/user/morphogen/examples/sparse_linalg/output_01_convergence.png', dpi=150)
    print(f"\n✓ Saved plot to output_01_convergence.png")
    plt.close()


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("SPARSE LINEAR ALGEBRA - HEAT EQUATION EXAMPLES")
    print("=" * 70)

    example_1d_heat_equation()
    example_2d_heat_equation()
    example_convergence_analysis()

    print("\n" + "=" * 70)
    print("ALL EXAMPLES COMPLETED SUCCESSFULLY ✓")
    print("=" * 70)
