"""Example 2: Poisson Equation Solver using Sparse Linear Algebra

This example demonstrates solving the Poisson equation using:
- 2D Laplacian operator (laplacian_2d)
- Conjugate Gradient (CG) solver for SPD systems
- Different boundary conditions (Dirichlet, Neumann, periodic)

The Poisson equation: ∇²φ = f

Applications:
  - Electrostatics: ∇²φ = -ρ/ε₀ (φ = electric potential, ρ = charge density)
  - Pressure projection: ∇²p = ∇·v (incompressible fluid flow)
  - Gravitational potential: ∇²φ = 4πGρ (φ = gravitational potential)
  - Heat steady-state: ∇²T = -Q/k (T = temperature, Q = heat source)
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.insert(0, '/home/user/morphogen')

from morphogen.stdlib.sparse_linalg import (
    laplacian_2d,
    solve_cg,
    gradient_2d
)


def example_electrostatics():
    """Solve Poisson equation for electrostatic potential"""
    print("=" * 70)
    print("Example 1: Electrostatic Potential (Point Charges)")
    print("=" * 70)

    # Parameters
    nx, ny = 128, 128
    L = 2.0  # Domain size [-L/2, L/2]

    # Create grid
    x = np.linspace(-L/2, L/2, nx)
    y = np.linspace(-L/2, L/2, ny)
    xx, yy = np.meshgrid(x, y)
    dx = x[1] - x[0]

    # Create charge distribution (two point charges)
    # Charge is represented as a source term
    rho = np.zeros((ny, nx))

    # Positive charge at (-0.5, 0)
    idx_x1, idx_y1 = int(nx * 0.25), int(ny * 0.5)
    rho[idx_y1-2:idx_y1+3, idx_x1-2:idx_x1+3] = 100.0

    # Negative charge at (0.5, 0)
    idx_x2, idx_y2 = int(nx * 0.75), int(ny * 0.5)
    rho[idx_y2-2:idx_y2+3, idx_x2-2:idx_x2+3] = -100.0

    # Flatten for solver
    rho_flat = rho.flatten()

    # Create Laplacian operator (Dirichlet BC: φ=0 at boundaries)
    lap = laplacian_2d(nx, ny, bc="dirichlet")

    # Solve: ∇²φ = -ρ/ε₀
    # For simplicity, we set ε₀ = 1
    # Note: -lap because our Laplacian is defined with opposite sign convention
    A = -lap.tocsr() / dx**2
    b = -rho_flat

    print(f"\nSolving Poisson equation for electrostatic potential:")
    print(f"  Grid size: {nx} × {ny} = {nx*ny} unknowns")
    print(f"  Matrix nonzeros: {A.nnz}")
    print(f"  Boundary condition: Dirichlet (φ=0 at walls)")

    # Solve
    phi_flat, iters, resid = solve_cg(A, b, tol=1e-10)
    phi = phi_flat.reshape(ny, nx)

    print(f"  CG iterations: {iters}")
    print(f"  Final residual: {resid:.2e}")
    print(f"  Potential range: [{np.min(phi):.4f}, {np.max(phi):.4f}]")

    # Compute electric field: E = -∇φ
    Gx, Gy = gradient_2d(nx, ny)
    Ex_flat = -Gx @ phi_flat / dx
    Ey_flat = -Gy @ phi_flat / dx
    Ex = Ex_flat.reshape(ny, nx)
    Ey = Ey_flat.reshape(ny, nx)

    # Plot results
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Charge distribution
    ax = axes[0]
    im = ax.imshow(rho, extent=[-L/2, L/2, -L/2, L/2], origin='lower', cmap='RdBu', vmin=-100, vmax=100)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title('Charge Distribution ρ')
    plt.colorbar(im, ax=ax, label='Charge density')

    # Electric potential
    ax = axes[1]
    im = ax.contourf(xx, yy, phi, levels=20, cmap='RdBu')
    ax.contour(xx, yy, phi, levels=10, colors='black', alpha=0.3, linewidths=0.5)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title('Electric Potential φ')
    plt.colorbar(im, ax=ax, label='Potential')

    # Electric field (streamlines)
    ax = axes[2]
    # Compute field magnitude
    E_mag = np.sqrt(Ex**2 + Ey**2)
    im = ax.imshow(E_mag, extent=[-L/2, L/2, -L/2, L/2], origin='lower', cmap='plasma')
    # Add streamlines
    skip = 4
    ax.quiver(xx[::skip, ::skip], yy[::skip, ::skip],
             Ex[::skip, ::skip], Ey[::skip, ::skip],
             color='white', alpha=0.7, scale=50)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title('Electric Field E = -∇φ')
    plt.colorbar(im, ax=ax, label='|E|')

    plt.tight_layout()
    plt.savefig('/home/user/morphogen/examples/sparse_linalg/output_02_electrostatics.png', dpi=150)
    print(f"\n✓ Saved plot to output_02_electrostatics.png")
    plt.close()


def example_pressure_projection():
    """Solve Poisson equation for pressure (divergence-free velocity field)"""
    print("\n" + "=" * 70)
    print("Example 2: Pressure Projection (Incompressible Flow)")
    print("=" * 70)

    # Parameters
    nx, ny = 64, 64
    L = 1.0

    # Create grid
    x = np.linspace(0, L, nx)
    y = np.linspace(0, L, ny)
    xx, yy = np.meshgrid(x, y)
    dx = x[1] - x[0]

    # Create velocity field with divergence (not divergence-free)
    # This might come from advection step in fluid simulation
    vx = np.sin(2 * np.pi * xx) * np.cos(2 * np.pi * yy)
    vy = -np.cos(2 * np.pi * xx) * np.sin(2 * np.pi * yy)

    # Add some divergence
    vx += 0.5 * xx
    vy += 0.3 * yy

    # Compute divergence: ∇·v
    Gx, Gy = gradient_2d(nx, ny)
    vx_flat = vx.flatten()
    vy_flat = vy.flatten()

    dvx_dx = (Gx @ vx_flat / dx).reshape(ny, nx)
    dvy_dy = (Gy @ vy_flat / dx).reshape(ny, nx)
    div_v = dvx_dx + dvy_dy

    print(f"\nProjecting velocity field to divergence-free:")
    print(f"  Initial divergence (max): {np.max(np.abs(div_v)):.6f}")

    # Solve Poisson equation for pressure: ∇²p = ∇·v
    lap = laplacian_2d(nx, ny, bc="neumann")
    A = lap.tocsr() / dx**2
    b = -div_v.flatten()

    # Solve
    p_flat, iters, resid = solve_cg(-A, b, tol=1e-8)
    p = p_flat.reshape(ny, nx)

    # Make pressure zero-mean (arbitrary constant doesn't matter for incompressible flow)
    p -= np.mean(p)

    print(f"  CG iterations: {iters}")
    print(f"  Final residual: {resid:.2e}")

    # Correct velocity: v_new = v_old - ∇p
    dp_dx = (Gx @ p_flat / dx).reshape(ny, nx)
    dp_dy = (Gy @ p_flat / dx).reshape(ny, nx)

    vx_proj = vx - dp_dx
    vy_proj = vy - dp_dy

    # Compute new divergence
    dvx_dx_new = (Gx @ vx_proj.flatten() / dx).reshape(ny, nx)
    dvy_dy_new = (Gy @ vy_proj.flatten() / dx).reshape(ny, nx)
    div_v_new = dvx_dx_new + dvy_dy_new

    print(f"  Final divergence (max): {np.max(np.abs(div_v_new)):.6f}")
    print(f"  Divergence reduced by: {np.max(np.abs(div_v)) / max(np.max(np.abs(div_v_new)), 1e-10):.1f}×")

    # Plot results
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    # Original velocity field
    ax = axes[0, 0]
    skip = 2
    ax.quiver(xx[::skip, ::skip], yy[::skip, ::skip],
             vx[::skip, ::skip], vy[::skip, ::skip])
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title('Original Velocity Field')
    ax.set_aspect('equal')

    # Original divergence
    ax = axes[0, 1]
    im = ax.imshow(div_v, extent=[0, L, 0, L], origin='lower', cmap='RdBu', vmin=-2, vmax=2)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title('Original Divergence ∇·v')
    plt.colorbar(im, ax=ax)

    # Pressure field
    ax = axes[0, 2]
    im = ax.contourf(xx, yy, p, levels=20, cmap='viridis')
    ax.contour(xx, yy, p, levels=10, colors='black', alpha=0.3, linewidths=0.5)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title('Pressure p')
    plt.colorbar(im, ax=ax)

    # Projected velocity field
    ax = axes[1, 0]
    ax.quiver(xx[::skip, ::skip], yy[::skip, ::skip],
             vx_proj[::skip, ::skip], vy_proj[::skip, ::skip])
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title('Projected Velocity Field (Divergence-Free)')
    ax.set_aspect('equal')

    # New divergence
    ax = axes[1, 1]
    im = ax.imshow(div_v_new, extent=[0, L, 0, L], origin='lower', cmap='RdBu', vmin=-0.01, vmax=0.01)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title('New Divergence ∇·v (nearly zero)')
    plt.colorbar(im, ax=ax)

    # Velocity correction
    ax = axes[1, 2]
    correction_mag = np.sqrt(dp_dx**2 + dp_dy**2)
    im = ax.imshow(correction_mag, extent=[0, L, 0, L], origin='lower', cmap='hot')
    ax.quiver(xx[::skip, ::skip], yy[::skip, ::skip],
             -dp_dx[::skip, ::skip], -dp_dy[::skip, ::skip],
             color='cyan', alpha=0.5)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title('Velocity Correction -∇p')
    plt.colorbar(im, ax=ax)

    plt.tight_layout()
    plt.savefig('/home/user/morphogen/examples/sparse_linalg/output_02_pressure_projection.png', dpi=150)
    print(f"\n✓ Saved plot to output_02_pressure_projection.png")
    plt.close()


def example_periodic_boundary():
    """Solve Poisson equation with periodic boundary conditions"""
    print("\n" + "=" * 70)
    print("Example 3: Poisson Equation with Periodic BC")
    print("=" * 70)

    # Parameters
    nx, ny = 64, 64
    L = 1.0

    # Create grid
    x = np.linspace(0, L, nx)
    y = np.linspace(0, L, ny)
    xx, yy = np.meshgrid(x, y)
    dx = x[1] - x[0]

    # Create periodic source term (sum must be zero for solvability)
    f = np.sin(4 * np.pi * xx) * np.cos(4 * np.pi * yy)
    f -= np.mean(f)  # Ensure zero mean

    # Create Laplacian with periodic BC
    lap = laplacian_2d(nx, ny, bc="periodic")
    A = lap.tocsr() / dx**2

    print(f"\nSolving Poisson equation with periodic BC:")
    print(f"  Grid size: {nx} × {ny}")
    print(f"  Source term mean: {np.mean(f):.2e} (must be zero)")

    # Solve: ∇²u = f
    # For periodic BC, solution is unique up to a constant
    u_flat, iters, resid = solve_cg(-A, -f.flatten(), tol=1e-8)
    u = u_flat.reshape(ny, nx)
    u -= np.mean(u)  # Set mean to zero

    print(f"  CG iterations: {iters}")
    print(f"  Final residual: {resid:.2e}")

    # Verify solution by computing Laplacian of u
    lap_u_flat = (lap @ u_flat) / dx**2
    lap_u = lap_u_flat.reshape(ny, nx)

    error = np.max(np.abs(lap_u - f))
    print(f"  Max error |∇²u - f|: {error:.2e}")

    # Plot results
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Source term
    ax = axes[0]
    im = ax.imshow(f, extent=[0, L, 0, L], origin='lower', cmap='RdBu')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title('Source Term f')
    plt.colorbar(im, ax=ax)

    # Solution
    ax = axes[1]
    im = ax.imshow(u, extent=[0, L, 0, L], origin='lower', cmap='viridis')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title('Solution u (∇²u = f)')
    plt.colorbar(im, ax=ax)

    # Error
    ax = axes[2]
    im = ax.imshow(lap_u - f, extent=[0, L, 0, L], origin='lower', cmap='RdBu')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title('Error (∇²u - f)')
    plt.colorbar(im, ax=ax)

    plt.tight_layout()
    plt.savefig('/home/user/morphogen/examples/sparse_linalg/output_02_periodic.png', dpi=150)
    print(f"\n✓ Saved plot to output_02_periodic.png")
    plt.close()


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("SPARSE LINEAR ALGEBRA - POISSON EQUATION EXAMPLES")
    print("=" * 70)

    example_electrostatics()
    example_pressure_projection()
    example_periodic_boundary()

    print("\n" + "=" * 70)
    print("ALL EXAMPLES COMPLETED SUCCESSFULLY ✓")
    print("=" * 70)
