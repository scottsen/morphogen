"""Example: Saving and resuming simulation checkpoints.

Demonstrates checkpoint/resume functionality for long-running simulations.
Shows how to save full simulation state including fields, particles, and metadata.
"""

import numpy as np
import sys
sys.path.insert(0, '/home/user/morphogen')

from morphogen.stdlib.io_storage import save_checkpoint, load_checkpoint
from morphogen.stdlib.integrators import rk4


def fluid_derivative(t, state):
    """Simple fluid simulation (mock)"""
    # Just for demonstration - real fluid would use proper physics
    velocity_field = state[:128*128*2].reshape(128, 128, 2)
    pressure_field = state[128*128*2:].reshape(128, 128)

    # Mock dynamics
    dvel = -0.01 * velocity_field + np.random.randn(*velocity_field.shape) * 0.001
    dpres = -0.01 * pressure_field + np.random.randn(*pressure_field.shape) * 0.001

    return np.concatenate([dvel.flatten(), dpres.flatten()])


def run_simulation(n_steps=100, checkpoint_interval=25, resume_from=None):
    """Run simulation with periodic checkpointing.

    Args:
        n_steps: Number of timesteps to simulate
        checkpoint_interval: Save checkpoint every N steps
        resume_from: Path to checkpoint to resume from (None = start from scratch)
    """
    dt = 0.01

    if resume_from:
        print(f"Resuming from checkpoint: {resume_from}")
        state, metadata = load_checkpoint(resume_from)

        velocity_field = state["velocity_field"]
        pressure_field = state["pressure_field"]
        start_iteration = metadata["iteration"]
        t = metadata["time"]

        print(f"  Resumed at iteration {start_iteration}, time {t:.2f}")
        print()

    else:
        print("Starting simulation from scratch...")
        # Initial conditions
        velocity_field = np.zeros((128, 128, 2), dtype=np.float32)
        pressure_field = np.ones((128, 128), dtype=np.float32)

        # Add some initial perturbation
        velocity_field[60:68, 60:68, 0] = 0.5
        start_iteration = 0
        t = 0.0

        print("  Initial conditions set")
        print()

    # Flatten for integrator
    combined_state = np.concatenate([
        velocity_field.flatten(),
        pressure_field.flatten()
    ])

    print(f"Running simulation for {n_steps} steps (dt={dt})...")
    print(f"Checkpoint interval: {checkpoint_interval} steps")
    print()

    for step in range(start_iteration, start_iteration + n_steps):
        # Integrate
        combined_state = rk4(t, combined_state, fluid_derivative, dt)
        t += dt

        # Extract fields
        velocity_field = combined_state[:128*128*2].reshape(128, 128, 2)
        pressure_field = combined_state[128*128*2:].reshape(128, 128)

        # Save checkpoint periodically
        if (step + 1) % checkpoint_interval == 0:
            checkpoint_state = {
                "velocity_field": velocity_field.copy(),
                "pressure_field": pressure_field.copy(),
                "parameters": {
                    "dt": dt,
                    "grid_size": 128
                }
            }

            checkpoint_metadata = {
                "iteration": step + 1,
                "time": t,
                "version": "0.8.0"
            }

            checkpoint_path = f"checkpoint_{step + 1:05d}.h5"
            save_checkpoint(checkpoint_path, checkpoint_state, checkpoint_metadata)

            print(f"  Step {step + 1:5d}: t={t:6.2f}, "
                  f"v_max={np.max(np.abs(velocity_field)):.4f}, "
                  f"p_mean={np.mean(pressure_field):.4f}, "
                  f"checkpoint saved → {checkpoint_path}")

    print()
    print("=" * 80)
    print("FINAL STATE")
    print("=" * 80)
    print(f"Iterations: {start_iteration + n_steps}")
    print(f"Time: {t:.2f}")
    print(f"Velocity field: max={np.max(np.abs(velocity_field)):.6f}, "
          f"mean={np.mean(np.abs(velocity_field)):.6f}")
    print(f"Pressure field: max={np.max(pressure_field):.6f}, "
          f"mean={np.mean(pressure_field):.6f}")
    print("=" * 80)


def main():
    print("=" * 80)
    print("SIMULATION CHECKPOINT/RESUME EXAMPLE")
    print("=" * 80)
    print()

    # Run simulation and save checkpoints
    print("PART 1: Run simulation with checkpoints")
    print("-" * 80)
    run_simulation(n_steps=100, checkpoint_interval=25)

    print()
    print()
    print("PART 2: Resume from checkpoint")
    print("-" * 80)

    # Resume from checkpoint 50
    run_simulation(n_steps=50, checkpoint_interval=25, resume_from="checkpoint_00050.h5")

    print()
    print("=" * 80)
    print("KEY OBSERVATIONS:")
    print("  - Checkpoints include full state (fields, parameters)")
    print("  - Metadata tracks iteration, time, version")
    print("  - Simulation can be resumed from any checkpoint")
    print("  - Useful for long-running simulations (hours/days)")
    print("  - HDF5 format is compact and efficient")
    print("=" * 80)


if __name__ == "__main__":
    main()
