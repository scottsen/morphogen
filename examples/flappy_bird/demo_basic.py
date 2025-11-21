#!/usr/bin/env python3
"""
Flappy Bird Basic Demo
======================

A simple demonstration of the Flappy Bird game physics domain.
Shows a single bird controlled by a random policy.

This is a minimal example showing just the game domain in action.
"""

import numpy as np
from morphogen.stdlib.flappy import flappy, random_controller


def main():
    """Run a simple Flappy Bird simulation with random control"""
    print("Flappy Bird Basic Demo")
    print("=" * 50)
    print()

    # Initialize game with 5 birds
    n_birds = 5
    state = flappy.alloc_game(n_birds=n_birds, n_pipes=3, seed=42)

    print(f"Simulating {n_birds} birds with random control...")
    print()

    # Run simulation
    max_steps = 500
    for step in range(max_steps):
        # Random actions (5% flap probability per bird per frame)
        actions = random_controller(state, flap_prob=0.05)

        # Step simulation
        state = flappy.step(state, actions)

        # Print status every 50 steps
        if step % 50 == 0:
            alive_count = np.sum(state.bird_alive)
            best_score = np.max(state.bird_score)
            print(f"Step {step:3d}: {alive_count} alive, best score: {best_score:.0f}")

        # Stop if all birds dead
        if not np.any(state.bird_alive):
            print(f"\nAll birds died at step {step}")
            break

    # Final statistics
    print()
    print("Final Statistics:")
    print("-" * 50)
    for i in range(n_birds):
        print(f"  Bird {i}: Score={state.bird_score[i]:.0f}, "
              f"Pipes={state.bird_pipes_passed[i]}, "
              f"Alive={state.bird_alive[i]}")

    print()
    print("Demo complete!")


if __name__ == '__main__':
    main()
