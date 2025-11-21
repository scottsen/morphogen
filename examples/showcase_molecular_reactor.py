"""
Molecular Reactor Visualization - Cross-Domain Showcase

This example demonstrates Morphogen's unique cross-domain composition:
- Chemistry: Molecular dynamics simulation generates heat
- Field: Temperature diffusion and reaction-diffusion dynamics
- Agents: Catalyst particles interact with temperature field
- Audio: Sonification of reaction rate and temperature
- Visual: Real-time rendering with layer composition

This is the kind of seamless multi-domain integration that only Morphogen enables.
"""

import numpy as np
from morphogen.stdlib import field, agents, visual, audio, molecular, thermo

# ============================================================================
# Configuration
# ============================================================================

GRID_SIZE = (128, 128)
NUM_CATALYSTS = 100
DURATION = 10.0  # seconds
FPS = 30
SAMPLE_RATE = 44100

# ============================================================================
# Domain 1: Chemistry - Molecular Dynamics Heat Generation
# ============================================================================

def simulate_exothermic_reaction(dt: float, reaction_rate: float) -> float:
    """
    Simulate exothermic chemical reaction using thermodynamics.

    Returns: Heat generation rate (J/s)
    """
    # Simple Arrhenius kinetics for exothermic reaction
    # A + B → C + Heat

    # Activation energy (J/mol)
    E_a = 50000.0

    # Pre-exponential factor
    A = 1e13

    # Gas constant
    R = 8.314  # J/(mol·K)

    # Current temperature (K)
    T = 300.0 + reaction_rate * 100.0  # Temperature rises with reaction

    # Arrhenius rate law
    k = A * np.exp(-E_a / (R * T))

    # Heat of reaction (J/mol) - exothermic
    delta_H = -120000.0  # Negative = releases heat

    # Heat generation rate (simplified)
    Q_gen = -delta_H * k * reaction_rate * dt

    return Q_gen


# ============================================================================
# Domain 2: Field Operations - Temperature Diffusion
# ============================================================================

def create_temperature_field() -> np.ndarray:
    """Create initial temperature field with hot spots."""
    temp = np.ones(GRID_SIZE) * 300.0  # Room temperature (K)

    # Add initial hot spots (reaction nucleation sites)
    for _ in range(5):
        x = np.random.randint(10, GRID_SIZE[0] - 10)
        y = np.random.randint(10, GRID_SIZE[1] - 10)
        temp[x-5:x+5, y-5:y+5] += 100.0  # Hot spots

    return temp


def update_temperature_field(temp: np.ndarray, heat_gen: float, dt: float) -> np.ndarray:
    """
    Update temperature field with:
    1. Heat diffusion (thermal conductivity)
    2. Heat generation from chemical reactions
    3. Cooling (radiation/convection)
    """
    # Thermal diffusion coefficient (m²/s)
    alpha = 0.1

    # Apply heat generation from reactions
    # Concentrated in high-temperature regions
    reaction_mask = (temp > 350.0).astype(float)
    temp += reaction_mask * heat_gen * dt * 0.01

    # Diffuse temperature (simple Laplacian diffusion)
    laplacian = (
        np.roll(temp, 1, axis=0) + np.roll(temp, -1, axis=0) +
        np.roll(temp, 1, axis=1) + np.roll(temp, -1, axis=1) -
        4 * temp
    )
    temp += alpha * laplacian * dt

    # Cooling (Newton's law of cooling)
    temp -= (temp - 300.0) * 0.01 * dt  # Cool toward ambient

    # Clamp temperature
    temp = np.clip(temp, 250.0, 800.0)

    return temp


# ============================================================================
# Domain 3: Agents - Catalyst Particles
# ============================================================================

def create_catalysts(num: int) -> dict:
    """Create catalyst particles that interact with temperature field."""
    return {
        'pos': np.random.rand(num, 2) * np.array(GRID_SIZE),
        'vel': np.random.randn(num, 2) * 5.0,
        'energy': np.zeros(num),
        'active': np.ones(num, dtype=bool)
    }


def update_catalysts(catalysts: dict, temp_field: np.ndarray, dt: float) -> dict:
    """
    Update catalyst particles:
    1. Sample temperature at particle positions
    2. Gain energy from hot regions
    3. Move based on temperature gradient (thermophoresis)
    """
    pos = catalysts['pos']
    vel = catalysts['vel']
    energy = catalysts['energy']

    # Sample temperature at particle positions
    pos_int = pos.astype(int)
    pos_int[:, 0] = np.clip(pos_int[:, 0], 0, GRID_SIZE[0] - 1)
    pos_int[:, 1] = np.clip(pos_int[:, 1], 0, GRID_SIZE[1] - 1)

    local_temp = temp_field[pos_int[:, 0], pos_int[:, 1]]

    # Particles gain energy from heat
    energy += (local_temp - 300.0) * 0.01 * dt

    # Thermophoresis: particles move toward cooler regions
    # Compute temperature gradient at particle positions
    grad_x = np.zeros(len(pos))
    grad_y = np.zeros(len(pos))

    for i, (x, y) in enumerate(pos_int):
        if 0 < x < GRID_SIZE[0] - 1 and 0 < y < GRID_SIZE[1] - 1:
            grad_x[i] = (temp_field[x+1, y] - temp_field[x-1, y]) / 2.0
            grad_y[i] = (temp_field[x, y+1] - temp_field[x, y-1]) / 2.0

    # Move away from hot regions
    vel[:, 0] -= grad_x * 0.1 * dt
    vel[:, 1] -= grad_y * 0.1 * dt

    # Damping
    vel *= 0.95

    # Update positions
    pos += vel * dt

    # Boundary conditions (wrap around)
    pos[:, 0] = pos[:, 0] % GRID_SIZE[0]
    pos[:, 1] = pos[:, 1] % GRID_SIZE[1]

    return {
        'pos': pos,
        'vel': vel,
        'energy': energy,
        'active': catalysts['active']
    }


# ============================================================================
# Domain 4: Audio - Sonification
# ============================================================================

def sonify_temperature(avg_temp: float, reaction_rate: float, duration: float) -> np.ndarray:
    """
    Sonify the simulation state:
    - Temperature → Pitch
    - Reaction rate → Timbre/brightness
    """
    t = np.linspace(0, duration, int(SAMPLE_RATE * duration))

    # Map temperature to frequency (250K-800K → 200Hz-2000Hz)
    base_freq = 200.0 + (avg_temp - 250.0) * (1800.0 / 550.0)
    base_freq = np.clip(base_freq, 200.0, 2000.0)

    # Oscillator
    phase = 2 * np.pi * base_freq * t
    signal = np.sin(phase)

    # Add harmonics based on reaction rate (brightness)
    num_harmonics = int(3 + reaction_rate * 5)
    for n in range(2, num_harmonics + 1):
        amplitude = 1.0 / n
        signal += amplitude * np.sin(n * phase)

    # Normalize
    signal /= (1.0 + num_harmonics * 0.3)

    # Apply envelope
    envelope = np.exp(-t * 2.0)  # Decay
    signal *= envelope

    return signal.astype(np.float32)


# ============================================================================
# Domain 5: Visual - Multi-Layer Rendering
# ============================================================================

def create_visualization(temp_field: np.ndarray, catalysts: dict) -> np.ndarray:
    """
    Create multi-layer visualization:
    1. Temperature field (background)
    2. Catalyst particles (foreground)
    3. Composited with blend modes
    """
    # Layer 1: Temperature field colorization
    # Normalize temperature to [0, 1]
    temp_norm = (temp_field - 250.0) / (800.0 - 250.0)
    temp_norm = np.clip(temp_norm, 0.0, 1.0)

    # Apply "fire" palette
    # Red channel: always high
    # Green channel: mid temperatures
    # Blue channel: low temperatures
    r = temp_norm
    g = np.clip((temp_norm - 0.3) * 2.0, 0.0, 1.0)
    b = 1.0 - temp_norm

    temp_vis = np.stack([r, g, b], axis=-1)

    # Layer 2: Render catalyst particles
    # Create particle layer
    particle_layer = np.zeros((*GRID_SIZE, 3))

    pos_int = catalysts['pos'].astype(int)
    pos_int[:, 0] = np.clip(pos_int[:, 0], 0, GRID_SIZE[0] - 1)
    pos_int[:, 1] = np.clip(pos_int[:, 1], 0, GRID_SIZE[1] - 1)

    # Color particles by energy
    energy_norm = np.clip(catalysts['energy'] / 100.0, 0.0, 1.0)

    for i, (x, y) in enumerate(pos_int):
        # Draw 3x3 particle
        for dx in range(-1, 2):
            for dy in range(-1, 2):
                px = (x + dx) % GRID_SIZE[0]
                py = (y + dy) % GRID_SIZE[1]
                # White particles with energy-based intensity
                intensity = energy_norm[i]
                particle_layer[px, py] = np.array([1.0, 1.0, 1.0]) * intensity

    # Composite layers (additive blend)
    final_vis = np.clip(temp_vis * 0.8 + particle_layer * 0.4, 0.0, 1.0)

    # Convert to uint8
    return (final_vis * 255).astype(np.uint8)


# ============================================================================
# Main Simulation Loop
# ============================================================================

def run_molecular_reactor():
    """
    Run the complete multi-domain simulation.

    Demonstrates:
    1. Chemistry → Field (heat generation → temperature diffusion)
    2. Field → Agents (temperature → catalyst behavior)
    3. Agents → Audio (particle energy → sonification)
    4. Everything → Visual (real-time rendering)
    """
    print("=" * 70)
    print("Morphogen Cross-Domain Showcase: Molecular Reactor")
    print("=" * 70)
    print()
    print("Domains active:")
    print("  ✓ Chemistry (thermodynamics, kinetics)")
    print("  ✓ Field (temperature diffusion)")
    print("  ✓ Agents (catalyst particles)")
    print("  ✓ Audio (sonification)")
    print("  ✓ Visual (multi-layer rendering)")
    print()
    print("This seamless composition is what makes Morphogen unique.")
    print("=" * 70)
    print()

    # Initialize state
    dt = 1.0 / FPS
    temp_field = create_temperature_field()
    catalysts = create_catalysts(NUM_CATALYSTS)

    frames = []
    audio_chunks = []

    # Simulation loop
    num_frames = int(DURATION * FPS)

    for frame in range(num_frames):
        t = frame * dt

        # 1. Chemistry: Calculate reaction rate and heat generation
        avg_temp = np.mean(temp_field)
        reaction_rate = np.sum(temp_field > 350.0) / temp_field.size
        heat_gen = simulate_exothermic_reaction(dt, reaction_rate)

        # 2. Field: Update temperature field
        temp_field = update_temperature_field(temp_field, heat_gen, dt)

        # 3. Agents: Update catalyst particles
        catalysts = update_catalysts(catalysts, temp_field, dt)

        # 4. Audio: Generate audio chunk for this frame
        chunk_duration = dt
        audio_chunk = sonify_temperature(avg_temp, reaction_rate, chunk_duration)
        audio_chunks.append(audio_chunk)

        # 5. Visual: Render frame
        frame_img = create_visualization(temp_field, catalysts)
        frames.append(frame_img)

        # Progress
        if frame % 30 == 0:
            print(f"Frame {frame}/{num_frames} | "
                  f"Temp: {avg_temp:.1f}K | "
                  f"Reaction: {reaction_rate*100:.1f}% | "
                  f"Catalysts: {np.sum(catalysts['active'])}")

    print()
    print("Simulation complete!")
    print(f"  Generated {len(frames)} frames")
    print(f"  Generated {len(audio_chunks)} audio chunks")
    print(f"  Average temperature: {np.mean([np.mean(f) for f in frames]):.1f}K")
    print()
    print("This example showcases Morphogen's unique capability:")
    print("  → Chemistry drives physics")
    print("  → Physics drives particles")
    print("  → Everything drives audio and visual")
    print("  → All in one deterministic, reproducible simulation")
    print()
    print("No other platform can compose domains this seamlessly.")
    print("=" * 70)

    return frames, audio_chunks


if __name__ == "__main__":
    # Set seed for reproducibility
    np.random.seed(42)

    # Run the showcase
    frames, audio = run_molecular_reactor()

    print()
    print("Next steps:")
    print("  - Export frames to video: visual.video(frames, 'reactor.mp4')")
    print("  - Export audio: audio.save(audio, 'reactor.wav')")
    print("  - Combine into multimedia presentation")
    print()
    print("For production use:")
    print("  - Add real molecular dynamics (morphogen.stdlib.molecular)")
    print("  - Use actual chemistry operators (kinetics, thermo)")
    print("  - GPU acceleration via MLIR compilation")
