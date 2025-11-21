"""Advanced Visualization Showcase

Demonstrates combining multiple new visualization features:
- Spectrogram + metrics dashboard
- Graph networks + centrality analysis
- Phase space + metrics overlay
- Multi-panel composite visualizations
"""

import numpy as np
from morphogen.stdlib import audio, visual, agents as agent_ops, graph as graph_ops, field


def create_audio_reactive_network():
    """Create a network visualization that could be driven by audio."""
    print("\n1. Audio-reactive network concept...")

    # Generate an audio signal (bass-heavy beat pattern)
    sample_rate = 44100
    duration = 2.0
    t = np.linspace(0, duration, int(duration * sample_rate))

    # Kick drum pattern (4 beats)
    beat_times = [0.0, 0.5, 1.0, 1.5]
    signal = np.zeros_like(t)

    for beat_time in beat_times:
        # Exponentially decaying sine wave (kick drum)
        decay = np.exp(-20 * np.maximum(0, t - beat_time))
        kick = np.sin(2 * np.pi * 60 * (t - beat_time)) * decay
        signal += kick

    audio_buffer = audio.AudioBuffer(signal.astype(np.float32), sample_rate=sample_rate)

    # Create spectrogram
    spec_vis = visual.spectrogram(
        audio_buffer,
        window_size=2048,
        hop_size=512,
        palette="fire",
        log_scale=True,
        freq_range=(20, 500)  # Focus on bass frequencies
    )

    # Add metrics
    metrics = {
        "Audio": "Kick Pattern",
        "BPM": 120,
        "Range": "20-500 Hz",
        "Type": "Percussive"
    }
    spec_vis = visual.add_metrics(spec_vis, metrics, position="top-left", font_size=12)

    visual.output(spec_vis, "output_showcase_audio_network.png")
    print("   Saved: output_showcase_audio_network.png")


def create_multi_network_comparison():
    """Compare different network topologies side by side."""
    print("\n2. Multi-network topology comparison...")

    # Create three different network types
    networks = {
        "Star": create_star(15),
        "Ring": create_ring(15),
        "Random": create_random(15, 0.3)
    }

    for name, g in networks.items():
        vis = visual.graph(
            g,
            width=500,
            height=500,
            layout="force" if name != "Ring" else "circular",
            iterations=100,
            node_size=10.0,
            color_by_centrality=True,
            palette="viridis",
            edge_color=(0.4, 0.4, 0.4),
            background=(0.05, 0.05, 0.1)
        )

        # Calculate network metrics
        n_edges = np.sum(g.adj > 0) // 2
        degrees = np.sum(g.adj > 0, axis=1)
        avg_degree = np.mean(degrees)

        metrics = {
            "Type": name,
            "Nodes": g.n_nodes,
            "Edges": n_edges,
            "Avg Degree": f"{avg_degree:.1f}"
        }
        vis = visual.add_metrics(vis, metrics, position="top-left")

        visual.output(vis, f"output_showcase_network_{name.lower()}.png")
        print(f"   Saved: output_showcase_network_{name.lower()}.png")


def create_dynamical_system_analysis():
    """Analyze a dynamical system with phase space."""
    print("\n3. Dynamical system analysis (Lorenz-like attractor)...")

    # Simplified Lorenz-like dynamics
    n_agents = 500
    n_steps = 200
    dt = 0.01

    # Random initial conditions
    np.random.seed(42)
    positions = np.random.uniform(-1, 1, (n_agents, 2))
    velocities = np.random.uniform(-1, 1, (n_agents, 2))

    particles = agent_ops.create(n_agents, pos=positions)
    particles = agent_ops.set(particles, 'vel', velocities)

    # Simulate chaotic dynamics
    for step in range(n_steps):
        pos = particles.get('pos')
        vel = particles.get('vel')

        # Simplified chaotic dynamics
        x, y = pos[:, 0:1], pos[:, 1:2]
        vx, vy = vel[:, 0:1], vel[:, 1:2]

        # Chaotic forces
        fx = 10.0 * (y - x)
        fy = x * (28.0 - np.abs(x)) - y

        # Update
        new_vel = vel + np.column_stack([fx, fy]) * dt
        new_pos = pos + new_vel * dt

        particles = agent_ops.set(particles, 'vel', new_vel)
        particles = agent_ops.set(particles, 'pos', new_pos)

    # Calculate trajectory divergence
    initial_dist = np.linalg.norm(positions, axis=1)
    final_dist = np.linalg.norm(particles.get('pos'), axis=1)
    divergence = np.abs(final_dist - initial_dist)
    particles = agent_ops.set(particles, 'divergence', divergence)

    # Create phase space visualization
    vis = visual.phase_space(
        particles,
        width=800,
        height=800,
        color_property='divergence',
        palette='coolwarm',
        point_size=2.5,
        alpha=0.6,
        background=(0.0, 0.0, 0.0)
    )

    metrics = {
        "System": "Chaotic Attractor",
        "Particles": n_agents,
        "Steps": n_steps,
        "dt": dt,
        "Type": "Lorenz-like"
    }
    vis = visual.add_metrics(vis, metrics, position="top-right", font_size=14)

    visual.output(vis, "output_showcase_chaotic_system.png")
    print("   Saved: output_showcase_chaotic_system.png")


def create_combined_field_network():
    """Combine field visualization with network overlay."""
    print("\n4. Combined field + network visualization...")

    # Create a field (e.g., temperature distribution)
    from morphogen.stdlib.field import Field2D

    size = 256
    x = np.linspace(-2, 2, size)
    y = np.linspace(-2, 2, size)
    X, Y = np.meshgrid(x, y)

    # Create interesting field (Gaussian mixture)
    field_data = np.zeros((size, size))
    centers = [(-1, -1), (1, 1), (-1, 1), (1, -1)]

    for cx, cy in centers:
        dist = np.sqrt((X - cx)**2 + (Y - cy)**2)
        field_data += np.exp(-dist**2 / 0.5)

    temp_field = Field2D(field_data, dx=1.0, dy=1.0)

    # Visualize field
    field_vis = visual.colorize(temp_field, palette="viridis")

    # Create a small network overlay (conceptually showing measurement stations)
    g = graph_ops.create(len(centers))
    # Connect nearby centers
    for i in range(len(centers)):
        for j in range(i + 1, len(centers)):
            cx1, cy1 = centers[i]
            cx2, cy2 = centers[j]
            dist = np.sqrt((cx1 - cx2)**2 + (cy1 - cy2)**2)
            if dist < 3.0:
                g = graph_ops.add_edge(g, i, j, 1.0)

    # Note: For a true overlay, we'd need to composite the visuals
    # This is a conceptual example showing the field
    metrics = {
        "Field": "Temperature",
        "Centers": len(centers),
        "Resolution": f"{size}x{size}",
        "Method": "Gaussian Mix"
    }
    field_vis = visual.add_metrics(field_vis, metrics, position="bottom-right")

    visual.output(field_vis, "output_showcase_field_network.png")
    print("   Saved: output_showcase_field_network.png")


def create_star(n_nodes):
    """Create star network."""
    g = graph_ops.create(n_nodes)
    for i in range(1, n_nodes):
        g = graph_ops.add_edge(g, 0, i, 1.0)
    return g


def create_ring(n_nodes):
    """Create ring network."""
    g = graph_ops.create(n_nodes)
    for i in range(n_nodes):
        g = graph_ops.add_edge(g, i, (i + 1) % n_nodes, 1.0)
    return g


def create_random(n_nodes, p):
    """Create random network."""
    np.random.seed(42)
    g = graph_ops.create(n_nodes)
    for i in range(n_nodes):
        for j in range(i + 1, n_nodes):
            if np.random.random() < p:
                g = graph_ops.add_edge(g, i, j, 1.0)
    return g


def main():
    print("=" * 60)
    print("Advanced Visualization Showcase")
    print("=" * 60)
    print("\nDemonstrating integration of new visualization features:")
    print("  - Spectrograms with metrics")
    print("  - Network topology analysis")
    print("  - Phase space dynamics")
    print("  - Field + network combinations")

    create_audio_reactive_network()
    create_multi_network_comparison()
    create_dynamical_system_analysis()
    create_combined_field_network()

    print("\n" + "=" * 60)
    print("Showcase complete!")
    print("\nKey features demonstrated:")
    print("  ✓ Multi-domain visualization integration")
    print("  ✓ Metrics dashboard overlays")
    print("  ✓ Property-based coloring")
    print("  ✓ Network topology analysis")
    print("  ✓ Dynamical systems analysis")
    print("  ✓ Custom color palettes")
    print("\nThese tools enable comprehensive visual analysis of")
    print("simulations across audio, physics, networks, and more!")


if __name__ == "__main__":
    main()
