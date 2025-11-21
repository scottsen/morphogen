"""Graph/Network Visualization Demo

Demonstrates the new visual.graph() function for network visualization.
Shows different layouts and centrality-based coloring.
"""

import numpy as np
from morphogen.stdlib import graph as graph_ops
from morphogen.stdlib import visual


def create_small_world_network(n_nodes=20, k_neighbors=4, rewire_prob=0.3, seed=42):
    """Create a small-world network (Watts-Strogatz model)."""
    np.random.seed(seed)

    # Start with ring lattice
    g = graph_ops.create(n_nodes)

    # Connect to k nearest neighbors
    for i in range(n_nodes):
        for j in range(1, k_neighbors // 2 + 1):
            neighbor = (i + j) % n_nodes
            g = graph_ops.add_edge(g, i, neighbor, 1.0)

    # Rewire edges with probability p
    edges = []
    for i in range(n_nodes):
        for j in range(i + 1, n_nodes):
            if g.get_edge(i, j) > 0:
                edges.append((i, j))

    for i, j in edges:
        if np.random.random() < rewire_prob:
            # Rewire to random node
            new_target = np.random.randint(0, n_nodes)
            if new_target != i and g.get_edge(i, new_target) == 0:
                g = graph_ops.remove_edge(g, i, j)
                g = graph_ops.add_edge(g, i, new_target, 1.0)

    return g


def create_scale_free_network(n_nodes=30, m_edges=2, seed=42):
    """Create a scale-free network (Barabási-Albert model)."""
    np.random.seed(seed)

    # Start with small complete graph
    g = graph_ops.create(n_nodes)

    # Initial complete graph
    for i in range(m_edges):
        for j in range(i + 1, m_edges):
            g = graph_ops.add_edge(g, i, j, 1.0)

    # Add remaining nodes with preferential attachment
    for new_node in range(m_edges, n_nodes):
        # Calculate degree-based probabilities
        degrees = np.array([np.sum(g.adj[i] > 0) for i in range(new_node)])
        if np.sum(degrees) == 0:
            probs = np.ones(new_node) / new_node
        else:
            probs = degrees / np.sum(degrees)

        # Select m_edges targets based on preferential attachment
        targets = np.random.choice(new_node, size=min(m_edges, new_node), replace=False, p=probs)

        for target in targets:
            g = graph_ops.add_edge(g, new_node, target, 1.0)

    return g


def create_star_network(n_nodes=15):
    """Create a star network (one central hub)."""
    g = graph_ops.create(n_nodes)

    # Connect all nodes to node 0 (the hub)
    for i in range(1, n_nodes):
        g = graph_ops.add_edge(g, 0, i, 1.0)

    return g


def create_grid_network(rows=5, cols=5):
    """Create a 2D grid network."""
    n_nodes = rows * cols
    g = graph_ops.create(n_nodes)

    for i in range(rows):
        for j in range(cols):
            node = i * cols + j

            # Connect to right neighbor
            if j < cols - 1:
                right = i * cols + (j + 1)
                g = graph_ops.add_edge(g, node, right, 1.0)

            # Connect to bottom neighbor
            if i < rows - 1:
                bottom = (i + 1) * cols + j
                g = graph_ops.add_edge(g, node, bottom, 1.0)

    return g


def main():
    print("Graph/Network Visualization Demo")
    print("=" * 50)

    # Example 1: Small-world network with force-directed layout
    print("\n1. Small-world network (Watts-Strogatz)...")
    g = create_small_world_network(n_nodes=20, k_neighbors=4, rewire_prob=0.3)

    print("   Visualizing with force-directed layout...")
    vis = visual.graph(
        g,
        width=800,
        height=800,
        layout="force",
        iterations=100,
        node_size=10.0,
        edge_color=(0.3, 0.3, 0.3),
        node_color=(0.2, 0.6, 1.0),
        background=(0.05, 0.05, 0.1)
    )

    visual.output(vis, "output_graph_smallworld_force.png")
    print("   Saved: output_graph_smallworld_force.png")

    # Example 2: Scale-free network with centrality coloring
    print("\n2. Scale-free network (Barabási-Albert)...")
    g = create_scale_free_network(n_nodes=30, m_edges=2)

    print("   Visualizing with centrality-based coloring...")
    vis = visual.graph(
        g,
        width=800,
        height=800,
        layout="force",
        iterations=150,
        node_size=12.0,
        edge_color=(0.4, 0.4, 0.4),
        edge_width=1.5,
        color_by_centrality=True,
        palette="fire",
        background=(0.0, 0.0, 0.0)
    )

    # Add metrics
    n_edges = np.sum(g.adj > 0) // 2
    metrics = {
        "Nodes": g.n_nodes,
        "Edges": n_edges,
        "Type": "Scale-free",
        "Layout": "Force-directed"
    }
    vis = visual.add_metrics(vis, metrics, position="top-left")

    visual.output(vis, "output_graph_scalefree_centrality.png")
    print("   Saved: output_graph_scalefree_centrality.png")

    # Example 3: Star network with circular layout
    print("\n3. Star network (hub-and-spoke)...")
    g = create_star_network(n_nodes=15)

    print("   Visualizing with circular layout...")
    vis = visual.graph(
        g,
        width=600,
        height=600,
        layout="circular",
        node_size=15.0,
        edge_color=(0.6, 0.3, 0.8),
        edge_width=2.0,
        color_by_centrality=True,
        palette="viridis",
        background=(0.1, 0.1, 0.15)
    )

    visual.output(vis, "output_graph_star_circular.png")
    print("   Saved: output_graph_star_circular.png")

    # Example 4: Grid network with grid layout
    print("\n4. 2D grid network...")
    g = create_grid_network(rows=6, cols=6)

    print("   Visualizing with grid layout...")
    vis = visual.graph(
        g,
        width=700,
        height=700,
        layout="grid",
        node_size=8.0,
        edge_color=(0.5, 0.7, 0.5),
        edge_width=2.0,
        node_color=(0.3, 0.8, 0.3),
        background=(0.0, 0.0, 0.0)
    )

    visual.output(vis, "output_graph_grid.png")
    print("   Saved: output_graph_grid.png")

    # Example 5: Comparison of layouts on same network
    print("\n5. Comparing layouts (small network)...")
    g = create_small_world_network(n_nodes=12, k_neighbors=4, rewire_prob=0.2, seed=123)

    layouts = ["force", "circular", "grid"]
    for layout_name in layouts:
        vis = visual.graph(
            g,
            width=500,
            height=500,
            layout=layout_name,
            iterations=80,
            node_size=12.0,
            color_by_centrality=True,
            palette="coolwarm",
            edge_color=(0.5, 0.5, 0.5),
            background=(0.05, 0.05, 0.05)
        )

        filename = f"output_graph_layout_{layout_name}.png"
        visual.output(vis, filename)
        print(f"   Saved: {filename}")

    print("\n" + "=" * 50)
    print("Demo complete! Graph visualizations showcase:")
    print("  - Force-directed layout (Fruchterman-Reingold)")
    print("  - Circular and grid layouts")
    print("  - Degree centrality coloring")
    print("  - Multiple network topologies")
    print("  - Customizable styling (colors, sizes, edges)")


if __name__ == "__main__":
    main()
