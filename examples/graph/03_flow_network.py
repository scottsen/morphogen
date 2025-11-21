"""
Example: Maximum Flow Network

Demonstrates max flow algorithm for network capacity problems.
Models a water distribution or network traffic routing system.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from morphogen.stdlib.graph import GraphOperations, GraphType
import matplotlib.pyplot as plt
import networkx as nx


def main():
    """Find maximum flow in a network"""

    # Create a flow network (water distribution system)
    nodes = {
        0: "Source",
        1: "PumpA",
        2: "PumpB",
        3: "TankC",
        4: "TankD",
        5: "Sink"
    }

    graph = GraphOperations.create_empty(6, GraphType.DIRECTED)

    # Add pipes with capacities (liters/minute)
    pipes = [
        (0, 1, 10.0),  # Source -> PumpA: 10 L/min
        (0, 2, 10.0),  # Source -> PumpB: 10 L/min
        (1, 2, 2.0),   # PumpA -> PumpB: 2 L/min
        (1, 3, 8.0),   # PumpA -> TankC: 8 L/min
        (2, 4, 9.0),   # PumpB -> TankD: 9 L/min
        (3, 4, 3.0),   # TankC -> TankD: 3 L/min
        (3, 5, 10.0),  # TankC -> Sink: 10 L/min
        (4, 5, 10.0),  # TankD -> Sink: 10 L/min
    ]

    for u, v, capacity in pipes:
        graph = GraphOperations.add_edge(graph, u, v, capacity)

    # Find maximum flow
    source, sink = 0, 5
    max_flow_value = GraphOperations.max_flow(graph, source, sink)

    print("=== Maximum Flow Analysis ===")
    print(f"\nWater Distribution Network: {len(nodes)} nodes, {len(pipes)} pipes")
    print(f"\nSource: {nodes[source]}")
    print(f"Sink: {nodes[sink]}")
    print(f"\n✓ Maximum flow: {max_flow_value:.1f} liters/minute")

    # Show pipe capacities
    print("\n--- Pipe Capacities ---")
    for u, v, capacity in pipes:
        print(f"{nodes[u]:8} → {nodes[v]:8} : {capacity:5.1f} L/min")

    # Calculate utilization percentage
    total_capacity = sum(capacity for _, _, capacity in pipes)
    utilization = (max_flow_value / (total_capacity / 2)) * 100  # Divide by 2 to account for directed paths

    print(f"\n--- Network Statistics ---")
    print(f"Total pipe capacity: {total_capacity:.1f} L/min")
    print(f"Flow utilization: {utilization:.1f}%")

    # Find bottleneck analysis
    print("\n--- Bottleneck Analysis ---")
    print("Critical paths (pipes at full capacity):")

    # For demonstration, identify low-capacity pipes
    bottlenecks = [(u, v, cap) for u, v, cap in pipes if cap < 5.0]
    for u, v, capacity in bottlenecks:
        print(f"  • {nodes[u]} → {nodes[v]}: {capacity:.1f} L/min (potential bottleneck)")

    # Visualize the network
    visualize_flow_network(graph, nodes, pipes, source, sink, max_flow_value)


def visualize_flow_network(graph, nodes, pipes, source, sink, max_flow):
    """Visualize the flow network"""

    # Create NetworkX directed graph
    G = nx.DiGraph()

    for node_id, node_name in nodes.items():
        G.add_node(node_id, label=node_name)

    for u, v, capacity in pipes:
        G.add_edge(u, v, capacity=capacity)

    # Hierarchical layout
    pos = nx.spring_layout(G, seed=42, k=2)

    # Manual positioning for better visualization
    pos = {
        0: (0, 1),     # Source (left, top)
        1: (1, 1.5),   # PumpA
        2: (1, 0.5),   # PumpB
        3: (2, 1.5),   # TankC
        4: (2, 0.5),   # TankD
        5: (3, 1),     # Sink (right, middle)
    }

    plt.figure(figsize=(14, 8))

    # Draw edges with widths proportional to capacity
    edge_widths = [G[u][v]['capacity'] / 2 for u, v in G.edges()]
    nx.draw_networkx_edges(G, pos, width=edge_widths, edge_color='steelblue',
                          arrows=True, arrowsize=20, arrowstyle='->', alpha=0.7,
                          connectionstyle='arc3,rad=0.1')

    # Draw nodes
    node_colors = []
    for node in G.nodes():
        if node == source:
            node_colors.append('limegreen')
        elif node == sink:
            node_colors.append('crimson')
        else:
            node_colors.append('skyblue')

    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=1200,
                          edgecolors='black', linewidths=2)

    # Draw labels
    labels = {node: nodes[node] for node in G.nodes()}
    nx.draw_networkx_labels(G, pos, labels, font_size=10, font_weight='bold')

    # Draw edge labels (capacities)
    edge_labels = {(u, v): f"{G[u][v]['capacity']:.0f} L/m"
                  for u, v in G.edges()}
    nx.draw_networkx_edge_labels(G, pos, edge_labels, font_size=9,
                                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

    # Add title with max flow
    plt.title(f"Water Distribution Network - Maximum Flow: {max_flow:.1f} L/min",
             fontsize=14, fontweight='bold', pad=20)

    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='limegreen', edgecolor='black', label='Source'),
        Patch(facecolor='crimson', edgecolor='black', label='Sink'),
        Patch(facecolor='skyblue', edgecolor='black', label='Intermediate')
    ]
    plt.legend(handles=legend_elements, loc='upper left', fontsize=10)

    plt.axis('off')
    plt.tight_layout()
    plt.savefig('/tmp/graph_flow_network.png', dpi=150, bbox_inches='tight')
    print("\nVisualization saved to: /tmp/graph_flow_network.png")
    plt.show()


if __name__ == "__main__":
    main()
