"""
Example: Social Network Analysis

Demonstrates graph centrality measures and community detection.
Analyzes a social network to identify influential nodes and communities.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from morphogen.stdlib.graph import GraphOperations, GraphType
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np


def main():
    """Analyze a social network"""

    # Create a social network (15 people)
    people = {
        0: "Alice", 1: "Bob", 2: "Carol", 3: "David", 4: "Emma",
        5: "Frank", 6: "Grace", 7: "Henry", 8: "Iris", 9: "Jack",
        10: "Kate", 11: "Leo", 12: "Mary", 13: "Nick", 14: "Olivia"
    }

    graph = GraphOperations.create_empty(15, GraphType.UNDIRECTED)

    # Add friendships (communities: 0-4, 5-9, 10-14 with some bridges)
    friendships = [
        # Community 1 (0-4)
        (0, 1), (0, 2), (1, 2), (1, 3), (2, 3), (3, 4), (0, 4),
        # Community 2 (5-9)
        (5, 6), (5, 7), (6, 7), (6, 8), (7, 8), (8, 9), (5, 9),
        # Community 3 (10-14)
        (10, 11), (10, 12), (11, 12), (11, 13), (12, 13), (13, 14), (10, 14),
        # Bridges between communities
        (2, 5),   # Carol-Frank (bridge 1-2)
        (4, 10),  # Emma-Kate (bridge 1-3)
        (9, 14),  # Jack-Olivia (bridge 2-3)
    ]

    for u, v in friendships:
        graph = GraphOperations.add_edge(graph, u, v, 1.0)

    print("=== Social Network Analysis ===")
    print(f"\nNetwork: {len(people)} people, {len(friendships)} friendships")

    # 1. Degree Centrality
    print("\n--- Degree Centrality (Most Connected) ---")
    degree_metrics = GraphOperations.degree_centrality(graph)
    degree_sorted = sorted(degree_metrics.node_metrics.items(),
                          key=lambda x: x[1], reverse=True)

    for i, (node, centrality) in enumerate(degree_sorted[:5], 1):
        connections = len(graph.adjacency_list[node])
        print(f"{i}. {people[node]:8} : {centrality:.3f} ({connections} friends)")

    # 2. Betweenness Centrality
    print("\n--- Betweenness Centrality (Bridges/Influencers) ---")
    betweenness_metrics = GraphOperations.betweenness_centrality(graph)
    betweenness_sorted = sorted(betweenness_metrics.node_metrics.items(),
                                key=lambda x: x[1], reverse=True)

    for i, (node, centrality) in enumerate(betweenness_sorted[:5], 1):
        print(f"{i}. {people[node]:8} : {centrality:.3f} (bridge score)")

    # 3. PageRank
    print("\n--- PageRank (Overall Influence) ---")
    pagerank_metrics = GraphOperations.pagerank(graph, damping=0.85)
    pagerank_sorted = sorted(pagerank_metrics.node_metrics.items(),
                            key=lambda x: x[1], reverse=True)

    for i, (node, score) in enumerate(pagerank_sorted[:5], 1):
        print(f"{i}. {people[node]:8} : {score:.4f}")

    # 4. Clustering Coefficient
    print("\n--- Clustering Coefficient (Community Cohesion) ---")
    clustering_metrics = GraphOperations.clustering_coefficient(graph)
    clustering_sorted = sorted(clustering_metrics.node_metrics.items(),
                              key=lambda x: x[1], reverse=True)

    for i, (node, coef) in enumerate(clustering_sorted[:5], 1):
        print(f"{i}. {people[node]:8} : {coef:.3f} (friend group cohesion)")

    # 5. Connected Components
    print("\n--- Community Detection ---")
    components = GraphOperations.connected_components(graph)
    print(f"Number of separate communities: {len(components)}")

    for i, component in enumerate(components, 1):
        members = [people[node] for node in sorted(component)]
        print(f"Community {i}: {', '.join(members)} ({len(component)} members)")

    # Visualize the network
    visualize_social_network(graph, people, degree_metrics, betweenness_metrics)


def visualize_social_network(graph, people, degree_metrics, betweenness_metrics):
    """Visualize the social network with centrality measures"""

    # Create NetworkX graph
    G = nx.Graph()

    for node, name in people.items():
        G.add_node(node, label=name)

    for node, neighbors in graph.adjacency_list.items():
        for neighbor, _ in neighbors:
            if node < neighbor:
                G.add_edge(node, neighbor)

    # Create figure with subplots
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    # Layout
    pos = nx.spring_layout(G, seed=42, k=0.5)

    # Plot 1: Degree Centrality
    ax = axes[0]
    plt.sca(ax)

    # Node sizes based on degree centrality
    node_sizes = [degree_metrics.node_metrics[node] * 2000 + 300 for node in G.nodes()]
    node_colors = [degree_metrics.node_metrics[node] for node in G.nodes()]

    nx.draw_networkx_edges(G, pos, alpha=0.3, width=2)
    nodes = nx.draw_networkx_nodes(G, pos, node_size=node_sizes,
                                   node_color=node_colors, cmap='YlOrRd',
                                   vmin=0, vmax=max(node_colors))

    labels = {node: people[node] for node in G.nodes()}
    nx.draw_networkx_labels(G, pos, labels, font_size=8, font_weight='bold')

    plt.colorbar(nodes, ax=ax, label='Degree Centrality')
    ax.set_title("Degree Centrality\n(Node size = # connections)", fontweight='bold')
    ax.axis('off')

    # Plot 2: Betweenness Centrality
    ax = axes[1]
    plt.sca(ax)

    # Node sizes based on betweenness centrality
    betweenness_values = [betweenness_metrics.node_metrics[node] for node in G.nodes()]
    max_betweenness = max(betweenness_values) if max(betweenness_values) > 0 else 1
    node_sizes = [betweenness_metrics.node_metrics[node] / max_betweenness * 1500 + 300
                  for node in G.nodes()]

    nx.draw_networkx_edges(G, pos, alpha=0.3, width=2)
    nodes = nx.draw_networkx_nodes(G, pos, node_size=node_sizes,
                                   node_color=betweenness_values, cmap='RdPu',
                                   vmin=0, vmax=max(betweenness_values))

    nx.draw_networkx_labels(G, pos, labels, font_size=8, font_weight='bold')

    plt.colorbar(nodes, ax=ax, label='Betweenness Centrality')
    ax.set_title("Betweenness Centrality\n(Node size = bridge importance)", fontweight='bold')
    ax.axis('off')

    plt.suptitle("Social Network Analysis - Centrality Measures", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('/tmp/graph_network_analysis.png', dpi=150, bbox_inches='tight')
    print("\nVisualization saved to: /tmp/graph_network_analysis.png")
    plt.show()


if __name__ == "__main__":
    main()
