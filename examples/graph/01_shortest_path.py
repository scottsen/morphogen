"""
Example: Shortest Path Finding

Demonstrates graph creation and Dijkstra's shortest path algorithm.
Creates a weighted graph and finds optimal routes between nodes.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from morphogen.stdlib.graph import GraphOperations, GraphType
import matplotlib.pyplot as plt
import networkx as nx


def main():
    """Find shortest paths in a road network"""

    # Create a simple road network (7 cities)
    cities = {
        0: "CityA",
        1: "CityB",
        2: "CityC",
        3: "CityD",
        4: "CityE",
        5: "CityF",
        6: "CityG"
    }

    graph = GraphOperations.create_empty(7, GraphType.UNDIRECTED)

    # Add roads with distances (weights)
    roads = [
        (0, 1, 4.0),  # A-B: 4km
        (0, 2, 3.0),  # A-C: 3km
        (1, 2, 1.0),  # B-C: 1km
        (1, 3, 2.0),  # B-D: 2km
        (2, 3, 4.0),  # C-D: 4km
        (2, 4, 2.0),  # C-E: 2km
        (3, 5, 3.0),  # D-F: 3km
        (4, 5, 2.0),  # E-F: 2km
        (5, 6, 1.0),  # F-G: 1km
    ]

    for u, v, weight in roads:
        graph = GraphOperations.add_edge(graph, u, v, weight)

    # Find shortest path from City A (0) to City G (6)
    start, end = 0, 6
    path, distance = GraphOperations.shortest_path(graph, start, end)

    print("=== Shortest Path Analysis ===")
    print(f"\nRoad Network: {len(cities)} cities, {len(roads)} roads")
    print(f"\nFinding shortest route from {cities[start]} to {cities[end]}...")
    print(f"\nShortest path: {' -> '.join(cities[n] for n in path)}")
    print(f"Total distance: {distance:.1f} km")

    # Find all shortest paths from City A
    print(f"\n=== All Routes from {cities[start]} ===")
    distances, predecessors = GraphOperations.dijkstra(graph, start)

    for city_id, city_name in cities.items():
        if city_id != start:
            dist = distances[city_id]
            # Reconstruct path
            path = []
            current = city_id
            while current is not None:
                path.append(current)
                current = predecessors[current]
            path.reverse()

            route = ' -> '.join(cities[n] for n in path)
            print(f"{city_name:8} : {dist:5.1f} km via {route}")

    # Visualize the network
    visualize_network(graph, cities, start, end, path)


def visualize_network(graph, cities, start, end, shortest_path):
    """Visualize the road network and shortest path"""

    # Create NetworkX graph for visualization
    G = nx.Graph()

    # Add nodes
    for node_id, city_name in cities.items():
        G.add_node(node_id, label=city_name)

    # Add edges
    for node, neighbors in graph.adjacency_list.items():
        for neighbor, weight in neighbors:
            if node < neighbor:  # Avoid duplicates in undirected graph
                G.add_edge(node, neighbor, weight=weight)

    # Layout
    pos = nx.spring_layout(G, seed=42)

    plt.figure(figsize=(12, 8))

    # Draw all edges in light gray
    nx.draw_networkx_edges(G, pos, width=1, alpha=0.3, edge_color='gray')

    # Draw shortest path edges in red
    path_edges = [(shortest_path[i], shortest_path[i+1]) for i in range(len(shortest_path)-1)]
    nx.draw_networkx_edges(G, pos, edgelist=path_edges, width=3, edge_color='red',
                          label='Shortest Path')

    # Draw nodes
    node_colors = []
    for node in G.nodes():
        if node == start:
            node_colors.append('green')
        elif node == end:
            node_colors.append('red')
        elif node in shortest_path:
            node_colors.append('orange')
        else:
            node_colors.append('lightblue')

    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=700)

    # Draw labels
    labels = {node: cities[node] for node in G.nodes()}
    nx.draw_networkx_labels(G, pos, labels, font_size=10, font_weight='bold')

    # Draw edge weights
    edge_labels = nx.get_edge_attributes(G, 'weight')
    edge_labels = {k: f'{v:.1f}km' for k, v in edge_labels.items()}
    nx.draw_networkx_edge_labels(G, pos, edge_labels, font_size=8)

    plt.title("Road Network - Shortest Path Highlighted", fontsize=14, fontweight='bold')
    plt.legend(['Shortest Path'], loc='upper right')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig('/tmp/graph_shortest_path.png', dpi=150, bbox_inches='tight')
    print("\nVisualization saved to: /tmp/graph_shortest_path.png")
    plt.show()


if __name__ == "__main__":
    main()
