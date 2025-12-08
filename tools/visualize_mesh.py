#!/usr/bin/env python3
"""
Morphogen Cross-Domain Mesh Visualizer

Generates visual representations of the cross-domain transformation mesh using
NetworkX and matplotlib.

Usage:
    python visualize_mesh.py [--output OUTPUT] [--format FORMAT] [--layout LAYOUT]

Options:
    --output OUTPUT    Output file path (default: cross_domain_mesh.png)
    --format FORMAT    Output format: png, svg, pdf (default: png)
    --layout LAYOUT    Graph layout: spring, kamada_kawai, circular, shell (default: kamada_kawai)
"""

import sys
import argparse
from pathlib import Path

# Add morphogen to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import networkx as nx
from morphogen.cross_domain import registry

def create_mesh_graph():
    """Create NetworkX graph from registered transforms.

    Note: Transforms are auto-registered on module import.
    """
    from morphogen.cross_domain.registry import CrossDomainRegistry

    G = nx.DiGraph()
    transforms = CrossDomainRegistry.list_all()

    # Add edges
    for src, tgt in transforms:
        G.add_edge(src, tgt)

    return G, transforms

def categorize_domains():
    """Categorize domains for color coding."""
    return {
        "Field/Spatial": ["field", "spatial", "terrain"],
        "Agents/Behavior": ["agent"],
        "Physics/Simulation": ["physics", "fluid", "acoustics"],
        "Audio/Sound": ["audio", "cepstral", "time", "wavelet"],
        "Visual/Graphics": ["visual", "graph"],
        "Biology": ["cellular"],
        "Geometry": ["cartesian", "polar"],
        "Perception": ["vision"],
    }

def get_category_colors():
    """Get colors for each category."""
    return {
        "Field/Spatial": "#64B5F6",      # Blue
        "Agents/Behavior": "#BA68C8",    # Purple
        "Physics/Simulation": "#FFB74D",  # Orange
        "Audio/Sound": "#81C784",        # Green
        "Visual/Graphics": "#F06292",    # Pink
        "Biology": "#FFF176",            # Yellow
        "Geometry": "#4DB6AC",           # Teal
        "Perception": "#AED581",         # Lime
        "Other": "#BDBDBD",              # Grey
    }

def get_node_colors(G, domains_by_category, category_colors):
    """Assign colors to nodes based on category."""
    domain_to_category = {}
    for cat, domains in domains_by_category.items():
        for d in domains:
            domain_to_category[d] = cat

    colors = []
    for node in G.nodes():
        category = domain_to_category.get(node, "Other")
        colors.append(category_colors.get(category, "#BDBDBD"))

    return colors

def identify_bidirectional_edges(transforms):
    """Identify bidirectional transform pairs."""
    bidirectional = set()
    for src, tgt in transforms:
        if (tgt, src) in transforms and (min(src, tgt), max(src, tgt)) not in bidirectional:
            bidirectional.add((min(src, tgt), max(src, tgt)))
    return bidirectional

def visualize_mesh(output_path="cross_domain_mesh.png", layout="kamada_kawai", dpi=300):
    """Generate mesh visualization."""
    print("Creating mesh graph...")
    G, transforms = create_mesh_graph()

    print(f"Graph stats: {len(G.nodes())} nodes, {len(G.edges())} edges")

    # Setup figure
    fig, ax = plt.subplots(figsize=(16, 12))

    # Get layout
    print(f"Computing {layout} layout...")
    if layout == "spring":
        pos = nx.spring_layout(G, k=2, iterations=50, seed=42)
    elif layout == "kamada_kawai":
        pos = nx.kamada_kawai_layout(G)
    elif layout == "circular":
        pos = nx.circular_layout(G)
    elif layout == "shell":
        pos = nx.shell_layout(G)
    else:
        pos = nx.spring_layout(G, seed=42)

    # Get colors
    domains_by_category = categorize_domains()
    category_colors = get_category_colors()
    node_colors = get_node_colors(G, domains_by_category, category_colors)

    # Identify bidirectional edges
    bidirectional = identify_bidirectional_edges(transforms)

    # Draw unidirectional edges
    unidirectional_edges = [(u, v) for u, v in G.edges()
                            if (min(u, v), max(u, v)) not in bidirectional]
    nx.draw_networkx_edges(G, pos, edgelist=unidirectional_edges,
                          edge_color="#666666", alpha=0.6, ax=ax,
                          arrows=True, arrowsize=20, arrowstyle='->',
                          connectionstyle='arc3,rad=0.1', width=1.5)

    # Draw bidirectional edges
    bidirectional_edges = [(u, v) for u, v in G.edges()
                          if (min(u, v), max(u, v)) in bidirectional]
    if bidirectional_edges:
        nx.draw_networkx_edges(G, pos, edgelist=bidirectional_edges,
                              edge_color="#2196F3", alpha=0.8, ax=ax,
                              arrows=True, arrowsize=20, arrowstyle='<->',
                              connectionstyle='arc3,rad=0.1', width=2.5)

    # Draw nodes
    nx.draw_networkx_nodes(G, pos, node_color=node_colors,
                          node_size=2000, alpha=0.9, ax=ax,
                          edgecolors='#333333', linewidths=2)

    # Draw labels
    nx.draw_networkx_labels(G, pos, font_size=10, font_weight='bold',
                           font_family='sans-serif', ax=ax)

    # Create legend
    legend_patches = []
    for category, color in category_colors.items():
        if category != "Other":
            patch = mpatches.Patch(color=color, label=category)
            legend_patches.append(patch)

    # Add edge type legend
    legend_patches.append(mpatches.Patch(color='white', label=''))  # Spacer
    legend_patches.append(plt.Line2D([0], [0], color='#666666', linewidth=2,
                                    label='Unidirectional'))
    legend_patches.append(plt.Line2D([0], [0], color='#2196F3', linewidth=2,
                                    label='Bidirectional'))

    ax.legend(handles=legend_patches, loc='upper left', fontsize=10,
             framealpha=0.9, title='Domain Categories')

    # Title and styling
    ax.set_title('Morphogen Cross-Domain Transformation Mesh\n' +
                f'{len(G.nodes())} Domains | {len(transforms)} Transforms | ' +
                f'{len(bidirectional)} Bidirectional Pairs',
                fontsize=16, fontweight='bold', pad=20)
    ax.axis('off')
    plt.tight_layout()

    # Save
    print(f"Saving to {output_path}...")
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    print(f"✓ Saved: {output_path}")

    return output_path

def print_mesh_stats():
    """Print mesh statistics."""
    print("=" * 70)
    print("MORPHOGEN CROSS-DOMAIN MESH STATISTICS")
    print("=" * 70)

    G, transforms = create_mesh_graph()
    bidirectional = identify_bidirectional_edges(transforms)

    print(f"\nDomains: {len(G.nodes())}")
    print(f"Transforms: {len(transforms)}")
    print(f"Bidirectional pairs: {len(bidirectional)}")

    # Connectivity analysis
    print(f"\nGraph connectivity:")
    print(f"  Weakly connected: {nx.is_weakly_connected(G)}")
    print(f"  Number of weakly connected components: {nx.number_weakly_connected_components(G)}")

    # Hub analysis
    in_degree = dict(G.in_degree())
    out_degree = dict(G.out_degree())

    print(f"\nTop hubs (by total degree):")
    total_degree = {node: in_degree[node] + out_degree[node] for node in G.nodes()}
    for node, degree in sorted(total_degree.items(), key=lambda x: x[1], reverse=True)[:5]:
        print(f"  {node}: {degree} total ({out_degree[node]} out, {in_degree[node]} in)")

    print(f"\nSink nodes (0 outbound):")
    sinks = [node for node in G.nodes() if out_degree[node] == 0]
    print(f"  {', '.join(sinks) if sinks else 'None'}")

    print(f"\nSource nodes (0 inbound):")
    sources = [node for node in G.nodes() if in_degree[node] == 0]
    print(f"  {', '.join(sources) if sources else 'None'}")

    print("=" * 70)

def main():
    parser = argparse.ArgumentParser(description="Visualize Morphogen cross-domain mesh")
    parser.add_argument("--output", default="cross_domain_mesh.png",
                       help="Output file path")
    parser.add_argument("--format", default="png", choices=["png", "svg", "pdf"],
                       help="Output format")
    parser.add_argument("--layout", default="kamada_kawai",
                       choices=["spring", "kamada_kawai", "circular", "shell"],
                       help="Graph layout algorithm")
    parser.add_argument("--stats", action="store_true",
                       help="Print mesh statistics only")
    parser.add_argument("--dpi", type=int, default=300,
                       help="DPI for raster outputs (default: 300)")

    args = parser.parse_args()

    if args.stats:
        print_mesh_stats()
    else:
        # Ensure output has correct extension
        output_path = args.output
        if not output_path.endswith(f".{args.format}"):
            output_path = output_path.rsplit(".", 1)[0] + f".{args.format}"

        visualize_mesh(output_path, args.layout, args.dpi)
        print("\n")
        print_mesh_stats()

if __name__ == "__main__":
    main()
