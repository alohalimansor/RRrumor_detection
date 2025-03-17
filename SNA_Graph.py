import networkx as nx
import matplotlib.pyplot as plt
from enhanced_explainability import NovelRumorExplainer
import re
import numpy as np
from matplotlib.patches import FancyBboxPatch

# Initialize the explainer
explainer = NovelRumorExplainer()

# Rumor text from your paper
rumor_text = "but mike brown knew about the robbery and DIDN'T know the cop didn't know - that's what makes it POSSIBLY relevant user_handle: DefinitelyMay_b topic: ferguson"


# Function to build optimized SNA graph without adjustText
def build_optimized_sna_graph(text, is_rumor=True):
    # Create a graph
    G = nx.Graph()

    # Define nodes, categories, and weights explicitly for complete control
    nodes = {
        # Core entities
        "mike brown": {"category": "entity", "weight": 0.9},
        "cop": {"category": "entity", "weight": 0.7},
        "robbery": {"category": "entity", "weight": 0.8},

        # Knowledge states
        "knew_1": {"category": "knowledge_state", "weight": 0.8},
        "knew_2": {"category": "knowledge_state", "weight": 0.7},
        "didn't know": {"category": "knowledge_state", "weight": 0.7},

        # Categories
        "nested knowledge": {"category": "category", "weight": 0.8},
        "authority-knowledge tension": {"category": "category", "weight": 0.8},
        "relevance framing": {"category": "category", "weight": 0.8},

        # Source and topic
        "DefinitelyMay_b": {"category": "source", "weight": 0.7},
        "ferguson": {"category": "topic", "weight": 0.7},

        # Rhetorical elements
        "POSSIBLY": {"category": "hedge", "weight": 0.6},
        "relevant": {"category": "judgment", "weight": 0.6},
    }

    # Add all nodes
    for node, attrs in nodes.items():
        G.add_node(node)

    # Define relationships explicitly for complete control
    relationships = [
        # Mike Brown relationships
        ("mike brown", "knew_1", 0.8),
        ("mike brown", "robbery", 0.8),
        ("mike brown", "cop", 0.7),
        ("mike brown", "ferguson", 0.6),

        # Knowledge relationships
        ("knew_1", "knew_2", 0.7),
        ("knew_1", "robbery", 0.7),
        ("nested knowledge", "knew_1", 0.6),
        ("nested knowledge", "knew_2", 0.6),

        # Authority-knowledge relationships
        ("authority-knowledge tension", "cop", 0.7),
        ("authority-knowledge tension", "didn't know", 0.7),
        ("cop", "didn't know", 0.8),

        # Relevance relationships
        ("relevance framing", "POSSIBLY", 0.6),
        ("relevance framing", "relevant", 0.6),
        ("POSSIBLY", "relevant", 0.7),

        # Source relationship
        ("DefinitelyMay_b", "mike brown", 0.5),
    ]

    # Add all edges
    G.add_edges_from([(u, v, {'weight': w}) for u, v, w in relationships])

    # Create color map based on node categories with improved colors
    color_map = {
        "category": "#20B2AA",  # Medium turquoise for categories
        "entity": "#9370DB",  # Medium purple for entities
        "knowledge_state": "#FF6347",  # Tomato red for knowledge states
        "source": "#4682B4",  # Steel blue for sources
        "topic": "#FF8C00",  # Dark orange for topics
        "emphasis": "#32CD32",  # Lime green for emphasized words
        "hedge": "#FF69B4",  # Hot pink for hedges
        "judgment": "#C0C0C0"  # Silver for judgments
    }

    # Draw graph with enhanced visual features
    plt.figure(figsize=(16, 12))

    # Custom node positioning for optimal clarity - CRUCIAL for avoiding overlaps
    # Define specific positions for each node to prevent overlaps
    custom_pos = {
        # Main entities in center
        "mike brown": (0, 0),
        "robbery": (1.5, -0.8),
        "cop": (1.2, 1.5),

        # Knowledge states
        "didn't know": (2.5, 2.2),
        "knew_1": (-0.5, -1.5),
        "knew_2": (-1.5, -2.5),

        # Categories - positioned further out
        "nested knowledge": (-2.5, -1.8),
        "authority-knowledge tension": (3, 1),
        "relevance framing": (-2.8, 1.5),

        # Rhetorical elements
        "POSSIBLY": (-3.2, 2.7),
        "relevant": (-1.8, 3),

        # Source and topic
        "DefinitelyMay_b": (-3, -0.5),
        "ferguson": (2.5, -1.2)
    }

    # Use the custom positions
    pos = custom_pos

    # Create semantic zones
    semantic_zones = {
        "Knowledge Attribution": ["mike brown", "knew_1", "knew_2", "robbery", "nested knowledge"],
        "Authority Tension": ["cop", "didn't know", "authority-knowledge tension"],
        "Rhetorical Framing": ["POSSIBLY", "relevant", "relevance framing"]
    }

    # Draw zone backgrounds first (so they're behind everything else)
    for zone_name, zone_nodes in semantic_zones.items():
        # Get positions of nodes in this zone
        zone_positions = [pos[node] for node in zone_nodes if node in pos]
        if zone_positions:
            # Calculate zone boundary
            xs, ys = zip(*zone_positions)
            min_x, max_x = min(xs) - 0.5, max(xs) + 0.5
            min_y, max_y = min(ys) - 0.5, max(ys) + 0.5
            width, height = max_x - min_x, max_y - min_y

            center_x, center_y = (min_x + max_x) / 2, (min_y + max_y) / 2

            # Create ellipse for zone
            ellipse = plt.matplotlib.patches.Ellipse(
                (center_x, center_y), width * 1.8, height * 1.5,
                alpha=0.15, facecolor='lightgray', edgecolor=None, zorder=-10
            )
            plt.gca().add_patch(ellipse)

            # Add zone label (ensuring it doesn't overlap with nodes)
            label_y = min_y - 0.3
            plt.text(center_x, label_y, zone_name,
                     ha='center', va='center', fontsize=12,
                     fontweight='bold', color='dimgray', zorder=5)

    # Draw edges with better styling
    for (u, v, data) in G.edges(data=True):
        edge_weight = data['weight']
        alpha = 0.4 + edge_weight * 0.4  # Scale alpha between 0.4 and 0.8

        # Determine if we need a curved edge (for cleaner look)
        # If nodes are far apart, use straight lines
        x1, y1 = pos[u]
        x2, y2 = pos[v]
        dist = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

        if dist < 2.5:
            rad = 0.15  # Less curve for shorter distances
        else:
            rad = 0.0  # No curve for longer distances

        # Draw edge
        nx.draw_networkx_edges(G, pos, edgelist=[(u, v)], width=edge_weight * 3,
                               alpha=alpha, edge_color="gray",
                               connectionstyle=f'arc3,rad={rad}')

    # Draw nodes with colors by category and size by weight
    for node, attrs in nodes.items():
        category = attrs["category"]
        weight = attrs["weight"]
        color = color_map[category]
        size = weight * 3000

        # Draw node
        nx.draw_networkx_nodes(G, pos,
                               nodelist=[node],
                               node_size=size,
                               node_color=color,
                               edgecolors='white',
                               linewidths=2,
                               alpha=0.85)

    # Draw labels with white background boxes to prevent overlap with edges
    for node in G.nodes():
        x, y = pos[node]
        fontsize = 13 if nodes[node]["weight"] >= 0.8 else 12

        # Create a white background for the text
        # The bbox ensures text stands out from edges and nodes
        plt.text(x, y, node,
                 horizontalalignment='center',
                 verticalalignment='center',
                 fontsize=fontsize,
                 fontweight='bold',
                 bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=1.0),
                 zorder=10)  # Higher zorder to keep on top

    # Build a more organized and clear legend
    legend_elements = []

    # Group legend by semantic function
    legend_groups = {
        "Semantic Categories": ["category"],
        "Entities": ["entity", "source", "topic"],
        "Knowledge Relations": ["knowledge_state"],
        "Rhetorical Elements": ["hedge", "judgment"]
    }

    # Build legend items by group
    for group_name, categories in legend_groups.items():
        # Add group header
        legend_elements.append(plt.Line2D([0], [0], marker='', color='none',
                                          markerfacecolor='none', markersize=0,
                                          label=group_name))

        # Add items in this group
        for category in categories:
            if category in color_map:
                legend_elements.append(plt.Line2D([0], [0], marker='o', color='white',
                                                  markerfacecolor=color_map[category], markersize=15,
                                                  markeredgecolor='white', markeredgewidth=1,
                                                  label=category.replace('_', ' ').title()))

    plt.legend(handles=legend_elements, loc='upper right', title="Node Categories",
               frameon=True, framealpha=0.9, facecolor='white', edgecolor='lightgray',
               title_fontsize=14, fontsize=12)

    # Add a stylish title
    plt.title('Enhanced Semantic Network Analysis of Rumor Text',
              fontsize=22, fontweight='bold', y=1.02,
              fontname='Arial')

    # Turn off the axis for a cleaner look
    plt.axis('off')

    # Add a clean text box for the original text
    wrapped_text = '\n'.join([text[i:i + 70] for i in range(0, len(text), 70)])
    plt.figtext(0.5, 0.05, f"Original text:\n{wrapped_text}",
                ha="center", fontsize=12,
                bbox={"boxstyle": "round,pad=0.8", "facecolor": "#F5F5F5",
                      "edgecolor": "darkgray", "alpha": 0.9})

    plt.tight_layout()
    plt.savefig('final_optimized_rumor_graph.png', dpi=300, bbox_inches='tight')
    plt.show()

    return G


# Generate optimized graph
rumor_graph = build_optimized_sna_graph(rumor_text, is_rumor=True)