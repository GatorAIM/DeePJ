from typing import Union, Dict
import torch
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
import matplotlib.patches as patches
from adjustText import adjust_text
import seaborn as sns
from copy import deepcopy as c


def plot_heatmap(ax, mtx, title):
    # if mtx is torch.Tensor, convert to numpy
    if isinstance(mtx, torch.Tensor):
        mtx = mtx.detach().numpy()
    sns.heatmap(mtx, square=True, cmap='Blues', cbar_kws={"shrink": .8}, ax = ax)
    ax.set_title(title)
    
    
    
def convert_to_numpy(adj_matrix, timestamps, clusters, clusters_weights, code_ints):
    """Convert PyTorch tensors to NumPy arrays if necessary."""
    if isinstance(adj_matrix, torch.Tensor):
        adj_matrix = adj_matrix.numpy()
    if isinstance(timestamps, torch.Tensor):
        timestamps = timestamps.numpy()
    if isinstance(clusters, torch.Tensor):
        clusters = clusters.numpy()
    if isinstance(clusters_weights, torch.Tensor):
        clusters_weights = clusters_weights.numpy()
    if isinstance(code_ints, torch.Tensor):
        code_ints = code_ints.numpy()
    return adj_matrix.T, np.array(timestamps), np.array(clusters), np.array(clusters_weights)[0], code_ints

def filter_graph_edges_and_nodes(G, threshold):
    """Remove edges below a threshold and delete isolated nodes."""
    edges_to_remove = [(u, v) for u, v, d in G.edges(data=True) if d["weight"] < threshold]
    G.remove_edges_from(edges_to_remove)

    isolated_nodes = list(nx.isolates(G))
    G.remove_nodes_from(isolated_nodes)

    return G, isolated_nodes

def normalize_timestamps(timestamps):
    """Normalize timestamps to the range [0, 1], handling constant timestamps."""
    if timestamps.max() == timestamps.min():
        return np.full_like(timestamps, 0.5, dtype=float)  # 设为固定值 0.5
    return (timestamps - timestamps.min()) / (timestamps.max() - timestamps.min())

def compute_node_positions(G, timestamps, clusters):
    """Compute node positions with jitter based on timestamps and clusters."""
    unique_timestamps = sorted(set(timestamps))
    unique_clusters = sorted(set(clusters))

    ts_base_positions = {ts: i for i, ts in enumerate(unique_timestamps)}
    cl_base_positions = {cl: i for i, cl in enumerate(unique_clusters)}

    groups = {(ts, cl): [] for ts in unique_timestamps for cl in unique_clusters}
    for node, ts, cl in zip(G.nodes, timestamps, clusters):
        groups[(ts, cl)].append(node)

    pos = {}
    x_block_spacing, y_block_spacing = 8, 3.0
    x_jitter_scale, y_jitter_scale = 3.0, 0.8

    for (ts, cl), nodes in groups.items():
        base_x, base_y = ts_base_positions[ts] * x_block_spacing, cl_base_positions[cl] * y_block_spacing
        for node in nodes:
            x_jitter = np.random.uniform(-x_jitter_scale, x_jitter_scale)
            y_jitter = np.random.uniform(-y_jitter_scale, y_jitter_scale)
            pos[node] = (base_x + x_jitter, base_y + y_jitter)
    
    return pos, unique_clusters, unique_timestamps, x_block_spacing

def draw_graph(G, pos, clusters, unique_clusters, unique_timestamps, x_block_spacing, all_int2str, raw_timestamps, clusters_weights, code_ints):
    """Draw the graph with custom formatting, including the timeline."""
    cmap = ListedColormap(plt.colormaps["tab10"].colors[:len(unique_clusters)])
    node_colors = [cmap(clusters[i]) for i in range(len(clusters))]

    edge_weights = nx.get_edge_attributes(G, "weight")
    max_weight = max(edge_weights.values()) if edge_weights else 1
    edge_widths = [0.05 + (weight / max_weight) * 2.0 for weight in edge_weights.values()]

    node_labels = {node: all_int2str.get(code_ints[node], f"Node {node}") for node in G.nodes}

    plt.figure(figsize=(14, 7))
    nx.draw(G, pos, with_labels=False, node_color=node_colors, edge_color="gray",
            node_size=200, width=edge_widths, arrows=True, connectionstyle="arc3,rad=0.2")

    texts = []
    for node, (x, y) in pos.items():
        label = node_labels[node]
        label_x, label_y = x + 0.4, y + 0.4
        texts.append(plt.text(label_x, label_y, label, fontsize=10, bbox=dict(facecolor="white", alpha=0.5, edgecolor="gray")))
        plt.plot([x, label_x], [y, label_y], color="black", linestyle="--", linewidth=0.2)
    
    adjust_text(texts, arrowprops=dict(arrowstyle="-", color="black", lw=0.2))

    draw_timeline(unique_timestamps, x_block_spacing, raw_timestamps)

    legend_elements = [Patch(facecolor=cmap(i), edgecolor="black", label=f"CM {unique_clusters[i] + 1} ({clusters_weights[unique_clusters[i]]:.2f})") 
                       for i in range(len(unique_clusters))]
    plt.legend(handles=legend_elements, loc="upper left", title="Clincial Modules", fontsize=12)
    plt.show()

def draw_timeline(unique_timestamps, x_block_spacing, raw_timestamps):
    """Draws a timeline at the bottom of the graph with labeled events and intervals."""
    ax = plt.gca()  # Get the current axes
    y_min, y_max = ax.get_ylim()  # Get y-axis limits
    timeline_y = y_min + 0.05 * (y_max - y_min)  # Position timeline at 5% from the bottom
    tick_height = 0.05 * (y_max - y_min)  # Tick height

    # Step 1: Extract sorted unique timestamps
    unique_values = sorted(set(raw_timestamps))

    # Step 2: If only one unique value (e.g., [0]), do nothing
    if len(unique_values) == 1 and unique_values[0] == 0:
        return  

    # Step 3: Compute differences between consecutive timestamps
    time_diffs = [unique_values[i] - unique_values[i - 1] for i in range(1, len(unique_values))]

    # Step 4: Draw timeline segments and labels
    segment_positions = []  # Store x_start positions for arrow placement
    for idx, _ in enumerate(unique_timestamps):
        x_start = (idx - 0.5) * x_block_spacing
        x_end = x_start + x_block_spacing
        segment_positions.append(x_start)  # Save for later arrow placement

        plt.plot([x_start, x_end], [timeline_y, timeline_y], color="black", linewidth=1)
        plt.vlines(x_start, timeline_y, timeline_y + tick_height, color="black", linewidth=1)
        plt.vlines(x_end, timeline_y, timeline_y + tick_height, color="black", linewidth=1)

        label = f"Encounter {idx + 1}"
        plt.text((x_start + x_end) / 2, timeline_y - 0.04 * (y_max - y_min),
                 label, ha="center", fontsize=15)

    # Step 5: Draw upward curved arrows between timeline segments
    for i, diff in enumerate(time_diffs):
        x_start = segment_positions[i] + (7/8) * x_block_spacing  # 7/8 of first segment
        x_end = segment_positions[i + 1] + (1/8) * x_block_spacing  # 1/8 of second segment
        y_arrow = timeline_y + 0.01  # Make the curve go upward

        # Define the curved line with an arrowhead only at the right side
        arrow = patches.FancyArrowPatch((x_start, y_arrow), (x_end, y_arrow),
                                        connectionstyle="arc3,rad=-0.55",  # Stronger upward curve
                                        arrowstyle=patches.ArrowStyle("fancy", head_length=5, head_width=4), 
                                        color="black", linewidth=0.8)

        ax.add_patch(arrow)  # Add the curved arrow to the plot

        # Label the arrow with the time difference
        plt.text((x_start + x_end) / 2, y_arrow + 0.08 * (y_max - y_min),
                 f"{diff} minutes", ha="center", fontsize=12, color="black")

def plot_graph(adj_matrix: Union[torch.Tensor, np.ndarray],
               timestamps: Union[list, np.ndarray, torch.Tensor],
               clusters: Union[list, np.ndarray, torch.Tensor],
               clusters_weights: Union[list, np.ndarray, torch.Tensor],
               all_int2str: Dict[int, str], threshold: float, code_ints: Union[torch.Tensor, np.ndarray],):
    """
    Plots a graph where nodes are positioned based on their timestamps and clusters.
    Filters edges by threshold, removes isolated nodes, adjusts edge thickness, and colors nodes by cluster.
    """
    
    
    # Step 1: Convert data
    adj_matrix, timestamps, clusters, clusters_weights, code_ints = convert_to_numpy(adj_matrix, timestamps, clusters, clusters_weights, code_ints)
    raw_timestamps = c(timestamps)

    # Step 2: Create graph & filter edges/nodes
    G = nx.from_numpy_array(adj_matrix, create_using=nx.DiGraph)
    G, isolated_nodes = filter_graph_edges_and_nodes(G, threshold)

    if len(G.nodes) == 0:
        print("The graph has no connected nodes.")
        return

    # Step 3: Normalize timestamps
    timestamps = [ts for i, ts in enumerate(timestamps) if i not in isolated_nodes]
    if len(set(timestamps)) == 1 and list(set(timestamps))[0] == 0:
        print("The graph has only one unique timestamp.")
        return  
    clusters = [cl for i, cl in enumerate(clusters) if i not in isolated_nodes]
    timestamps = normalize_timestamps(np.array(timestamps))

    # Step 4: Compute node positions
    pos, unique_clusters, unique_timestamps, x_block_spacing = compute_node_positions(G, timestamps, clusters)

    # Step 5: Draw the graph
    draw_graph(G, pos, clusters, unique_clusters, unique_timestamps, x_block_spacing, all_int2str, raw_timestamps, clusters_weights, code_ints)