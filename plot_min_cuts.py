
"""
Plotting Min-Cut Edges
-----------------------------------------------------
Graph:
    Large two cliques + two gates (N=16)

Threshold:
    gamma = 0.5  -> threshold_size = 8

Seeds:
    seed 0  -> cut 1
    seed 8  -> cut 2
"""

import numpy as np
import matplotlib.pyplot as plt
from collections import deque
from tqdm import tqdm
import networkx as nx
from matplotlib.lines import Line2D


# ============================================================
# USER PARAMETERS
# ============================================================

w = 0.05
SEEDS = [0, 8]

gamma = 0.5

ROTATION_DEGREES = 50

# ============================================================
# GRAPH CONSTRUCTION
# ============================================================

def make_large_two_cliques_two_gates():
    G = nx.DiGraph()

    A = list(range(0, 7))
    B = list(range(7, 14))
    h1, h2 = 14, 15

    for i in A:
        for j in A:
            if i != j:
                G.add_edge(i, j)

    for i in B:
        for j in B:
            if i != j:
                G.add_edge(i, j)

    G.add_edge(h1, h2)
    G.add_edge(h2, h1)

    G.add_edge(6, h1)
    G.add_edge(h1, 6)

    G.add_edge(h2, 7)
    G.add_edge(h2, 8)
    G.add_edge(7, h2)
    G.add_edge(8, h2)

    return G


# ============================================================
# ROTATION (TOPOLOGY ONLY)
# ============================================================

def rotate_positions(pos, degrees_clockwise):

    coords = np.array([pos[n] for n in pos.keys()])
    coords -= coords.mean(axis=0)

    theta = -np.deg2rad(degrees_clockwise)

    R = np.array([
        [np.cos(theta), -np.sin(theta)],
        [np.sin(theta),  np.cos(theta)]
    ])

    rotated = coords @ R.T

    return {node: rotated[i] for i, node in enumerate(pos.keys())}



# ============================================================
# REACHABILITY + CUT
# ============================================================

def reachable_transient_sets(G, seed, threshold_size):
    visited = set()
    q = deque()
    start = frozenset([seed])
    visited.add(start)
    q.append(start)

    while q:
        I = q.popleft()

        if len(I) >= threshold_size:
            continue

        frontier = set()
        for i in I:
            frontier.update(G[i])
        frontier -= set(I)

        for j in frontier:
            new_set = frozenset(set(I) | {j})
            if len(new_set) >= threshold_size:
                continue
            if new_set not in visited:
                visited.add(new_set)
                q.append(new_set)

    return visited


def boundary_size(G, I):
    b = 0
    Iset = set(I)
    for k in Iset:
        for j in G[k]:
            if j not in Iset:
                b += 1
    return b


def compute_kappa(G, reachable_transient, w, return_set):
    min_cut = float("inf")
    I_min = None
    for I in reachable_transient:
        b = boundary_size(G, I)
        if b < min_cut:
            min_cut = b
            I_min = I
    kappa_v = -np.log1p(-w) * min_cut
    if return_set:
      return kappa_v, min_cut, I_min
    return kappa_v, min_cut


def plot_cut_lines(ax, G, pos, I_min, color, single_line=False):
    """
    Plots boundary edges for the set I_min.
    If single_line is True, projects a single separating line based on cluster centroids.
    """
    cut_edges = [(u, v) for u in I_min for v in G[u] if v not in I_min]
    if not cut_edges:
        return

    if single_line:
        # Calculate centroids of the two sets
        I_min_nodes = list(I_min)
        other_nodes = [n for n in G.nodes() if n not in I_min]

        c1 = np.mean([pos[n] for n in I_min_nodes], axis=0)
        c2 = np.mean([pos[n] for n in other_nodes], axis=0)

        # Calculate mean midpoint of all cut edges
        midpoints = []
        for u, v in cut_edges:
            midpoints.append((np.array(pos[u]) + np.array(pos[v])) / 2.0)
        center_point = np.mean(midpoints, axis=0)

        # Calculate perpendicular vector to the centroid axis
        vec = c2 - c1
        norm = np.linalg.norm(vec)
        if norm != 0:
            ortho = np.array([-vec[1], vec[0]]) / norm

            # Define line length (heuristically scaled by graph size)
            pos_array = np.array(list(pos.values()))
            graph_span = np.max(pos_array) - np.min(pos_array)
            line_half_len = graph_span * 0.1

            start = center_point - line_half_len * ortho
            end = center_point + line_half_len * ortho

            ax.plot([start[0], end[0]], [start[1], end[1]],
                    color=color, linestyle='--', linewidth=2, zorder=5)
    else:
        # Standard orthogonal line per edge
        for u, v in cut_edges:
            p1, p2 = np.array(pos[u]), np.array(pos[v])
            mid = (p1 + p2) / 2.0
            vec = p2 - p1
            norm = np.linalg.norm(vec)

            if norm == 0:
                continue

            ortho = np.array([-vec[1], vec[0]]) / norm
            cut_half_len = norm * 0.4

            start = mid - cut_half_len * ortho
            end = mid + cut_half_len * ortho

            ax.plot([start[0], end[0]], [start[1], end[1]],
                    color=color, linestyle='--', linewidth=2, zorder=5)

# ============================================================
# MAIN
# ============================================================

G = make_large_two_cliques_two_gates()

pos = nx.kamada_kawai_layout(G)
rot_pos = rotate_positions(pos, ROTATION_DEGREES)

clique_small = set(range(0, 7))
clique_large = set(range(7, 14))
N_NODES = len(G)
threshold_size = int(np.ceil(gamma * N_NODES))

print("\nGraph: Large two cliques + two gates")
print("threshold_size =", threshold_size)

seed_info = {}
kappas = []

return_set = True

for seed in SEEDS:
    R_trans = reachable_transient_sets(G, seed, threshold_size)
    if return_set:
        kappa_v, min_cut, nodes = compute_kappa(G, R_trans, w, return_set)
    else:
        kappa_v, min_cut = compute_kappa(G, R_trans, w, return_set)
        nodes = None

    # Modified: Store nodes in the dictionary
    seed_info[seed] = dict(kappa_v=kappa_v, min_cut=min_cut, nodes=nodes)
    kappas.append(kappa_v)

    print(f"\nSeed {seed}")
    print("  min_cut =", min_cut)
    print("  κ_v =", kappa_v)
    print("  nodes =", nodes)

# ============================================================
# FIGURE 1
# ============================================================

fig1 = plt.figure(figsize=(8, 4))
ax1 = fig1.add_subplot(111)

node_colors = []
for node in G.nodes():
    if node in clique_small:
        node_colors.append("tab:blue")
    elif node in clique_large:
        node_colors.append("tab:orange")
    else:
        node_colors.append("tab:red")

nx.draw_networkx_nodes(G, rot_pos, ax=ax1,
                       node_color=node_colors, node_size=800)
nx.draw_networkx_edges(G, rot_pos, ax=ax1, arrows=False)
nx.draw_networkx_labels(G, rot_pos, ax=ax1)

legend_elements = [
    Line2D([0], [0], marker='o', color='w', label='Small Clique',
           markerfacecolor='tab:blue', markersize=10),
    Line2D([0], [0], marker='o', color='w', label='Large Clique',
           markerfacecolor='tab:orange', markersize=10),
    Line2D([0], [0], marker='o', color='w', label='Bridge Nodes',
           markerfacecolor='tab:red', markersize=10),
]

seed_cut_colors = {0: "tab:green", 8: "tab:purple"}

for seed in SEEDS:
    I_min = seed_info[seed]["nodes"]
    if I_min is not None:
        # Apply the single line heuristic exclusively for seed 8
        use_single_line = (seed == 8)

        plot_cut_lines(ax1, G, rot_pos, I_min, seed_cut_colors[seed], single_line=use_single_line)

        legend_elements.append(
            Line2D([0], [0], color=seed_cut_colors[seed], linestyle='--',
                   linewidth=2, label=f'Constrained Min Cut (Seed {seed})')
        )

ax1.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=2)
ax1.set_aspect('equal')
ax1.axis("off")
ax1.set_title("Asymmetric Barbell Topology with Min Cuts", fontsize=14)

plt.savefig("./topology.png", bbox_inches='tight',dpi=400)

plt.tight_layout()
plt.show()