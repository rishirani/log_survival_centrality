"""
Centrality Comparison on Large Two-Clique Graph
------------------------------------------------

Figure 1:
    Asymmetric Barbell Topology (8 x 4)

Figure 2:
    Log Survival Centrality vs Traditional Centrality Metrics (9 x 6)
"""

import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from tqdm import tqdm
from matplotlib.lines import Line2D


# ============================================================
# PARAMETERS
# ============================================================

w = 0.05
N_MC = 500

T1 = 10
gamma1 = 0.2

T2 = 100
gamma2 = 0.8

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
# SI MONTE CARLO
# ============================================================

def single_trial(G, seed, w, threshold_size, Tmax, rng):
    infected = {seed}

    for _ in range(Tmax):

        if len(infected) >= threshold_size:
            return False

        new_inf = set()

        for i in infected:
            for j in G.successors(i):
                if j not in infected and rng.random() < w:
                    new_inf.add(j)

        infected |= new_inf

    return True


def estimate_delta(G, T, gamma):

    n = G.number_of_nodes()
    threshold = int(np.ceil(gamma * n))
    deltas = []

    rng = np.random.default_rng(1234)

    print(f"\nEstimating δ(T={T}, γ={gamma})")

    for seed in tqdm(G.nodes(), desc="Seeds"):
        survivors = 0
        for _ in tqdm(range(N_MC), leave=False):
            if single_trial(G, seed, w, threshold, T, rng):
                survivors += 1
        deltas.append(survivors / N_MC)

    return np.array(deltas)


# ============================================================
# NORMALIZATION
# ============================================================

def normalize_to_unit(x):
    xmin = np.min(x)
    xmax = np.max(x)
    if xmax - xmin < 1e-12:
        return np.zeros_like(x)
    return (x - xmin) / (xmax - xmin)


# ============================================================
# MAIN
# ============================================================

G = make_large_two_cliques_two_gates()

pos = nx.kamada_kawai_layout(G)
rot_pos = rotate_positions(pos, ROTATION_DEGREES)

clique_small = set(range(0, 7))
clique_large = set(range(7, 14))


# Classical centralities
deg = np.array([G.out_degree(i) for i in G.nodes()])
bet = np.array(list(nx.betweenness_centrality(G).values()))
clo = np.array(list(nx.closeness_centrality(G).values()))
eig = np.array(list(nx.eigenvector_centrality_numpy(G).values()))

# Dynamical centralities
delta1 = estimate_delta(G, T1, gamma1)
delta2 = estimate_delta(G, T2, gamma2)

eps = 1e-12
inf1 = -np.log(delta1 + eps)
inf2 = -np.log(delta2 + eps)

# Normalize
deg = normalize_to_unit(deg)
bet = normalize_to_unit(bet)
clo = normalize_to_unit(clo)
eig = normalize_to_unit(eig)
inf1 = normalize_to_unit(inf1)
inf2 = normalize_to_unit(inf2)


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

ax1.legend(handles=legend_elements, loc='upper center')
ax1.set_aspect('equal')
ax1.axis("off")
ax1.set_title("Asymmetric Barbell Topology", fontsize=14)

plt.tight_layout()
plt.savefig("./topology.png", bbox_inches='tight',dpi=400)
plt.show()


# ============================================================
# FIGURE 2
# ============================================================

fig2, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.flatten()

def draw_heat_panel(ax, values, title):

    nodes = nx.draw_networkx_nodes(
        G, pos, ax=ax,
        node_color=values,
        cmap='coolwarm',
        vmin=0,
        vmax=1,
        node_size=350
    )

    nx.draw_networkx_edges(G, pos, ax=ax, arrows=False)
    nx.draw_networkx_labels(G, pos, ax=ax)

    ax.set_title(title, fontsize=10)
    ax.set_axis_off()

    fig2.colorbar(nodes, ax=ax, fraction=0.04, pad=0.03)


draw_heat_panel(axes[0], deg, "Degree")
draw_heat_panel(axes[1], bet, "Betweenness")
draw_heat_panel(axes[2], clo, "Closeness")
draw_heat_panel(axes[3], eig, "Eigenvector")
draw_heat_panel(axes[4], inf1, r"Log Survival Centrality $(T=10,\gamma=0.2)$")
draw_heat_panel(axes[5], inf2, r"Log Survival Centrality $(T=100,\gamma=0.8)$")

fig2.suptitle(
    "Log Survival Centrality vs Traditional Centrality Metrics",
    fontsize=14,
    y=0.98
)

fig2.subplots_adjust(
    top=0.90,
    hspace=0.35,
    wspace=0.30
)
plt.savefig("./centrality_comparison.png", bbox_inches='tight',dpi=400)
plt.show()