import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import KDTree
import networkx as nx


def calculate_R(n, r):
    return np.sqrt((r * np.log(n)) / (np.pi * n))


def is_connected(n, r):
    points = np.random.rand(n, 2)

    radius = calculate_R(n, r)

    tree = KDTree(points)
    pairs = tree.query_pairs(radius)

    G = nx.Graph()
    G.add_nodes_from(range(n))
    G.add_edges_from(pairs)

    return nx.is_connected(G)

# --- Experiment Parameters ---
n_values = [200, 500, 1000]
r_values = np.linspace(0.1, 2.5, 20)  # Sweeping r from 0.1 to 2.5
iterations = 50  # Number of trials per (n, r) pair

results = {n: [] for n in n_values}

# --- Run Empirical Regime ---
for n in n_values:
    print(f"Running simulation for n={n}...")
    for r in r_values:
        success_count = sum(is_connected(n, r) for _ in range(iterations))
        prob = success_count / iterations
        results[n].append(prob)

# --- Plot the Results ---
plt.figure(figsize=(10, 6))
for n in n_values:
    plt.plot(r_values, results[n], marker='o', label=f'n={n}')

plt.axvline(x=1.0, color='r', linestyle='--', label='Theoretical r_c ≈ 1.0')
plt.title('Probability of Connectivity in Random Geometric Graphs')
plt.xlabel('Constant r')
plt.ylabel('Probability of Full Connectivity')
plt.legend()
plt.grid(True)
plt.show()