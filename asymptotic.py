"""
Monte Carlo Verification of Asymptotic Survival Decay
-----------------------------------------------------

Verifies:
    (1/T) (-log10 δ_T) -> κ_v / ln(10)

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
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp
import os
from tqdm import tqdm


# ============================================================
# USER PARAMETERS
# ============================================================

w = 0.05
SEEDS = [0, 8]

gamma = 0.5
N_MC = 500_000
TMAX_CAP = 40000

MIN_SURVIVORS = 100
T_STEP = 1

TAIL_FRACTION = 0.5   # envelope constraint computed on latter fraction only

USE_PARALLEL = True
N_WORKERS = os.cpu_count()


# ============================================================
# GRAPH
# ============================================================

def make_large_two_cliques_two_gates():
    N = 16
    G = {i: [] for i in range(N)}

    A = list(range(0, 7))
    B = list(range(7, 14))
    h1, h2 = 14, 15

    for i in A:
        G[i].extend([j for j in A if j != i])

    for i in B:
        G[i].extend([j for j in B if j != i])

    G[h1].append(h2)
    G[h2].append(h1)

    G[6].append(h1)
    G[h1].append(6)

    G[h2].extend([7, 8])
    G[7].append(h2)
    G[8].append(h2)

    return G


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


def compute_kappa(G, reachable_transient, w):
    min_cut = float("inf")
    for I in reachable_transient:
        b = boundary_size(G, I)
        if b < min_cut:
            min_cut = b
    kappa_v = -np.log1p(-w) * min_cut
    return kappa_v, min_cut


# ============================================================
# MONTE CARLO
# ============================================================

def single_trial(G, seed, w, threshold_size, Tmax, rng):
    infected = set([seed])

    for t in range(Tmax):
        if len(infected) >= threshold_size:
            return t

        new_infected = set()
        for i in infected:
            for j in G[i]:
                if j not in infected and rng.random() < w:
                    new_infected.add(j)

        if new_infected:
            infected |= new_infected

    return np.inf


def worker_chunk(args):
    G, seed, w, threshold_size, Tmax, n_trials, seed_offset = args
    rng = np.random.default_rng(seed_offset)
    taus_local = np.empty(n_trials)

    for i in range(n_trials):
        taus_local[i] = single_trial(G, seed, w, threshold_size, Tmax, rng)

    return taus_local


def generate_tau_samples(G, seed, w, threshold_size, N, Tmax):

    if not USE_PARALLEL:
        rng = np.random.default_rng(123456 + seed)
        taus = np.empty(N)
        for i in tqdm(range(N), desc=f"MC seed={seed}"):
            taus[i] = single_trial(G, seed, w, threshold_size, Tmax, rng)
        return taus

    ctx = mp.get_context("spawn")
    chunk_sizes = np.full(N_WORKERS, N // N_WORKERS)
    chunk_sizes[:N % N_WORKERS] += 1

    args_list = []
    base_seed = 123456 + 1000 * seed
    for i, size in enumerate(chunk_sizes):
        args_list.append((G, seed, w, threshold_size, Tmax, int(size), base_seed + i))

    taus_all = []
    with ProcessPoolExecutor(max_workers=N_WORKERS, mp_context=ctx) as executor:
        futures = [executor.submit(worker_chunk, args) for args in args_list]
        for f in tqdm(as_completed(futures), total=len(futures), desc=f"Parallel MC seed={seed}"):
            taus_all.append(f.result())

    return np.concatenate(taus_all)


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    G = make_large_two_cliques_two_gates()
    N_NODES = len(G)
    threshold_size = int(np.ceil(gamma * N_NODES))

    print("\nGraph: Large two cliques + two gates")
    print("threshold_size =", threshold_size)

    seed_info = {}
    kappas = []

    for seed in SEEDS:
        R_trans = reachable_transient_sets(G, seed, threshold_size)
        kappa_v, min_cut = compute_kappa(G, R_trans, w)
        seed_info[seed] = dict(kappa_v=kappa_v, min_cut=min_cut)
        kappas.append(kappa_v)

        print(f"\nSeed {seed}")
        print("  min_cut =", min_cut)
        print("  κ_v =", kappa_v)

    kappa_min = min(kappas)
    Tmax = min(int(5 * np.log(N_MC) / kappa_min), TMAX_CAP)
    print("\nShared Tmax =", Tmax)

    Ts_grid = np.arange(1, Tmax + 1, T_STEP)

    results = {}
    for seed in SEEDS:
        print(f"\nRunning MC for seed={seed}")
        taus = generate_tau_samples(G, seed, w, threshold_size, N_MC, Tmax)

        deltas = []
        for T in Ts_grid:
            survivors = np.sum(taus > T)
            if survivors < MIN_SURVIVORS:
                break
            deltas.append(survivors / N_MC)

        Ts_valid = Ts_grid[:len(deltas)]
        results[seed] = dict(
            Ts=Ts_valid,
            deltas=np.array(deltas)
        )

    last_common = min(len(results[s]["Ts"]) for s in SEEDS)
    Ts_plot = results[SEEDS[0]]["Ts"][:last_common]

    # ============================================================
    # PLOT 1: -log10 δ_T vs T with UPPER envelope asymptote
    # ============================================================

    plt.figure(figsize=(11, 7))

    for seed in SEEDS:
        deltas = results[seed]["deltas"][:last_common]
        neglog_deltas = -np.log10(deltas)

        kappa_v = seed_info[seed]["kappa_v"]
        slope_theory = kappa_v / np.log(10)  # since -log10 δ ~ (κ/ln 10) T

        n = len(Ts_plot)
        start_idx = int((1 - TAIL_FRACTION) * n)

        Ts_tail = Ts_plot[start_idx:]
        y_tail = neglog_deltas[start_idx:]

        # Choose intercept so the theory line is ABOVE the curve on the tail
        a = np.max(y_tail - slope_theory * Ts_tail)

        line = a + slope_theory * Ts_plot

        plt.plot(
            Ts_plot,
            neglog_deltas,
            '.',
            markersize=3,
            label=f"seed node {seed}, " + r"$-\log_{10}\delta_{T,\gamma}(i)$"
        )
        plt.plot(
            Ts_plot,
            line,
            '--',
            linewidth=2,
            label=f"seed node {seed}, theoretical asymptotic upper bound"
        )

    plt.xlabel(r"$T$")
    plt.ylabel(r"$-\log_{10}\delta_{T,\gamma}(i)$")
    plt.title("Asymptotic Negative Log Survival with Tail Upper Envelope")
    plt.grid(True)
    plt.legend()
    plt.show()

    # ============================================================
    # PLOT 2: (1/T) (-log10 δ_T) vs T with UPPER horizontal bound
    # ============================================================

    plt.figure(figsize=(11, 7))

    for seed in SEEDS:
        deltas = results[seed]["deltas"][:last_common]
        neglog_deltas = -np.log10(deltas)

        normalized_curve = neglog_deltas / Ts_plot

        kappa_v = seed_info[seed]["kappa_v"]
        horiz_theory = kappa_v / np.log(10)

        n = len(Ts_plot)
        start_idx = int((1 - TAIL_FRACTION) * n)

        tail_vals = normalized_curve[start_idx:]

        # Choose a horizontal line ABOVE the empirical tail
        horiz_adjusted = max(horiz_theory, np.max(tail_vals))

        plt.plot(
            Ts_plot,
            normalized_curve,
            '.',
            markersize=3,
            label=f"seed node={seed} " + r"$\frac{-\log_{10}\delta_{T,\gamma}(i)}{T}$"
        )

        plt.hlines(
            horiz_adjusted,
            xmin=Ts_plot[0],
            xmax=Ts_plot[-1],
            linestyles='--',
            linewidth=2,
            label=f"seed={seed} theoretical limit upper bound"
        )

    plt.xlabel(r"$T$")
    plt.ylabel(r"$\frac{-\log_{10}\delta_{T,\gamma}(i)}{T}$")
    plt.title("Convergence of Normalized Negative Log Survival to Theoretical Limit")
    plt.grid(True)
    plt.legend()
    plt.show()

    print("\nDone.")