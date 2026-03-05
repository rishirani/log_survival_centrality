# log_survival_centrality

This repository contains simulation code used in the CDC 2026 conference paper

"A Dynamical Centrality Based on Macro-Infection Hitting Times in SI Epidemic Networks"

by Sunreet Khanna, Rishi Rani, Behrouz Touri, and Massimo Franceschetti.

The purpose of this code is to numerically verify the asymptotic decay of survival probabilities in SI epidemic processes and illustrate the behavior of the proposed centrality measure.

---

# What the Code Does

The code simulates a discrete-time Susceptible–Infected (SI) epidemic process on a network.

Each infected node independently attempts to infect its neighbors with probability

$$
A_{ij}
$$

at every time step.

Once a node becomes infected it remains infected permanently.

For a given macro-infection threshold

$$
\gamma
$$

and time horizon

$$
T
$$

we study the survival probability

$$
\delta_T(v) = \mathbb{P}_v(\tau_\gamma > T)
$$

which represents the probability that the epidemic has not yet reached the macro-infected set by time $T$ when initialized from seed node $v$.

The simulations focus on the asymptotic decay rate

$$
\frac{1}{T}\log \delta_T(v)
$$

which theoretically converges to

$$
-\kappa_v
$$

where $\kappa_v$ is determined by a reachability-constrained min-cut.

---

# Scripts

## asymptotic.py

Monte Carlo simulation used to verify the asymptotic decay

$$
\frac{1}{T}\log \delta_T(v) \rightarrow -\kappa_v
$$

The script

- simulates independent SI cascades
- estimates survival probabilities
- plots the log survival curve
- compares empirical decay rates with theoretical slopes

Parallel execution is used to run large numbers of Monte Carlo trials efficiently.

---

## centrality_comp.py

Utility script for comparing survival probabilities for different seed nodes.

This allows visualization of how nodes with similar degree can produce different epidemic delay behavior depending on downstream connectivity.

---

# Installation

Install dependencies

    pip install -r requirements.txt

---

# Running the Experiments

Example

    python asymptotic.py

This generates plots of the empirical survival probability decay.

---

# Notes

The code is intended for simulation and verification of theoretical results in the CDC 2026 paper and is not intended to be a general-purpose epidemic modeling package.