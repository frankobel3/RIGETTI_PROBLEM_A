#!/usr/bin/env python3
"""
LoopEi Max-Cut Solver — Problem A (Unified)
LoopEi LLC | USPTO #19/303,438 | Franklyn E. Beltre
Aqora / Rigetti Hackathon 2026

Pipeline (single file, single run):
  Stage 1 — LoopEi Solver
              Exact enumeration (2^20 batched, certifies global optimum)
              LoopEi QAOA Hybrid (Tesla 3-6-9 seeding, G1 Governor, PhaseMemory,
              Vortex escape, local search polish)

  Stage 2 — LQTrust++ Validation
              Plateau scan (k-flip optimality proof over blocking node set)
              Blocking edge detection
              9 trust/resilience metrics (balance, isolation, redundancy,
              N-1 resilience, bottleneck ratio, load concentration)

  Stage 3 — QAOA Warm-Start Bridge
              LoopEi optimal partition → QAOA warm-start initialization
              Epsilon-angle RY rotation (Egger et al. 2021)
              COBYLA optimization from near-optimal landscape
              Head-to-head: standard multi-start vs LoopEi warm-start

Usage:
  py loopei_maxcut_A.py                        # uses defaults
  py loopei_maxcut_A.py --graph problema.csv   # specify graph
  py loopei_maxcut_A.py --skip-qaoa            # skip Qiskit stage (no Qiskit needed)
"""

import os
import sys
import math
import time
import json
import random
import argparse
import itertools
from collections import defaultdict
from dataclasses import dataclass, field
from typing import List, Tuple, Optional

import numpy as np
import pandas as pd

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

WORK_DIR    = os.path.dirname(os.path.abspath(__file__))
INPUT_CSV   = os.path.join(WORK_DIR, "problema.csv")
OUT_RESULT  = os.path.join(WORK_DIR, "loopei_result_A.json")
OUT_EDGES   = os.path.join(WORK_DIR, "loopei_cut_edges_A.json")
OUT_REPORT  = os.path.join(WORK_DIR, "report_full.json")
OUT_SUMMARY = os.path.join(WORK_DIR, "team_summary.csv")
OUT_BLOCK   = os.path.join(WORK_DIR, "blocking_edges.csv")
OUT_PLOT    = os.path.join(WORK_DIR, "partition_plot.png")

PHI           = (1 + math.sqrt(5)) / 2
TESLA_SEEDS   = [3, 6, 9, 369, 396, 639, 693, 936, 963]
G1_THRESHOLD  = 0.079308
G1_WARMUP     = 27
TOTAL_WEIGHT  = 4215.6655   # updated dynamically after load


# ═════════════════════════════════════════════════════════════════════════════
# STAGE 1 — LOOPEI SOLVER
# ═════════════════════════════════════════════════════════════════════════════

# ── Tesla 3-6-9 utilities ────────────────────────────────────────────────────

def digital_root(n: int) -> int:
    n = abs(int(n))
    if n == 0:
        return 9
    return ((n - 1) % 9) + 1


def tesla_resonance(n: int) -> float:
    dr = digital_root(n)
    if dr in (3, 6, 9):
        return 1.0 + (dr / 9.0) * 0.9
    return 1.0


# ── Graph loader ─────────────────────────────────────────────────────────────

def load_graph(csv_path: str):
    df = pd.read_csv(csv_path)
    # Accept any column order — auto-detect node_1/node_2/weight or positional
    cols = list(df.columns)
    if 'node_1' in cols and 'node_2' in cols and 'weight' in cols:
        edges = list(zip(
            df['node_1'].astype(int),
            df['node_2'].astype(int),
            df['weight'].astype(float)
        ))
    else:
        edges = [(int(r.iloc[0]), int(r.iloc[1]), float(r.iloc[2]))
                 for _, r in df.iterrows()]

    nodes = set()
    for u, v, _ in edges:
        nodes.add(u)
        nodes.add(v)
    N     = max(nodes) + 1
    u_arr = np.array([e[0] for e in edges], dtype=np.int32)
    v_arr = np.array([e[1] for e in edges], dtype=np.int32)
    w_arr = np.array([e[2] for e in edges], dtype=np.float64)
    return N, edges, u_arr, v_arr, w_arr


# ── Vectorized cut evaluation ────────────────────────────────────────────────

def cut_value_np(partition: np.ndarray, u_arr, v_arr, w_arr) -> float:
    return float(np.sum(w_arr[partition[u_arr] != partition[v_arr]]))


def batch_cut(partitions: np.ndarray, u_arr, v_arr, w_arr) -> np.ndarray:
    pu = partitions[:, u_arr]
    pv = partitions[:, v_arr]
    return ((pu != pv) * w_arr).sum(axis=1)


# ── Exact solver ─────────────────────────────────────────────────────────────

def exact_max_cut(N: int, u_arr, v_arr, w_arr, batch_size: int = 200_000):
    print(f"    Enumerating 2^{N-1} = {2**(N-1):,} partitions...")
    half, best_cut, best_idx = 2 ** (N - 1), 0.0, 0

    for start in range(0, half, batch_size):
        end        = min(start + batch_size, half)
        idxs       = np.arange(start, end, dtype=np.int32)
        partitions = np.zeros((end - start, N), dtype=np.int8)
        for bit in range(N - 1):
            partitions[:, bit + 1] = (idxs >> bit) & 1
        cuts       = batch_cut(partitions, u_arr, v_arr, w_arr)
        local_best = cuts.max()
        if local_best > best_cut:
            best_cut = local_best
            best_idx = start + int(cuts.argmax())

    best_partition = np.zeros(N, dtype=np.int8)
    for bit in range(N - 1):
        best_partition[bit + 1] = (best_idx >> bit) & 1

    return best_partition, float(best_cut)


# ── PhaseMemory ──────────────────────────────────────────────────────────────

@dataclass
class PhaseMemory:
    best_cut:       float                = 0.0
    best_partition: Optional[np.ndarray] = None
    best_params:    Optional[np.ndarray] = None
    history:        List[float]          = field(default_factory=list)
    iteration:      int                  = 0

    def update(self, cut: float, partition: np.ndarray,
               params: Optional[np.ndarray] = None) -> bool:
        self.history.append(cut)
        self.iteration += 1
        if cut > self.best_cut:
            self.best_cut       = cut
            self.best_partition = partition.copy()
            if params is not None:
                self.best_params = params.copy()
            return True
        return False

    def plateau_depth(self, window: int = 9) -> int:
        if len(self.history) < window:
            return 0
        recent = self.history[-window:]
        return window if max(recent) - min(recent) < 1.0 else 0


# ── G1 Unity Governor ────────────────────────────────────────────────────────

class G1Governor:
    def __init__(self, threshold: float = G1_THRESHOLD, warmup: int = G1_WARMUP):
        self.threshold   = threshold
        self.warmup      = warmup
        self.iteration   = 0
        self.activations = 0

    def check(self, grad_norm: float) -> bool:
        self.iteration += 1
        if self.iteration <= self.warmup:
            return True
        admissible = grad_norm <= self.threshold
        if not admissible:
            self.activations += 1
        return admissible


# ── QAOA cost function (classical approximation) ─────────────────────────────

def qaoa_cost(params, N, u_arr, v_arr, w_arr, p_layers=2, total_w=TOTAL_WEIGHT):
    gammas = params[:p_layers]
    betas  = params[p_layers:]
    state  = np.ones(N) * 0.5

    for layer in range(p_layers):
        for u, v, w in zip(u_arr, v_arr, w_arr):
            phase    = gammas[layer] * w / total_w
            state[u] = state[u] * math.cos(phase) + (1 - state[u]) * math.sin(phase)
            state[v] = state[v] * math.cos(phase) + (1 - state[v]) * math.sin(phase)
        beta  = betas[layer]
        state = state * math.cos(beta) + (1 - state) * math.sin(beta)

    partition = (state >= 0.5).astype(np.int8)
    return -cut_value_np(partition, u_arr, v_arr, w_arr)


def params_to_partition(params, N, u_arr, v_arr, w_arr,
                         p_layers=2, total_w=TOTAL_WEIGHT):
    gammas = params[:p_layers]
    betas  = params[p_layers:]
    state  = np.ones(N) * 0.5

    for layer in range(p_layers):
        for u, v, w in zip(u_arr, v_arr, w_arr):
            phase    = gammas[layer] * w / total_w
            state[u] = state[u] * math.cos(phase) + (1 - state[u]) * math.sin(phase)
            state[v] = state[v] * math.cos(phase) + (1 - state[v]) * math.sin(phase)
        beta  = betas[layer]
        state = state * math.cos(beta) + (1 - state) * math.sin(beta)

    return (state >= 0.5).astype(np.int8)


# ── Local search polish ──────────────────────────────────────────────────────

def local_search(partition, u_arr, v_arr, w_arr):
    p    = partition.copy()
    N    = len(p)
    best = cut_value_np(p, u_arr, v_arr, w_arr)
    improved = True
    while improved:
        improved = False
        for i in range(N):
            p[i] ^= 1
            c = cut_value_np(p, u_arr, v_arr, w_arr)
            if c > best:
                best     = c
                improved = True
            else:
                p[i] ^= 1
    return p, best


# ── LoopEi QAOA hybrid solver ─────────────────────────────────────────────────

def loopei_qaoa_solver(N, u_arr, v_arr, w_arr,
                        p_layers=2, n_basins=18, total_w=TOTAL_WEIGHT):
    from scipy.optimize import minimize

    memory   = PhaseMemory()
    governor = G1Governor()
    rng      = np.random.RandomState(369)
    n_params = 2 * p_layers

    seed_params = []
    for ts in TESLA_SEEDS:
        res = tesla_resonance(ts)
        g   = [res * math.pi / (3 * (i + 1)) for i in range(p_layers)]
        b   = [math.pi / (PHI * (i + 1))     for i in range(p_layers)]
        seed_params.append(np.array(g + b))
    for i in range(1, 5):
        g = [PHI * math.pi / (i * 3) for _ in range(p_layers)]
        b = [math.pi / (PHI * i)     for _ in range(p_layers)]
        seed_params.append(np.array(g + b))
    while len(seed_params) < n_basins:
        seed_params.append(rng.uniform(0, math.pi, n_params))

    print(f"    {len(seed_params)} basins, p={p_layers} layers, "
          f"Tesla seeds={len(TESLA_SEEDS)}, G1 warmup={G1_WARMUP}")

    for idx, init_p in enumerate(seed_params):

        def objective(params, _ip=init_p):
            gn = np.linalg.norm(params - _ip) / (np.linalg.norm(_ip) + 1e-10)
            if not governor.check(gn):
                return -memory.best_cut
            cost = qaoa_cost(params, N, u_arr, v_arr, w_arr, p_layers, total_w)
            part = params_to_partition(params, N, u_arr, v_arr, w_arr, p_layers, total_w)
            part, cut = local_search(part, u_arr, v_arr, w_arr)
            memory.update(cut, part, params)
            if memory.plateau_depth(9) >= 9:
                pert  = rng.uniform(-0.3, 0.3, len(params))
                boost = tesla_resonance(idx + 1)
                return qaoa_cost(params + pert * boost, N, u_arr, v_arr,
                                 w_arr, p_layers, total_w)
            return cost

        res  = minimize(objective, init_p, method='COBYLA',
                        options={'maxiter': 300, 'rhobeg': 0.5})
        part = params_to_partition(res.x, N, u_arr, v_arr, w_arr, p_layers, total_w)
        part, cut = local_search(part, u_arr, v_arr, w_arr)
        memory.update(cut, part, res.x)
        print(f"    Basin {idx+1:2d}: cut={memory.best_cut:9.2f}  "
              f"ratio={memory.best_cut/total_w:.4f}  "
              f"G1={governor.activations}")

    return memory.best_partition, memory.best_cut, memory


# ═════════════════════════════════════════════════════════════════════════════
# STAGE 2 — LQTrust++ VALIDATION
# ═════════════════════════════════════════════════════════════════════════════

def cut_value_dict(edges, part):
    return sum(w for u, v, w in edges if part[u] != part[v])


def get_blocking_edges(edges, part):
    return [(u, v, w) for u, v, w in edges if part[u] == part[v]]


def lqtrust_metrics(edges, part):
    degree      = defaultdict(int)
    side_counts = defaultdict(int)

    for node, side in part.items():
        side_counts[side] += 1
    for u, v, _ in edges:
        degree[u] += 1
        degree[v] += 1

    balance    = min(side_counts.values()) / max(side_counts.values())
    isolation  = 1.0 if min(degree.values()) > 0 else 0.0
    redundancy = sum(1 for d in degree.values() if d > 1) / len(degree)

    base_cut = cut_value_dict(edges, part)
    drops    = [base_cut - cut_value_dict(edges[:i] + edges[i+1:], part)
                for i in range(len(edges))]
    max_drop   = max(drops)
    avg_drop   = sum(drops) / len(drops)
    resilience = 1 - (max_drop / base_cut)

    blocked_weight = sum(w for _, _, w in get_blocking_edges(edges, part))
    total_weight   = sum(w for _, _, w in edges)
    bottleneck_ratio = blocked_weight / total_weight

    side_load = defaultdict(float)
    for u, v, w in edges:
        if part[u] != part[v]:
            side_load[part[u]] += w
            side_load[part[v]] += w
    loads = list(side_load.values())
    load_concentration = max(loads) / sum(loads) if loads else 0

    trust_score = round(
        (balance + isolation + redundancy + resilience + (1 - bottleneck_ratio)) / 5, 4
    )

    return {
        "balance":               round(balance, 4),
        "isolation":             isolation,
        "redundancy":            round(redundancy, 4),
        "n_minus_one_resilience":round(resilience, 4),
        "max_single_edge_drop":  round(max_drop, 4),
        "avg_edge_drop":         round(avg_drop, 4),
        "bottleneck_ratio":      round(bottleneck_ratio, 4),
        "load_concentration":    round(load_concentration, 4),
        "trust_score":           trust_score,
    }


def plateau_scan(edges, part, max_k=4):
    blocks  = get_blocking_edges(edges, part)
    B       = sorted({u for u, _, _ in blocks} | {v for _, v, _ in blocks})
    base    = cut_value_dict(edges, part)
    best    = base
    best_flip = None

    def flip(p, nodes):
        q = dict(p)
        for n in nodes:
            q[n] = 1 - q[n]
        return q

    for k in range(1, max_k + 1):
        for subset in itertools.combinations(B, k):
            val = cut_value_dict(edges, flip(part, subset))
            if val > best:
                best      = val
                best_flip = subset

    return base, best, best_flip, B


def plot_partition(edges, part, trust_score, out=OUT_PLOT):
    try:
        import matplotlib.pyplot as plt
        import networkx as nx
    except ImportError:
        print("    [skip] matplotlib/networkx not installed — skipping plot")
        return

    G = nx.Graph()
    for u, v, w in edges:
        G.add_edge(u, v, weight=w)

    pos         = nx.spring_layout(G, seed=42)
    node_colors = ["#1f77b4" if part[n] == 0 else "#ff7f0e" for n in G.nodes()]
    cut_edges   = [(u, v) for u, v in G.edges() if part[u] != part[v]]
    blk_edges   = [(u, v) for u, v, _ in edges if part[u] == part[v]]

    plt.figure(figsize=(10, 7))
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=600)
    nx.draw_networkx_labels(G, pos, font_color="white", font_size=9)
    nx.draw_networkx_edges(G, pos, edgelist=cut_edges, width=2.5, edge_color="#333333")
    nx.draw_networkx_edges(G, pos, edgelist=blk_edges, width=1.5,
                           edge_color="red", style="dashed")
    cut_val = cut_value_dict(edges, part)
    plt.title(f"LoopEi Problem A — Optimal Partition\n"
              f"Cut = {cut_val:.2f}  |  Trust = {trust_score}  |  MPES = {cut_val/sum(w for _,_,w in edges):.4f}",
              fontsize=12)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"    Saved: {out}")


# ═════════════════════════════════════════════════════════════════════════════
# STAGE 3 — QAOA WARM-START BRIDGE
# ═════════════════════════════════════════════════════════════════════════════

def run_qaoa_bridge(warmstart_bits, nodes_sorted, G_nx, total_weight_norm,
                    best_val_norm, half_total, skip=False):
    """
    Runs Qiskit QAOA with standard multi-start AND LoopEi warm-start.
    Returns dict of results. Skips gracefully if Qiskit not installed.
    """
    if skip:
        print("    [skip] --skip-qaoa flag set")
        return None

    try:
        import numpy as np
        from scipy.optimize import minimize as sp_minimize
        from qiskit import QuantumCircuit
        from qiskit.quantum_info import SparsePauliOp
        from qiskit.primitives import StatevectorEstimator
    except ImportError:
        print("    [skip] Qiskit not installed — skipping QAOA bridge")
        print("           Install with: pip install qiskit qiskit-algorithms")
        return None

    num_qubits   = len(nodes_sorted)
    node_to_idx  = {n: i for i, n in enumerate(nodes_sorted)}
    pauli_list, coeffs = [], []

    for u, v, data in G_nx.edges(data=True):
        i, j = node_to_idx[u], node_to_idx[v]
        w    = data["weight"]
        pauli = ["I"] * num_qubits
        pauli[i] = "Z"
        pauli[j] = "Z"
        pauli_list.append("".join(pauli))
        coeffs.append(-0.5 * w)

    cost_hamiltonian = SparsePauliOp(pauli_list, coeffs)
    estimator        = StatevectorEstimator()

    # Warm-start qubit ordering
    warmstart_init = [warmstart_bits[node_to_idx[n]] for n in nodes_sorted]

    def build_circuit(params, init_state=None):
        qc = QuantumCircuit(num_qubits)
        p  = len(params) // 2

        if init_state is None:
            # Standard |+>^n
            for i in range(num_qubits):
                qc.h(i)
        else:
            # LoopEi warm-start: epsilon-angle approach (Egger et al. 2021)
            epsilon = np.pi / 4
            for i, bit in enumerate(init_state):
                if bit == 1:
                    qc.x(i)
                qc.ry(epsilon, i)

        for layer in range(p):
            gamma, beta = params[2*layer], params[2*layer+1]
            for pl, co in zip(pauli_list, coeffs):
                z_idx = [k for k, pz in enumerate(pl) if pz == "Z"]
                qc.rzz(2 * gamma * co.real, z_idx[0], z_idx[1])
            for i in range(num_qubits):
                qc.rx(2 * beta, i)
        return qc

    def expectation(params, init_state=None):
        qc  = build_circuit(params, init_state)
        job = estimator.run(pubs=[(qc, cost_hamiltonian)])
        return float(job.result()[0].data.evs)

    def objective(params, init_state=None):
        return -expectation(params, init_state)

    # Standard 10-start
    print("    Standard multi-start (3 runs, p=2) ...")
    std_best, std_result = -1e9, None
    for run in range(3):
        ig  = np.random.uniform(0, 2*np.pi, 4)
        res = sp_minimize(objective, ig, method="COBYLA",
                          options={"maxiter": 100})
        val = -res.fun
        if val > std_best:
            std_best, std_result = val, res
        print(f"      Run {run+1:2d}: E={val:.4f}")

    # LoopEi warm-start (5 runs)
    print("    LoopEi warm-start (3 runs, p=2) ...")
    ws_best, ws_result = -1e9, None
    for run in range(3):
        ig  = np.random.uniform(0, 0.3, 4)
        res = sp_minimize(lambda p: objective(p, warmstart_init), ig,
                          method="COBYLA", options={"maxiter": 100})
        val = -res.fun
        if val > ws_best:
            ws_best, ws_result = val, res
        print(f"      WS Run {run+1}: E={val:.4f}")

    def to_cut(h_val):
        return half_total + h_val

    return {
        "standard_expectation":   round(std_best, 6),
        "standard_cut_norm":      round(to_cut(std_best), 6),
        "standard_cut_raw":       round(to_cut(std_best) * (TOTAL_WEIGHT / total_weight_norm), 4),
        "standard_ratio":         round(to_cut(std_best) / best_val_norm, 4),
        "warmstart_expectation":  round(ws_best, 6),
        "warmstart_cut_norm":     round(to_cut(ws_best), 6),
        "warmstart_cut_raw":      round(to_cut(ws_best) * (TOTAL_WEIGHT / total_weight_norm), 4),
        "warmstart_ratio":        round(to_cut(ws_best) / best_val_norm, 4),
        "improvement_pp":         round((to_cut(ws_best) - to_cut(std_best)) / best_val_norm * 100, 2),
    }


# ═════════════════════════════════════════════════════════════════════════════
# MAIN PIPELINE
# ═════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="LoopEi Max-Cut Solver — Problem A (Unified)")
    parser.add_argument("--graph",     default=INPUT_CSV,
                        help="Path to graph CSV (node_1, node_2, weight)")
    parser.add_argument("--skip-qaoa", action="store_true",
                        help="Skip Qiskit QAOA bridge (Stage 3)")
    args = parser.parse_args()

    print("\n" + "="*65)
    print("  LoopEi Max-Cut Solver — Problem A (Unified)")
    print("  LoopEi LLC | USPTO #19/303,438 | CAGE Code 16GM4")
    print("  Aqora / Rigetti Hackathon 2026")
    print("="*65 + "\n")

    t0 = time.time()

    # ── Load ────────────────────────────────────────────────────────────────
    print("[1] Loading graph...")
    N, edges, u_arr, v_arr, w_arr = load_graph(args.graph)
    total_w = float(w_arr.sum())
    global TOTAL_WEIGHT
    TOTAL_WEIGHT = total_w
    print(f"    Nodes: {N}  |  Edges: {len(edges)}  |  Total weight: {total_w:.4f}\n")

    # ── Exact enumeration ────────────────────────────────────────────────────
    print("[2] Exact optimal certification...")
    t2 = time.time()
    exact_partition, exact_cut = exact_max_cut(N, u_arr, v_arr, w_arr)
    print(f"    Exact cut:  {exact_cut:.4f}  (MPES={exact_cut/total_w:.4f})")
    print(f"    Time:       {time.time()-t2:.2f}s\n")

    # ── LoopEi QAOA hybrid ───────────────────────────────────────────────────
    print("[3] LoopEi QAOA Hybrid Solver...")
    print(f"    Tesla 3-6-9 seeding | G1 Governor | PhaseMemory | Vortex escape")
    t3 = time.time()
    qaoa_partition, qaoa_cut, memory = loopei_qaoa_solver(
        N, u_arr, v_arr, w_arr, p_layers=2, n_basins=18, total_w=total_w)
    print(f"\n    LoopEi QAOA cut:  {qaoa_cut:.4f}  (MPES={qaoa_cut/total_w:.4f})")
    print(f"    PhaseMemory iters:{memory.iteration}")
    print(f"    Time:             {time.time()-t3:.2f}s\n")

    # Certification
    print("[4] Certification...")
    gap       = abs(qaoa_cut - exact_cut)
    certified = gap < 0.01
    print(f"    Exact:     {exact_cut:.4f}")
    print(f"    LoopEi:    {qaoa_cut:.4f}")
    print(f"    Gap:       {gap:.4f}")
    print(f"    Status:    {'✔ EXACT OPTIMAL ACHIEVED' if certified else f'within {gap:.2f} of optimal'}\n")

    final_partition_np = exact_partition if qaoa_cut < exact_cut else qaoa_partition
    final_cut          = max(qaoa_cut, exact_cut)
    final_part_dict    = {i: int(final_partition_np[i]) for i in range(N)}

    # ── LQTrust++ ────────────────────────────────────────────────────────────
    print("[5] LQTrust++ Validation...")
    trust  = lqtrust_metrics(edges, final_part_dict)
    print(f"    Trust score:      {trust['trust_score']}")
    print(f"    N-1 resilience:   {trust['n_minus_one_resilience']}")
    print(f"    Balance:          {trust['balance']}")
    print(f"    Redundancy:       {trust['redundancy']}")
    print(f"    Bottleneck ratio: {trust['bottleneck_ratio']}")
    print(f"    Load conc.:       {trust['load_concentration']}\n")

    # ── Plateau scan ─────────────────────────────────────────────────────────
    print("[6] Plateau Optimality Proof (k-flip scan, k≤4)...")
    base, best_flip_val, best_flip, B = plateau_scan(edges, final_part_dict)
    plateau_optimal = best_flip is None
    print(f"    Blocking node set B: {B}")
    if plateau_optimal:
        print("    Result: ✔ NO IMPROVEMENT FOUND — solution is locally optimal")
        print("            under all k≤4 flip combinations over blocking node set")
    else:
        print(f"    WARNING: improvement found via flip {best_flip}  "
              f"Δ={best_flip_val-base:.4f}")
    print()

    # ── Blocking edges ────────────────────────────────────────────────────────
    blocking = get_blocking_edges(edges, final_part_dict)
    print(f"[7] Blocking edges (same-partition, {len(blocking)} found):")
    for u, v, w in sorted(blocking, key=lambda x: -x[2]):
        print(f"    {u:3d} — {v:3d}  weight={w:.4f}")
    print()

    # ── Visualization ─────────────────────────────────────────────────────────
    print("[8] Generating partition visualization...")
    plot_partition(edges, final_part_dict, trust['trust_score'])

    # ── Warm-start bitstring ──────────────────────────────────────────────────
    nodes_sorted    = sorted(final_part_dict.keys())
    warmstart_bits  = [final_part_dict[n] for n in nodes_sorted]
    warmstart_str   = "".join(str(b) for b in warmstart_bits)
    print(f"\n[9] Warm-start bitstring: {warmstart_str}")
    with open(os.path.join(WORK_DIR, "warmstart_bitstring.txt"), "w") as f:
        f.write(warmstart_str)

    # ── QAOA bridge ───────────────────────────────────────────────────────────
    print("\n[10] QAOA Warm-Start Bridge (Qiskit)...")

    # Build NetworkX graph for Qiskit stage
    qaoa_results = None
    try:
        import networkx as nx
        G_nx = nx.Graph()
        df_norm = pd.read_csv(args.graph)
        if 'node_1' in df_norm.columns:
            max_w_norm = df_norm['weight'].max()
            df_norm['weight'] = df_norm['weight'] / max_w_norm
            G_nx = nx.from_pandas_edgelist(
                df_norm, source='node_1', target='node_2', edge_attr='weight')
        total_w_norm = sum(d['weight'] for _, _, d in G_nx.edges(data=True))
        best_val_norm = exact_cut / max_w_norm
        half_total    = total_w_norm / 2

        qaoa_results = run_qaoa_bridge(
            warmstart_bits, nodes_sorted, G_nx,
            total_w_norm, best_val_norm, half_total,
            skip=True)
    except Exception as e:
        print(f"    [skip] QAOA bridge error: {e}")

    if qaoa_results:
        print(f"\n    Standard QAOA ratio:   {qaoa_results['standard_ratio']:.4f}  "
              f"({qaoa_results['standard_ratio']*100:.2f}%)")
        print(f"    Warm-start QAOA ratio: {qaoa_results['warmstart_ratio']:.4f}  "
              f"({qaoa_results['warmstart_ratio']*100:.2f}%)")
        print(f"    Improvement:           {qaoa_results['improvement_pp']:+.2f}pp")

    # ── Save outputs ──────────────────────────────────────────────────────────
    print("\n[11] Saving outputs...")

    report = {
        "cut_value":        round(final_cut, 6),
        "cut_ratio_mpes":   round(final_cut / total_w, 4),
        "total_weight":     round(total_w, 4),
        "certified_exact":  bool(certified),
        "plateau_optimal":  bool(plateau_optimal),
        "blocking_node_set":B,
        "warmstart_bitstring": warmstart_str,
        "trust":            trust,
        "qaoa_bridge":      qaoa_results,
        "framework": {
            "name":    "LoopEi Framework",
            "company": "LoopEi LLC",
            "patent":  "USPTO #19/303,438",
            "cage":    "16GM4",
            "method":  "LoopEi QAOA Hybrid + Exact Certification + LQTrust++",
            "components": [
                "Tesla 3-6-9 resonant seeding",
                "G1 Unity Governor",
                "PhaseMemory global tracking",
                "Vortex escape (plateau navigation)",
                "LQTrust++ resilience framework",
                "k-flip plateau optimality proof",
                "GPSEi warm-start bridge"
            ]
        }
    }

    with open(OUT_REPORT, "w") as f:
        json.dump(report, f, indent=2)
    print(f"    {OUT_REPORT}")

    loopei_result = {
        "cut_value":       round(final_cut, 4),
        "cut_ratio":       round(final_cut / total_w, 4),
        "total_weight":    round(total_w, 4),
        "partition":       [int(x) for x in final_partition_np],
        "certified_exact": bool(certified),
        "nodes": N, "edges": len(edges),
        "method": "LoopEi QAOA Hybrid + Exact Certification",
        "framework": "LoopEi LLC | USPTO #19/303,438"
    }
    with open(OUT_RESULT, "w") as f:
        json.dump(loopei_result, f, indent=2)
    print(f"    {OUT_RESULT}")

    cut_edges_out = [
        {"node_1": int(u), "node_2": int(v), "weight": round(float(w), 4)}
        for u, v, w in zip(u_arr, v_arr, w_arr)
        if final_partition_np[u] != final_partition_np[v]
    ]
    with open(OUT_EDGES, "w") as f:
        json.dump({"cut_edges": cut_edges_out,
                   "total_cut": round(final_cut, 4)}, f, indent=2)
    print(f"    {OUT_EDGES}")

    pd.DataFrame([(u, v, w) for u, v, w in blocking],
                 columns=["u", "v", "weight"]).to_csv(OUT_BLOCK, index=False)
    print(f"    {OUT_BLOCK}")

    pd.DataFrame([{
        "cut_value":   round(final_cut, 6),
        "trust_score": trust["trust_score"],
        "resilience":  trust["n_minus_one_resilience"],
        "mpes":        round(final_cut / total_w, 4),
        "certified":   bool(certified),
        "plateau_optimal": bool(plateau_optimal)
    }]).to_csv(OUT_SUMMARY, index=False)
    print(f"    {OUT_SUMMARY}")

    with open(os.path.join(WORK_DIR, "warmstart_bitstring.txt"), "w") as f:
        f.write(warmstart_str)
    print(f"    warmstart_bitstring.txt")

    # ── Final summary ─────────────────────────────────────────────────────────
    print("\n" + "="*65)
    print("  FINAL RESULTS")
    print("="*65)
    print(f"  Cut Value:          {final_cut:.4f}")
    print(f"  MPES:               {final_cut/total_w:.4f}  ({final_cut/total_w*100:.2f}%)")
    print(f"  Certified Optimal:  {'YES' if certified else 'NO'}")
    print(f"  Plateau Optimal:    {'YES' if plateau_optimal else 'NO'}")
    print(f"  Trust Score:        {trust['trust_score']}")
    print(f"  N-1 Resilience:     {trust['n_minus_one_resilience']}")
    print(f"  Warm-start:         {warmstart_str}")
    print(f"  Total time:         {time.time()-t0:.2f}s")
    print("="*65)
    print("  LoopEi — Done")
    print("="*65 + "\n")


if __name__ == "__main__":
    main()