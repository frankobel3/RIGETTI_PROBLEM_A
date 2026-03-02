MPES Computation Workflow
21-Node Grid (Problem A)
Team Documentation
1 Overview
This document describes the workflow used to compute the Maximum Power Energy Section
(MPES) for a 21-node grid dataset using a weighted Max-Cut formulation.
The focus is on the computation pipeline, reproducibility, and outputs.
2 Problem Definition
The grid is modeled as a weighted undirected graph:
• Nodes → substations or grid points
• Edges → transmission lines
• Weights → power transfer capacity
The objective is to compute the MPES:
The partition of nodes that maximizes the total weight of edges crossing the partition.
This is equivalent to the weighted Max-Cut problem.
3 Outputs Produced
The workflow computes:
• Exact optimal partition
• Cut value and MPES
• Local stability verification (k-flip plateau check)
• Grid stability metrics (trust model)
• Optional inputs for quantum workflows
1
4 Repository Contents
RigettiASolver.py
problema.csv
partition.csv
report_full.json
loopei_result_A.json
loopei_cut_edges_A.json
team_summary.csv
blocking_edges.csv
warmstart_bitstring.txt
RigettiProblemA.lean
assets/
# Main computation pipeline
# Graph dataset
# Final partition vector
# Full analysis output
# Compact results
# Cut edge list
# Trust metric components
# Plateau analysis support
# Partition seed
# Formal spec scaffold
# Visualizations
5 Computation Pipeline
5.1 Graph Loading
The dataset problema.csv is loaded into a weighted undirected graph.
Columns:
• source node
• destination node
• edge weight
5.2 Exact Optimal Cut
For 21 nodes, the global optimum is computed via half-enumeration:
• Enumerate half the partition space
• Compute cut weight
• Track maximum
This guarantees the optimal solution.
5.3 Plateau Verification (Local Optimality)
We test whether the solution can be improved by flipping small sets of nodes.
1. Identify blocking edges (edges not in the cut)
2. Extract blocking node set
3. Test all k-node flips (k ≤ 4)
4. Confirm no improvement exists
This verifies local stability of the partition.
2
5.4 Trust and Stability Metrics
The workflow computes grid-relevant indicators:
• Load balance — partition symmetry
• Redundancy — alternative cross-partition paths
• N-1 resilience — effect of single edge removal
• Bottleneck ratio — concentration of flow through few edges
These metrics are aggregated into a trust score.
5.5 Optional QAOA Bridge
The solver can generate inputs for a QAOA workflow:
• Cost Hamiltonian parameters
• Warm-start partition
• Classical baseline for comparison
This step is optional and skipped if quantum libraries are not installed.
6 Howto Run
Install dependencies:
pip install-r requirements.txt
Run the solver:
python RigettiASolver.py--input problema.csv
7 Expected Output
Nodes: 21 | Edges: 28
Exact optimal certification: SUCCESS
Cut value: 3728.4132
MPES: 0.8844
Trust score: 0.9382
Plateau optimal (k
4): TRUE
Matching these values confirms correct execution.
3
8 Output Files
8.1 report full.json
Contains:
• cut value
• mpes
• plateau verification
• trust metrics
8.2 loopei result A.json
Compact summary for downstream use.
8.3 loopei cut edges A.json
Edges contributing to the cut.
8.4 team summary.csv
Breakdown of trust components.
9 Determinism
The pipeline is deterministic:
• No randomness
• Exact enumeration
• Exhaustive k-flip checks
Repeated runs produce identical results.
10 Design Rationale
10.1 Exact Enumeration
Feasible for 21 nodes and provides ground truth.
10.2 Plateau Verification
Ensures solution stability and interpretability.
10.3 Trust Metrics
Provide operational context beyond MPES.
4
10.4 JSON Outputs
Enable reuse in visualization and quantum workflows.
11 Extending the Workflow
11.1 Larger Graphs
• Replace enumeration with heuristic search
• Retain plateau checks on subsets
11.2 Quantum Workflow
• Integrate QAOA parameter optimization
• Compare with classical optimum
11.3 Grid Analysis
• Incorporate dynamic weights
• Simulate multi-edge failures
12 Limitations
• Plateau verification limited to k ≤ 4
• Trust metrics are heuristic indicators
• Lean file is a specification scaffold, not a mechanized proof
13 Summary
This workflow provides a reproducible method for computing and analyzing MPES via weighted
Max-Cut, including optimality certification, stability analysis, and extensible outputs for future
work.
5
