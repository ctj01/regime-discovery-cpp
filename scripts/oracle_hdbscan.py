#!/usr/bin/env python3
# Copyright (c) 2026 Cristian Mendoza <openbeta.finance>
# SPDX-License-Identifier: AGPL-3.0-or-later
#
# This file is part of regime-discovery-cpp, licensed under the GNU Affero
# General Public License v3 or later. See LICENSE.
"""Independent HDBSCAN oracle for the C++ implementation (Phase 4).

Loads the standardized feature matrix dumped by the C++ pipeline, applies the
SAME temporal blocking (mutual-reachability inputs with |i-j| < W set to
infinity), runs a reference HDBSCAN (sklearn) with metric="precomputed" and
matched parameters, and writes day->label to data/oracle_labels.csv.

The C++ test suite compares its own labels against this file via ARI. The
oracle plays the role brute force played for HNSW: ground truth to prove the
from-scratch implementation correct, not code to copy.

Usage:
    ./build/mrd features data/spy.csv data/features_spy   # -> _std.csv
    python3 scripts/oracle_hdbscan.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
STD_CSV = ROOT / "data" / "features_spy_std.csv"
OUT_CSV = ROOT / "data" / "oracle_labels.csv"

W = 20              # temporal blocking window, derived from the feature window
MIN_CLUSTER_SIZE = 30
MIN_SAMPLES = 10


def main() -> None:
    df = pd.read_csv(STD_CSV)
    X = df.iloc[:, 1:].to_numpy()
    n = len(X)
    print(f"loaded {n} standardized rows from {STD_CSV.name}")

    # Full distance matrix with the temporal band blocked to infinity:
    # |i - j| < W pairs share feature-window data and must not be neighbors.
    diff = X[:, None, :] - X[None, :, :]
    D = np.sqrt((diff * diff).sum(axis=2))
    idx = np.arange(n)
    band = np.abs(idx[:, None] - idx[None, :]) < W
    D[band] = np.inf
    np.fill_diagonal(D, 0.0)

    from sklearn.cluster import HDBSCAN

    labels = HDBSCAN(
        metric="precomputed",
        min_cluster_size=MIN_CLUSTER_SIZE,
        min_samples=MIN_SAMPLES,
        cluster_selection_method="eom",
        allow_single_cluster=False,
    ).fit_predict(D)

    n_clusters = labels.max() + 1
    noise = int((labels == -1).sum())
    print(f"oracle: {n_clusters} clusters, noise {noise} ({100.0*noise/n:.1f}%)")

    pd.DataFrame({"index": idx, "label": labels}).to_csv(OUT_CSV, index=False)
    print(f"labels -> {OUT_CSV}")


if __name__ == "__main__":
    sys.exit(main())
