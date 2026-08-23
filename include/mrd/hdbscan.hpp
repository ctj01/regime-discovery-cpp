// Copyright (c) 2026 Cristian Mendoza <openbeta.finance>
// SPDX-License-Identifier: AGPL-3.0-or-later
//
// This file is part of regime-discovery-cpp. It is free software: you can
// redistribute it and/or modify it under the terms of the GNU Affero General
// Public License as published by the Free Software Foundation, either version
// 3 of the License, or (at your option) any later version. See LICENSE.

#pragma once

#include <cstdint>
#include <vector>

#include "mrd/neighbor_graph.hpp"
#include "mrd/types.hpp"

namespace mrd {

struct HdbscanParams {
    /// Smallest point count that can constitute a regime. 30 ≈ six trading
    /// weeks: a "regime" that cannot accumulate 30 days across 33 years is
    /// not a regime. Judgment call, CLI-tunable (unlike W, which is derived).
    std::size_t min_cluster_size = 30;
    /// k for core distances; must be <= the graph's k.
    std::size_t min_samples = 10;
    /// MEASURED FAILURE MODE, kept only for comparison: the MST restricted to
    /// k-NN edges misses portal edges between dense regions (a cluster-edge
    /// point's k neighbors are all interior points, so the true inter-cluster
    /// bridge enters the edge set only when k ~ cluster size). On the blob
    /// fixture the two blobs merge at 16.6 through a noise chain instead of
    /// at the true 6.0 bridge, distorting merge heights exactly where regime
    /// structure lives. Default false => exact implicit-matrix Prim.
    bool substrate_mst = false;
};

struct HdbscanResult {
    /// Per-row regime id in [0, n_clusters), or -1 = noise. Noise is a
    /// first-class outcome: most market days belong to no distinctive regime.
    std::vector<int> labels;
    /// Stability (excess of mass) of each extracted cluster, indexed by id.
    std::vector<double> stabilities;
    std::size_t n_clusters = 0;
};

/// HDBSCAN over a k-NN graph substrate (Campello/Moulavi/Sander via the
/// Malkov-free formulation used by the reference `hdbscan` library):
/// core distances from the graph's min_samples-th neighbor, mutual
/// reachability, MST, single-linkage condensation with min_cluster_size,
/// excess-of-mass cluster extraction. The root cluster is never selected
/// (no allow_single_cluster).
///
/// `graph` is expected to be the TEMPORALLY BLOCKED view (Stage 1); passing
/// the raw graph is valid but rediscovers calendar stretches — the blocking
/// decision deliberately lives outside this function (independent stages).
///
/// `points` must be the matrix the graph was built on (needed for exact MST
/// bridging edges between k-NN components; see build_mst).
///
/// Deterministic: no RNG anywhere; ties resolved by fixed orderings.
/// Throws std::invalid_argument on min_samples > graph.k, min_cluster_size
/// < 2, or points/graph size mismatch.
HdbscanResult hdbscan(const FeatureMatrix& points, const NeighborGraph& graph,
                      const HdbscanParams& params);

/// Sub-pieces exposed for validation — each is independently testable
/// against an oracle (in-test exact Prim, hand fixtures, Python hdbscan).
namespace hdbscan_detail {

/// core_dist[t] = distance to the (min_samples - 1)-th OTHER neighbor in `g`
/// — min_samples counts the point itself, matching sklearn's HDBSCAN and the
/// hdbscan library's primary (tree-based) path. (The mcinnes precomputed
/// path counts one more neighbor; that inconsistency cost an ARI of 0.75
/// against the oracle until pinned down — hence this explicit contract.)
/// Requires 2 <= min_samples <= g.k + 1. Free: the graph already holds it.
std::vector<double> core_distances(const NeighborGraph& g, std::size_t min_samples);

/// d_mreach(a,b) = max(d(a,b), core_a, core_b).
double mutual_reachability(double dist, double core_a, double core_b);

struct Edge {
    std::uint32_t a, b;
    double w;  // mutual reachability distance
};

/// EXACT MST of the mutual-reachability graph: Prim over the implicit
/// complete graph (distances computed on the fly, O(N^2) time, O(N) memory —
/// ~70M d=4 evaluations at N=8428, sub-second). This is the default: the
/// k-NN substrate keeps feeding HDBSCAN through the core distances, but MST
/// topology must be exact (see HdbscanParams::substrate_mst for why).
///
/// block_window applies the SAME temporal exclusion as Stage 1 to the MST
/// edge set (pairs with |a-b| < W are unusable, as if at infinite distance):
/// day t must not connect to day t±1 directly in the discovery structure —
/// blocking only the core distances would let window-overlap pairs back in
/// through the tree. Throws std::runtime_error if blocking disconnects the
/// graph (cannot happen for W << n).
std::vector<Edge> build_mst_exact(const FeatureMatrix& points,
                                  const std::vector<double>& core,
                                  std::size_t block_window = 0);

/// APPROXIMATE MST on the k-NN substrate (kept as a measurable variant, not
/// the default — see HdbscanParams::substrate_mst): Kruskal over the
/// symmetrized, deduplicated k-NN edges; if the forest keeps multiple
/// components, Boruvka rounds add the exact minimum cross-component edge by
/// full scan. Connectivity is exact; edges *within* an already-connected
/// region are limited to the k-NN edge set, which omits inter-cluster portal
/// edges once noise chains connect the graph.
std::vector<Edge> build_mst(const FeatureMatrix& points, const NeighborGraph& g,
                            const std::vector<double>& core);

/// Condensation + excess-of-mass extraction from the MST (edges in any
/// order; sorted internally). Split point at distance d has lambda = 1/d;
/// a child with < min_cluster_size points falls out of its parent at that
/// lambda, a child with >= continues (or opens a new cluster when both
/// qualify). stability(C) = sum over departures of (lambda_leave -
/// lambda_birth) * size. Selection: bottom-up, a cluster beats its
/// descendants iff its stability exceeds their selected sum; root excluded.
HdbscanResult extract_clusters(std::vector<Edge> mst, std::size_t n,
                               std::size_t min_cluster_size);

}  // namespace hdbscan_detail
}  // namespace mrd
