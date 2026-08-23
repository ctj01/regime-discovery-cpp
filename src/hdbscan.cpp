// Copyright (c) 2026 Cristian Mendoza <openbeta.finance>
// SPDX-License-Identifier: AGPL-3.0-or-later
//
// This file is part of regime-discovery-cpp. It is free software: you can
// redistribute it and/or modify it under the terms of the GNU Affero General
// Public License as published by the Free Software Foundation, either version
// 3 of the License, or (at your option) any later version. See LICENSE.

#include "mrd/hdbscan.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>

namespace mrd {
namespace hdbscan_detail {

namespace {

double dist2(const double* a, const double* b, std::size_t d) {
    double s = 0.0;
    for (std::size_t j = 0; j < d; ++j) {
        const double t = a[j] - b[j];
        s += t * t;
    }
    return s;
}

struct Dsu {
    std::vector<std::uint32_t> parent;
    explicit Dsu(std::size_t n) : parent(n) {
        std::iota(parent.begin(), parent.end(), 0u);
    }
    std::uint32_t find(std::uint32_t x) {
        while (parent[x] != x) {
            parent[x] = parent[parent[x]];
            x = parent[x];
        }
        return x;
    }
    bool unite(std::uint32_t a, std::uint32_t b) {
        a = find(a);
        b = find(b);
        if (a == b) return false;
        parent[b] = a;
        return true;
    }
};

// Guard against exactly-duplicate points: lambda = 1/max(d, kMinDist).
// Standardized features live at O(1), so 1e-12 is unambiguously "identical".
constexpr double kMinDist = 1e-12;

}  // namespace

std::vector<double> core_distances(const NeighborGraph& g, std::size_t min_samples) {
    if (min_samples < 2 || min_samples > g.k + 1) {
        throw std::invalid_argument("core_distances: need 2 <= min_samples <= graph k + 1");
    }
    const std::size_t n = g.n_rows();
    std::vector<double> core(n);
    for (std::size_t t = 0; t < n; ++t) {
        // min_samples counts the point itself: (min_samples-1)-th other
        // neighbor, which is row index min_samples - 2.
        core[t] = g.row(t)[min_samples - 2].dist;
    }
    return core;
}

double mutual_reachability(double dist, double core_a, double core_b) {
    return std::max({dist, core_a, core_b});
}

std::vector<Edge> build_mst_exact(const FeatureMatrix& points,
                                  const std::vector<double>& core,
                                  std::size_t block_window) {
    const std::size_t n = points.n_rows();
    if (core.size() != n) {
        throw std::invalid_argument("build_mst_exact: points/core size mismatch");
    }
    const std::size_t d = points.dim();
    constexpr double kInf = std::numeric_limits<double>::infinity();

    std::vector<char> in_tree(n, 0);
    std::vector<double> best(n, kInf);
    std::vector<std::uint32_t> best_from(n, 0);
    best[0] = 0.0;

    std::vector<Edge> mst;
    mst.reserve(n - 1);
    for (std::size_t it = 0; it < n; ++it) {
        std::uint32_t u = 0;
        double bu = kInf;
        for (std::uint32_t v = 0; v < n; ++v) {
            if (!in_tree[v] && best[v] < bu) {
                bu = best[v];
                u = v;
            }
        }
        if (bu == kInf && it > 0) {
            throw std::runtime_error("build_mst_exact: blocking disconnected the graph");
        }
        in_tree[u] = 1;
        if (it > 0) mst.push_back({best_from[u], u, best[u]});

        const double* pu = points.data() + u * d;
        for (std::uint32_t v = 0; v < n; ++v) {
            if (in_tree[v]) continue;
            // Same Stage-1 exclusion as the neighbor graph: temporally
            // overlapping pairs cannot be tree edges.
            const std::size_t lag = u > v ? u - v : v - u;
            if (lag < block_window) continue;
            const double w = mutual_reachability(
                std::sqrt(dist2(pu, points.data() + v * d, d)), core[u], core[v]);
            if (w < best[v]) {
                best[v] = w;
                best_from[v] = u;
            }
        }
    }
    return mst;
}

std::vector<Edge> build_mst(const FeatureMatrix& points, const NeighborGraph& g,
                            const std::vector<double>& core) {
    const std::size_t n = g.n_rows();
    if (points.n_rows() != n || core.size() != n) {
        throw std::invalid_argument("build_mst: points/graph/core size mismatch");
    }
    const std::size_t d = points.dim();

    // Candidate edges from the k-NN substrate, canonicalized a < b, deduped.
    std::vector<Edge> cand;
    cand.reserve(n * g.k);
    for (std::uint32_t t = 0; t < n; ++t) {
        for (const auto& nb : g.row(t)) {
            const auto s = static_cast<std::uint32_t>(nb.index);
            const std::uint32_t a = std::min(t, s), b = std::max(t, s);
            cand.push_back({a, b, mutual_reachability(nb.dist, core[a], core[b])});
        }
    }
    std::sort(cand.begin(), cand.end(), [](const Edge& x, const Edge& y) {
        return x.a != y.a ? x.a < y.a : x.b < y.b;
    });
    cand.erase(std::unique(cand.begin(), cand.end(),
                           [](const Edge& x, const Edge& y) {
                               return x.a == y.a && x.b == y.b;
                           }),
               cand.end());
    std::sort(cand.begin(), cand.end(), [](const Edge& x, const Edge& y) {
        return x.w != y.w ? x.w < y.w : (x.a != y.a ? x.a < y.a : x.b < y.b);
    });

    // Kruskal over substrate edges.
    Dsu dsu(n);
    std::vector<Edge> mst;
    mst.reserve(n - 1);
    std::size_t components = n;
    for (const auto& e : cand) {
        if (dsu.unite(e.a, e.b)) {
            mst.push_back(e);
            --components;
        }
    }

    // Boruvka rounds with EXACT minimum cross-component edges (full scan):
    // guarantees connectivity even when the blocked k-NN graph is
    // disconnected across dense regions.
    while (components > 1) {
        constexpr auto kInf = std::numeric_limits<double>::infinity();
        std::vector<Edge> best(n, {0, 0, kInf});  // indexed by component root
        for (std::uint32_t a = 0; a < n; ++a) {
            const std::uint32_t ra = dsu.find(a);
            for (std::uint32_t b = a + 1; b < n; ++b) {
                const std::uint32_t rb = dsu.find(b);
                if (ra == rb) continue;
                if (b - a < g.block_window) continue;  // Stage-1 exclusion
                const double w = mutual_reachability(
                    std::sqrt(dist2(points.data() + a * d, points.data() + b * d, d)),
                    core[a], core[b]);
                if (w < best[ra].w) best[ra] = {a, b, w};
                if (w < best[rb].w) best[rb] = {a, b, w};
            }
        }
        for (std::uint32_t r = 0; r < n; ++r) {
            if (best[r].w == kInf) continue;
            if (dsu.unite(best[r].a, best[r].b)) {
                mst.push_back(best[r]);
                --components;
            }
        }
    }

    return mst;
}

HdbscanResult extract_clusters(std::vector<Edge> mst, std::size_t n,
                               std::size_t min_cluster_size) {
    if (min_cluster_size < 2) {
        throw std::invalid_argument("extract_clusters: min_cluster_size must be >= 2");
    }
    HdbscanResult res;
    res.labels.assign(n, -1);
    if (n < 2 || mst.size() != n - 1) {
        if (n >= 2) throw std::invalid_argument("extract_clusters: mst must have n-1 edges");
        return res;
    }

    // --- Single-linkage dendrogram: leaves 0..n-1, merges n..2n-2. ---
    std::sort(mst.begin(), mst.end(), [](const Edge& x, const Edge& y) {
        return x.w != y.w ? x.w < y.w : (x.a != y.a ? x.a < y.a : x.b < y.b);
    });
    std::vector<std::uint32_t> left(n - 1), right(n - 1);
    std::vector<double> merge_dist(n - 1);
    std::vector<std::uint32_t> subtree_size(2 * n - 1, 1);
    Dsu dsu(n);
    std::vector<std::uint32_t> node_of(n);
    std::iota(node_of.begin(), node_of.end(), 0u);
    for (std::size_t i = 0; i < mst.size(); ++i) {
        const std::uint32_t ra = dsu.find(mst[i].a);
        const std::uint32_t rb = dsu.find(mst[i].b);
        left[i] = node_of[ra];
        right[i] = node_of[rb];
        merge_dist[i] = mst[i].w;
        dsu.unite(ra, rb);
        const auto merged = static_cast<std::uint32_t>(n + i);
        node_of[dsu.find(ra)] = merged;
        subtree_size[merged] = subtree_size[left[i]] + subtree_size[right[i]];
    }

    // --- Condensation. Rows: (parent cond-cluster, child, lambda, size);
    // child < n is a point falling out, child >= n is a new cond cluster. ---
    struct CondRow {
        std::uint32_t parent, child;
        double lambda;
        std::uint32_t size;
    };
    std::vector<CondRow> rows;
    rows.reserve(2 * n);
    const std::uint32_t root_cluster = static_cast<std::uint32_t>(n);
    std::uint32_t next_cluster = root_cluster + 1;

    auto spill_leaves = [&](std::uint32_t node, std::uint32_t parent, double lambda) {
        std::vector<std::uint32_t> stack{node};
        while (!stack.empty()) {
            const std::uint32_t x = stack.back();
            stack.pop_back();
            if (x < n) {
                rows.push_back({parent, x, lambda, 1});
            } else {
                stack.push_back(left[x - n]);
                stack.push_back(right[x - n]);
            }
        }
    };

    std::vector<std::pair<std::uint32_t, std::uint32_t>> work;  // dendro node, cond cluster
    work.emplace_back(static_cast<std::uint32_t>(2 * n - 2), root_cluster);
    while (!work.empty()) {
        const auto [node, cluster] = work.back();
        work.pop_back();
        const std::size_t i = node - n;  // node >= n always: leaves never pushed
        const double lambda = 1.0 / std::max(merge_dist[i], kMinDist);
        const std::uint32_t a = left[i], b = right[i];
        const std::uint32_t sa = subtree_size[a], sb = subtree_size[b];
        const bool a_big = sa >= min_cluster_size, b_big = sb >= min_cluster_size;

        if (a_big && b_big) {  // true split: two new condensed clusters
            const std::uint32_t ca = next_cluster++;
            rows.push_back({cluster, ca, lambda, sa});
            work.emplace_back(a, ca);
            const std::uint32_t cb = next_cluster++;
            rows.push_back({cluster, cb, lambda, sb});
            work.emplace_back(b, cb);
        } else if (a_big) {  // b falls out; cluster continues as a
            spill_leaves(b, cluster, lambda);
            work.emplace_back(a, cluster);
        } else if (b_big) {
            spill_leaves(a, cluster, lambda);
            work.emplace_back(b, cluster);
        } else {  // cluster dissolves entirely at this level
            spill_leaves(a, cluster, lambda);
            spill_leaves(b, cluster, lambda);
        }
    }

    // --- Stability per condensed cluster. ---
    const std::size_t n_cond = next_cluster - root_cluster;
    std::vector<double> birth(n_cond, 0.0);  // root birth = 0
    std::vector<std::uint32_t> parent_of(n_cond, root_cluster);
    std::vector<std::vector<std::uint32_t>> children(n_cond);
    for (const auto& r : rows) {
        if (r.child >= root_cluster) {
            birth[r.child - root_cluster] = r.lambda;
            parent_of[r.child - root_cluster] = r.parent;
            children[r.parent - root_cluster].push_back(r.child);
        }
    }
    std::vector<double> stability(n_cond, 0.0);
    for (const auto& r : rows) {
        stability[r.parent - root_cluster] +=
            (r.lambda - birth[r.parent - root_cluster]) * r.size;
    }

    // --- Excess-of-mass selection (root excluded; children have larger ids
    // than parents by construction, so descending id = bottom-up). ---
    std::vector<char> selected(n_cond, 0);
    for (std::uint32_t c = next_cluster - 1; c > root_cluster; --c) {
        const std::size_t ci = c - root_cluster;
        double child_sum = 0.0;
        for (const std::uint32_t ch : children[ci]) {
            child_sum += stability[ch - root_cluster];
        }
        if (!children[ci].empty() && child_sum > stability[ci]) {
            stability[ci] = child_sum;  // propagate upward
        } else {
            selected[ci] = 1;  // this cluster beats its descendants
            std::vector<std::uint32_t> stack(children[ci]);
            while (!stack.empty()) {
                const std::uint32_t x = stack.back();
                stack.pop_back();
                selected[x - root_cluster] = 0;
                for (const std::uint32_t ch : children[x - root_cluster]) {
                    stack.push_back(ch);
                }
            }
        }
    }

    // --- Labels: ascending selected-cluster id -> 0.. (deterministic). ---
    std::vector<int> label_of(n_cond, -1);
    for (std::uint32_t c = root_cluster + 1; c < next_cluster; ++c) {
        if (selected[c - root_cluster]) {
            label_of[c - root_cluster] = static_cast<int>(res.n_clusters++);
            res.stabilities.push_back(stability[c - root_cluster]);
        }
    }
    for (const auto& r : rows) {
        if (r.child >= root_cluster) continue;
        std::uint32_t c = r.parent;
        while (c != root_cluster && !selected[c - root_cluster]) {
            c = parent_of[c - root_cluster];
        }
        if (c != root_cluster) res.labels[r.child] = label_of[c - root_cluster];
    }
    return res;
}

}  // namespace hdbscan_detail

HdbscanResult hdbscan(const FeatureMatrix& points, const NeighborGraph& graph,
                      const HdbscanParams& params) {
    if (points.n_rows() != graph.n_rows()) {
        throw std::invalid_argument("hdbscan: points/graph size mismatch");
    }
    using namespace hdbscan_detail;
    const auto core = core_distances(graph, params.min_samples);
    auto mst = params.substrate_mst
                   ? build_mst(points, graph, core)
                   : build_mst_exact(points, core, graph.block_window);
    return extract_clusters(std::move(mst), points.n_rows(),
                            params.min_cluster_size);
}

}  // namespace mrd
