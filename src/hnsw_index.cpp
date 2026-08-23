// Copyright (c) 2026 Cristian Mendoza <openbeta.finance>
// SPDX-License-Identifier: AGPL-3.0-or-later
//
// This file is part of regime-discovery-cpp. It is free software: you can
// redistribute it and/or modify it under the terms of the GNU Affero General
// Public License as published by the Free Software Foundation, either version
// 3 of the License, or (at your option) any later version. See LICENSE.

#include "mrd/nn_index.hpp"

#include <algorithm>
#include <cmath>
#include <queue>
#include <random>
#include <stdexcept>

namespace mrd {

namespace {

double dist2(const double* a, const double* b, std::size_t d) {
    double s = 0.0;
    for (std::size_t j = 0; j < d; ++j) {
        const double t = a[j] - b[j];
        s += t * t;
    }
    return s;
}

// Level cap: at mL = 1/ln(12), P(level >= 24) ~ 12^-24 — unreachable at any
// realistic N; the cap only guards against a pathological RNG draw.
constexpr int kMaxLevel = 24;

}  // namespace

HnswIndex::HnswIndex() : HnswIndex(Params()) {}

HnswIndex::HnswIndex(Params params) : params_(params) {
    if (params_.M < 2) throw std::invalid_argument("HnswIndex: M must be >= 2");
    if (params_.ef_construction < params_.M) {
        throw std::invalid_argument("HnswIndex: ef_construction must be >= M");
    }
    level_mult_ = 1.0 / std::log(static_cast<double>(params_.M));
}

/// Best-first expansion at one layer (alg. 2): min-heap of frontier
/// candidates, bounded max-heap of the ef best results. Terminates when the
/// closest frontier candidate is farther than the worst kept result.
/// Returns (squared dist, id) ascending. `visited` is caller-owned scratch,
/// reset here.
std::vector<HnswIndex::Cand> HnswIndex::search_layer(
    const double* q, std::uint32_t ep, double ep_d2, std::size_t ef,
    std::size_t level, std::vector<char>& visited) const {
    std::priority_queue<Cand, std::vector<Cand>, std::greater<>> frontier;
    std::priority_queue<Cand> results;  // max-heap: top == worst kept

    visited.assign(n_, 0);
    visited[ep] = 1;
    frontier.emplace(ep_d2, ep);
    results.emplace(ep_d2, ep);

    while (!frontier.empty()) {
        const auto [dc, c] = frontier.top();
        if (results.size() >= ef && dc > results.top().first) break;
        frontier.pop();

        for (const std::uint32_t nb : links_[c][level]) {
            if (visited[nb]) continue;
            visited[nb] = 1;
            const double dn = dist2(q, vec(nb), d_);
            if (results.size() < ef || dn < results.top().first) {
                frontier.emplace(dn, nb);
                results.emplace(dn, nb);
                if (results.size() > ef) results.pop();
            }
        }
    }

    std::vector<Cand> out(results.size());
    for (auto it = out.rbegin(); it != out.rend(); ++it) {
        *it = results.top();
        results.pop();
    }
    return out;
}

/// Diversity heuristic (alg. 4) over candidates sorted by ascending distance
/// to the base point (distances precomputed in `cand_sorted`): keep c only if
/// c is closer to base than to every already-selected neighbor — this spreads
/// links across directions instead of clustering them. Pruned candidates are
/// then filled back (closest first) up to m: at d=4 the heuristic prunes
/// hard, and fill-back preserves graph connectivity.
std::vector<HnswIndex::Cand> HnswIndex::select_neighbors(
    const std::vector<Cand>& cand_sorted, std::size_t m) const {
    std::vector<Cand> selected;
    std::vector<Cand> pruned;
    selected.reserve(m);

    for (const auto& [dc, c] : cand_sorted) {
        if (selected.size() >= m) break;
        bool diverse = true;
        for (const auto& [ds, s] : selected) {
            (void)ds;
            if (dist2(vec(c), vec(s), d_) < dc) {
                diverse = false;
                break;
            }
        }
        if (diverse) {
            selected.emplace_back(dc, c);
        } else {
            pruned.emplace_back(dc, c);
        }
    }
    for (const auto& p : pruned) {
        if (selected.size() >= m) break;
        selected.push_back(p);
    }
    return selected;
}

/// Re-prunes an overflowing adjacency list with the same diversity heuristic,
/// distances taken from `node` itself.
void HnswIndex::prune_links(std::uint32_t node, std::size_t level,
                            std::size_t max_links) {
    auto& lst = links_[node][level];
    if (lst.size() <= max_links) return;

    std::vector<Cand> cand;
    cand.reserve(lst.size());
    for (const std::uint32_t nb : lst) {
        cand.emplace_back(dist2(vec(node), vec(nb), d_), nb);
    }
    std::sort(cand.begin(), cand.end());

    const auto kept = select_neighbors(cand, max_links);
    lst.clear();
    for (const auto& [d, nb] : kept) {
        (void)d;
        lst.push_back(nb);
    }
}

void HnswIndex::build(const FeatureMatrix& data) {
    n_ = data.n_rows();
    d_ = data.dim();
    data_.assign(data.data(), data.data() + n_ * d_);
    links_.assign(n_, {});
    max_level_ = -1;

    std::mt19937_64 rng(params_.seed);
    std::uniform_real_distribution<double> uni(0.0, 1.0);
    std::vector<char> visited;

    for (std::uint32_t i = 0; i < n_; ++i) {
        // 1 - U in (0, 1] so the log is finite; exponential decay in levels.
        const double u = 1.0 - uni(rng);
        const int level = std::min(
            kMaxLevel, static_cast<int>(std::floor(-std::log(u) * level_mult_)));
        links_[i].resize(static_cast<std::size_t>(level) + 1);

        if (max_level_ < 0) {  // first node
            entry_ = i;
            max_level_ = level;
            continue;
        }

        const double* q = vec(i);
        std::uint32_t cur = entry_;
        double cur_d2 = dist2(q, vec(cur), d_);

        // Greedy descent through layers above the new node's level (ef = 1).
        for (int l = max_level_; l > level; --l) {
            bool improved = true;
            while (improved) {
                improved = false;
                for (const std::uint32_t nb :
                     links_[cur][static_cast<std::size_t>(l)]) {
                    const double dn = dist2(q, vec(nb), d_);
                    if (dn < cur_d2) {
                        cur_d2 = dn;
                        cur = nb;
                        improved = true;
                    }
                }
            }
        }

        // Insert at each layer from min(level, max_level_) down to 0.
        for (int l = std::min(level, max_level_); l >= 0; --l) {
            const auto lvl = static_cast<std::size_t>(l);
            const auto cand =
                search_layer(q, cur, cur_d2, params_.ef_construction, lvl, visited);
            const auto selected = select_neighbors(cand, params_.M);

            const std::size_t cap = (l == 0) ? 2 * params_.M : params_.M;
            for (const auto& [ds, s] : selected) {
                (void)ds;
                links_[i][lvl].push_back(s);
                links_[s][lvl].push_back(i);
                prune_links(s, lvl, cap);
            }
            cur = cand.front().second;  // best hit seeds the next layer down
            cur_d2 = cand.front().first;
        }

        if (level > max_level_) {
            max_level_ = level;
            entry_ = i;
        }
    }
}

std::vector<Neighbor> HnswIndex::query(std::span<const double> q, int k,
                                       std::optional<std::size_t> exclude) const {
    if (max_level_ < 0) throw std::logic_error("HnswIndex::query before build");
    if (k <= 0) throw std::invalid_argument("k must be positive");
    if (q.size() != d_) throw std::logic_error("query dim mismatch");

    std::uint32_t cur = entry_;
    double cur_d2 = dist2(q.data(), vec(cur), d_);
    for (int l = max_level_; l >= 1; --l) {
        bool improved = true;
        while (improved) {
            improved = false;
            for (const std::uint32_t nb : links_[cur][static_cast<std::size_t>(l)]) {
                const double dn = dist2(q.data(), vec(nb), d_);
                if (dn < cur_d2) {
                    cur_d2 = dn;
                    cur = nb;
                    improved = true;
                }
            }
        }
    }

    // One extra candidate absorbs a potential self-exclusion hit.
    const std::size_t ef = std::max(params_.ef_search,
                                    static_cast<std::size_t>(k) + (exclude ? 1u : 0u));
    std::vector<char> visited;
    const auto cand = search_layer(q.data(), cur, cur_d2, ef, 0, visited);

    std::vector<Neighbor> out;
    out.reserve(static_cast<std::size_t>(k));
    for (const auto& [d2, id] : cand) {
        if (exclude && *exclude == id) continue;
        out.push_back({id, std::sqrt(d2)});
        if (out.size() == static_cast<std::size_t>(k)) break;
    }
    return out;
}

}  // namespace mrd
