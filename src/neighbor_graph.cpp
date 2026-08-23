// Copyright (c) 2026 Cristian Mendoza <openbeta.finance>
// SPDX-License-Identifier: AGPL-3.0-or-later
//
// This file is part of regime-discovery-cpp. It is free software: you can
// redistribute it and/or modify it under the terms of the GNU Affero General
// Public License as published by the Free Software Foundation, either version
// 3 of the License, or (at your option) any later version. See LICENSE.

#include "mrd/neighbor_graph.hpp"

#include <cstdlib>
#include <stdexcept>
#include <string>

namespace mrd {

NeighborGraph build_neighbor_graph(const NNIndex& index, const FeatureMatrix& m,
                                   std::size_t k, std::size_t block_window) {
    const std::size_t n = m.n_rows();
    const std::size_t band = block_window > 0 ? 2 * (block_window - 1) : 0;
    if (k == 0) throw std::invalid_argument("build_neighbor_graph: k must be > 0");
    if (n <= k + band) {
        throw std::invalid_argument(
            "build_neighbor_graph: need n > k + 2*(W-1) = " +
            std::to_string(k + band) + ", got n = " + std::to_string(n));
    }

    NeighborGraph g;
    g.k = k;
    g.block_window = block_window;
    g.flat.reserve(n * k);

    const int query_k = static_cast<int>(k + band);
    for (std::size_t t = 0; t < n; ++t) {
        const auto cand = index.query(m.row(t), query_k, t);
        std::size_t kept = 0;
        for (const auto& nb : cand) {
            if (block_window > 0) {
                const std::size_t lag = t > nb.index ? t - nb.index : nb.index - t;
                if (lag < block_window) continue;
            }
            g.flat.push_back(nb);
            if (++kept == k) break;
        }
        // Unreachable given the n > k + band precondition; guard anyway so a
        // future index bug cannot silently produce ragged rows.
        if (kept != k) {
            throw std::runtime_error("build_neighbor_graph: row " +
                                     std::to_string(t) + " kept " +
                                     std::to_string(kept) + " < k neighbors");
        }
    }
    return g;
}

}  // namespace mrd
