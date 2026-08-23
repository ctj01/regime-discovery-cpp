// Copyright (c) 2026 Cristian Mendoza <openbeta.finance>
// SPDX-License-Identifier: AGPL-3.0-or-later
//
// This file is part of regime-discovery-cpp. It is free software: you can
// redistribute it and/or modify it under the terms of the GNU Affero General
// Public License as published by the Free Software Foundation, either version
// 3 of the License, or (at your option) any later version. See LICENSE.

#pragma once

#include <span>
#include <vector>

#include "mrd/nn_index.hpp"
#include "mrd/types.hpp"

namespace mrd {

/// Materialized k-NN graph: row t holds the k nearest neighbors of row t,
/// ascending by distance, in one contiguous block. This is a VIEW derived
/// from an NNIndex query sweep — building a blocked view never touches the
/// index or any previously built graph (same raw-vs-derived discipline as
/// the standardized matrix in Phase 2).
struct NeighborGraph {
    std::size_t k = 0;
    /// The temporal blocking this view was built with (0 = raw). Carried as
    /// metadata so downstream consumers (e.g. the HDBSCAN MST) can honor the
    /// SAME exclusion — Stage 1 sanitizes the entire clustering input, not
    /// just the core distances.
    std::size_t block_window = 0;
    std::vector<Neighbor> flat;  // n * k, row-major

    [[nodiscard]] std::size_t n_rows() const noexcept {
        return k == 0 ? 0 : flat.size() / k;
    }
    [[nodiscard]] std::span<const Neighbor> row(std::size_t t) const noexcept {
        return {flat.data() + t * k, k};
    }
};

/// Builds the k-NN graph over the rows of `m` using `index` (already built
/// on the same matrix), excluding self-matches.
///
/// block_window = 0: raw graph — the truth for "most similar days".
/// block_window = W: temporal blocking — neighbor s of day t is excluded iff
/// |t - s| < W (row indices are consecutive trading days, so index distance
/// is trading-day distance).
///
/// W = 20 IS DERIVED, NOT TUNED: the day-t feature window covers returns
/// r[t-19..t], so windows at lag D share 20 - D returns — overlap reaches
/// exactly zero at D = 20. Blocking |t - s| < 20 removes every pair that is
/// near-identical by construction (shared window data) and no pair that is
/// similar by market structure. It is a constant of the problem geometry:
/// change the Phase 1 window and W must change with it (see kDefaultWindow).
///
/// Guarantee: every row of the result has exactly k neighbors. The internal
/// index query asks for k + 2*(W-1) candidates — the blocked band around t
/// contains at most 2*(W-1) rows besides t itself — so k survivors always
/// exist. Throws std::invalid_argument if n <= k + 2*(W-1) (series too short
/// to guarantee that).
NeighborGraph build_neighbor_graph(const NNIndex& index, const FeatureMatrix& m,
                                   std::size_t k, std::size_t block_window = 0);

}  // namespace mrd
