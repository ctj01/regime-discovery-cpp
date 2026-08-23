// Copyright (c) 2026 Cristian Mendoza <openbeta.finance>
// SPDX-License-Identifier: AGPL-3.0-or-later
//
// This file is part of regime-discovery-cpp. It is free software: you can
// redistribute it and/or modify it under the terms of the GNU Affero General
// Public License as published by the Free Software Foundation, either version
// 3 of the License, or (at your option) any later version. See LICENSE.

#pragma once

#include <cstddef>
#include <vector>

#include "mrd/types.hpp"

namespace mrd {

/// Stage 3 validators: pure functions of the FINAL day->regime assignment
/// (plus price data for the OOS test). None of them look at the neighbor
/// graph — persistence measured on the blocked graph would be circular.
/// Unsupervised clustering always "finds" clusters; these three tests decide
/// whether the regimes are real.

/// Persistence: run-length structure of a label sequence.
struct RunStats {
    int label;               ///< regime id, or -1 for the noise segments
    std::size_t n_days;
    std::size_t n_runs;      ///< maximal runs of consecutive identical labels
    double mean_run_length;  ///< ~1 => temporal noise; ~10-30 => real structure
};

/// One entry per label present in `labels` (noise included, as its own row —
/// informative but not a regime), ascending by label.
std::vector<RunStats> persistence(const std::vector<int>& labels);

/// Recurrence: temporally disjoint episodes per regime. The days of a regime
/// are split into episodes wherever the gap to the next occurrence exceeds
/// `gap` trading days. A real regime recurs across separated epochs; a
/// calendar artifact is a single contiguous stretch. `gap` is a judgment
/// call (default 60 ≈ one quarter, well above the derived W=20) — unlike W
/// it is tunable, so episode counts should be read alongside it.
struct EpisodeStats {
    int label;
    std::size_t n_days;
    std::size_t n_episodes;
    std::size_t first_day, last_day;  ///< row indices of first/last occurrence
};

/// Noise (-1) excluded: episodes of "no regime" are not meaningful.
std::vector<EpisodeStats> recurrence(const std::vector<int>& labels,
                                     std::size_t gap);

/// Out-of-sample economic content: realized volatility of the NEXT `horizon`
/// days, grouped by regime.
///
/// FORWARD-LOOKING BY DESIGN: this is the one place in the project that reads
/// the future, and it exists only to validate — its output must never feed a
/// feature or an assignment. For label row t (series day T = t + first_row_day),
/// it computes the sample std (ddof=1) of log returns r[T+1 .. T+horizon].
/// Rows without a complete forward window are excluded (their count is the
/// difference between n here and n in persistence()).
struct ForwardVolStats {
    int label;           ///< regime id, or -1 for noise
    std::size_t n;
    double mean;
    double median;
};

/// One entry per label (noise included), ascending by label. `first_row_day`
/// is the series index of label row 0 (== the Phase 1 window length).
std::vector<ForwardVolStats> forward_vol_by_regime(
    const std::vector<int>& labels, const OhlcvSeries& series,
    std::size_t first_row_day, std::size_t horizon = 20);

}  // namespace mrd
