// Copyright (c) 2026 Cristian Mendoza <openbeta.finance>
// SPDX-License-Identifier: AGPL-3.0-or-later
//
// This file is part of regime-discovery-cpp. It is free software: you can
// redistribute it and/or modify it under the terms of the GNU Affero General
// Public License as published by the Free Software Foundation, either version
// 3 of the License, or (at your option) any later version. See LICENSE.

#pragma once

#include <span>

#include "mrd/types.hpp"

namespace mrd {

/// Default trailing window length, in trading days (returns per window).
inline constexpr std::size_t kDefaultWindow = 20;

// ---------------------------------------------------------------------------
// Window kernels. Each kernel sees ONLY the span it is given; the no-lookahead
// guarantee is established by the caller (compute_features), which slices
// spans ending at day t. Kernels are pure functions of their input.
// ---------------------------------------------------------------------------

/// Realized volatility: sample standard deviation (ddof=1) of the given
/// log returns.
///
/// Units: daily (NOT annualized — annualization is a constant scalar that the
/// Phase 2 z-score would absorb anyway).
/// Assumes: returns.size() >= 2.
double realized_volatility(std::span<const double> returns);

/// Momentum: cumulative log return over the window, sum(r_i).
///
/// Telescopes to ln(C_t / C_{t-w}) when the returns are consecutive log
/// returns of the same close series (checked in tests).
/// Units: log return over the window.
double momentum(std::span<const double> returns);

/// Maximum drawdown within the window, on close prices, in log space:
///
///     min over j of  ln( C_j / max_{i<=j} C_i )
///
/// Result is <= 0; 0 means the window never traded below a prior peak.
/// Log space keeps units consistent with momentum (drawdown and recovery of
/// equal magnitude cancel).
/// Assumes: closes.size() >= 1, all closes > 0.
double max_drawdown(std::span<const double> closes);

/// Relative volume: ln( mean(win_vol) / mean(base_vol) ).
///
/// win_vol is the feature window's volumes (20 days ending at t), base_vol a
/// LONGER trailing baseline (252 days ending at t). The ratio removes the
/// secular volume trend — measured on SPY: median annual volume 138k (1993)
/// -> 257M (2008, 1856x) -> 68M (2025), i.e. huge and NON-monotonic, so only
/// an adaptive rolling baseline (not a parametric detrend) can absorb it.
/// The log tames the fat right tail (p99.9/median ~ 11x, panic-day spikes)
/// and makes the feature ~0 in normal regimes, positive in panic.
///
/// Units: log-ratio, dimensionless. No lookahead: both spans are trailing
/// and end at day t (asserted by the caller).
/// Assumes: both spans non-empty, volumes >= 0 with positive means; throws
/// std::invalid_argument on a non-positive mean (dead market data).
double relative_volume(std::span<const double> win_vol,
                       std::span<const double> base_vol);

/// Sample skewness of the given log returns, with bias correction
/// (adjusted Fisher–Pearson, the n^2/((n-1)(n-2)) factor — matches
/// scipy.stats.skew(bias=False)). The correction matters at n=20.
///
/// Degenerate contract: if the sample variance is ~0 (constant returns),
/// skewness is undefined; this returns 0.0 by convention.
/// Units: dimensionless.
/// Assumes: returns.size() >= 3.
double skewness(std::span<const double> returns);

// ---------------------------------------------------------------------------

/// Computes the full feature matrix over a trailing window of `window` log
/// returns per day.
///
/// Row k of the result corresponds to day t = window + k of `series` (the
/// first day with `window` returns of history), and dates()[k] ==
/// series.dates[window + k].
///
/// NO-LOOKAHEAD GUARANTEE: row k is computed exclusively from
/// series.close[t - window .. t] with t = window + k — i.e. information up to
/// and including day t. This is enforced structurally: the only data reaching
/// the kernels are spans whose last element is index t. Verified end-to-end
/// by the truncation test in tests/features_test.cpp (truncating the series
/// after day t must not change row k, bit for bit).
///
/// Throws std::invalid_argument if series.size() < window + 1 or window < 3.
FeatureMatrix compute_features(const OhlcvSeries& series,
                               std::size_t window = kDefaultWindow);

/// Phase B feature-set configuration. The 4-feature default of the plain
/// overload above is preserved bit-for-bit (d=4 stays reproducible).
struct FeatureParams {
    std::size_t window = kDefaultWindow;
    /// Adds Feature::RelVolume as a 5th column (d=5).
    bool include_volume = false;
    /// Trailing baseline length for relative_volume, in trading days.
    /// 252 = one trading year: long vs the 20d window, adaptive vs the
    /// secular trend. A judgment call (unlike W) — tunable, not tuned.
    std::size_t volume_baseline = 252;
    /// Forces the first feature row to a later day index (0 = automatic:
    /// max(window, volume_baseline - 1) when volume is on, window otherwise).
    /// Used to align a d=4 run onto the d=5 sample for controlled
    /// comparison — same days, only the feature set differs.
    std::size_t min_first_day = 0;
};

/// As compute_features above, with a configurable feature set. Row k maps to
/// day t = first_day + k where first_day = max(window,
/// include_volume ? volume_baseline - 1 : 0, min_first_day). The
/// no-lookahead guarantee covers every column: all spans (returns, closes,
/// window volume, baseline volume) end at day t. Verified by the truncation
/// test for both d=4 and d=5 parameter sets.
FeatureMatrix compute_features(const OhlcvSeries& series,
                               const FeatureParams& params);

}  // namespace mrd
