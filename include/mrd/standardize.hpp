// Copyright (c) 2026 Cristian Mendoza <openbeta.finance>
// SPDX-License-Identifier: AGPL-3.0-or-later
//
// This file is part of regime-discovery-cpp. It is free software: you can
// redistribute it and/or modify it under the terms of the GNU Affero General
// Public License as published by the Free Software Foundation, either version
// 3 of the License, or (at your option) any later version. See LICENSE.

#pragma once

#include <memory>
#include <span>
#include <string_view>
#include <vector>

#include "mrd/types.hpp"

namespace mrd {

/// Whole-matrix standardization primitive.
///
/// Granularity rationale: whole-matrix is the only interface that can express
/// both global and (future) causal/expanding modes uniformly — a stateless
/// per-row apply cannot represent position-dependent causal standardization.
///
/// Contract shared by all implementations:
///  - fit() learns and freezes parameters from the given matrix; calling it
///    again refits (previous parameters are discarded).
///  - transform() produces a NEW derived matrix (same layout contract, dates
///    copied through); the input is never mutated. The input may have a
///    different row count than the fit matrix, but must have the same dim.
///  - transform() before fit(), or with a dim mismatch, throws
///    std::logic_error (programming error, not a data error).
class Standardizer {
public:
    virtual ~Standardizer() = default;

    /// Learn & freeze parameters from historical data. No-op for stateless
    /// causal modes. Throws std::runtime_error on degenerate data (see impls).
    virtual void fit(const FeatureMatrix& raw) = 0;

    /// Batch transform for index building. Universal primitive; may be
    /// sequential internally (causal modes).
    [[nodiscard]] virtual FeatureMatrix transform(const FeatureMatrix& raw) const = 0;

    // Deferred (second act — live single-row apply against frozen params):
    // virtual void transform_row(std::span<const double> in,
    //                            std::span<double> out) const = 0;
};

/// Per-feature z-score (x − μ)/σ with μ, σ (ddof=1, consistent with the
/// Phase 1 realized_volatility convention) frozen at fit() over the FULL
/// matrix.
///
/// GLOBAL LOOKAHEAD by construction: every row is scaled by full-sample
/// moments, so row t depends on data after t. Legitimate for the current
/// descriptive goal (regimes defined relative to the full-sample
/// distribution); NOT valid for a live/predictive setting, which requires a
/// causal Standardizer. Recorded in README decision record.
///
/// fit() throws std::runtime_error if any feature has ~zero variance — a
/// constant feature over decades of data is a data bug, and silently mapping
/// it to 0 would hide a dead dimension inside the index metric.
class ZScoreGlobal final : public Standardizer {
public:
    void fit(const FeatureMatrix& raw) override;
    [[nodiscard]] FeatureMatrix transform(const FeatureMatrix& raw) const override;

    /// Frozen parameters (size d), for inspection/dump. Empty before fit().
    [[nodiscard]] std::span<const double> mean() const noexcept { return mu_; }
    [[nodiscard]] std::span<const double> stddev() const noexcept { return sigma_; }

private:
    std::vector<double> mu_;
    std::vector<double> sigma_;
};

/// Robust per-feature standardization: (x − median) / (IQR / 1.34898).
///
/// Motivated by the measured fat tail in skewness (min −3.36 vs p1 −1.85):
/// classic z-score lets those outliers inflate σ and compress the bulk.
/// The 1.34898 = 2·Φ⁻¹(0.75) factor makes IQR a consistent estimator of σ
/// under normality, so robust and classic outputs live on comparable scales.
///
/// Quantiles use linear interpolation (numpy default) so results are
/// bit-comparable against a numpy reference. fit() throws std::runtime_error
/// if any feature has ~zero IQR (same dead-dimension rationale as ZScoreGlobal).
class ZScoreRobust final : public Standardizer {
public:
    void fit(const FeatureMatrix& raw) override;
    [[nodiscard]] FeatureMatrix transform(const FeatureMatrix& raw) const override;

    /// Frozen parameters (size d), for inspection/dump. Empty before fit().
    [[nodiscard]] std::span<const double> median() const noexcept { return med_; }
    [[nodiscard]] std::span<const double> scale() const noexcept { return scale_; }

private:
    std::vector<double> med_;
    std::vector<double> scale_;  // IQR / 1.34898, per feature
};

/// IQR-to-sigma consistency factor: 2·Φ⁻¹(0.75).
inline constexpr double kIqrToSigma = 1.3489795003921634;

/// Config-driven factory: "zscore" -> ZScoreGlobal, "robust" -> ZScoreRobust.
/// Throws std::invalid_argument on unknown name.
std::unique_ptr<Standardizer> make_standardizer(std::string_view name);

}  // namespace mrd
