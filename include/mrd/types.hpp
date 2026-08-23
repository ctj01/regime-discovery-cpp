// Copyright (c) 2026 Cristian Mendoza <openbeta.finance>
// SPDX-License-Identifier: AGPL-3.0-or-later
//
// This file is part of regime-discovery-cpp. It is free software: you can
// redistribute it and/or modify it under the terms of the GNU Affero General
// Public License as published by the Free Software Foundation, either version
// 3 of the License, or (at your option) any later version. See LICENSE.

#pragma once

#include <array>
#include <cassert>
#include <cstddef>
#include <span>
#include <string>
#include <vector>

namespace mrd {

/// ISO-8601 calendar date "YYYY-MM-DD". Lexicographic order == chronological
/// order, which is the only property the core relies on (no date arithmetic).
using Date = std::string;

/// Daily OHLCV series, struct-of-arrays.
///
/// Invariants (enforced by the CSV loader, assumed everywhere else):
///  - all vectors have equal length
///  - dates are strictly increasing (chronological, no duplicates)
///  - prices are > 0 (log returns must be well-defined)
struct OhlcvSeries {
    std::vector<Date>   dates;
    std::vector<double> open;
    std::vector<double> high;
    std::vector<double> low;
    std::vector<double> close;
    std::vector<double> volume;

    [[nodiscard]] std::size_t size() const noexcept { return dates.size(); }
};

/// Feature indices into a FeatureMatrix row. Order here defines the column
/// order of the matrix and of any CSV dump — keep kFeatureNames in sync.
enum class Feature : std::size_t {
    RealizedVol,
    Momentum,
    MaxDrawdown,
    Skewness,
    RelVolume,  ///< Phase B: only present when FeatureParams::include_volume
    Count
};

inline constexpr std::size_t kNumFeatures = static_cast<std::size_t>(Feature::Count);

inline constexpr std::array<const char*, kNumFeatures> kFeatureNames = {
    "realized_vol",
    "momentum",
    "max_drawdown",
    "skewness",
    "rel_volume",
};

/// Row-major N×d feature matrix with an aligned date vector.
///
/// Layout contract: row t is the contiguous d-length block data()[t*d .. t*d+d),
/// so row(t).data() can be handed directly to an HNSW index as the point
/// coordinates for day dates()[t] — no copy, no gather.
///
/// dates()[t] is the *last* day of the window that produced row t: row t uses
/// information up to and including dates()[t] and nothing after it.
class FeatureMatrix {
public:
    FeatureMatrix(std::size_t n_rows, std::size_t dim)
        : dim_(dim), data_(n_rows * dim), dates_(n_rows) {}

    [[nodiscard]] std::size_t n_rows() const noexcept { return dates_.size(); }
    [[nodiscard]] std::size_t dim() const noexcept { return dim_; }

    [[nodiscard]] std::span<const double> row(std::size_t t) const noexcept {
        assert(t < n_rows());
        return {data_.data() + t * dim_, dim_};
    }
    [[nodiscard]] std::span<double> row(std::size_t t) noexcept {
        assert(t < n_rows());
        return {data_.data() + t * dim_, dim_};
    }

    [[nodiscard]] double operator()(std::size_t t, Feature f) const noexcept {
        assert(t < n_rows() && static_cast<std::size_t>(f) < dim_);
        return data_[t * dim_ + static_cast<std::size_t>(f)];
    }
    [[nodiscard]] double& operator()(std::size_t t, Feature f) noexcept {
        assert(t < n_rows() && static_cast<std::size_t>(f) < dim_);
        return data_[t * dim_ + static_cast<std::size_t>(f)];
    }

    [[nodiscard]] const double* data() const noexcept { return data_.data(); }

    [[nodiscard]] const std::vector<Date>& dates() const noexcept { return dates_; }
    [[nodiscard]] std::vector<Date>& dates() noexcept { return dates_; }

private:
    std::size_t         dim_;
    std::vector<double> data_;   // row-major, n_rows * dim_
    std::vector<Date>   dates_;  // dates_[t] <-> row t
};

}  // namespace mrd
