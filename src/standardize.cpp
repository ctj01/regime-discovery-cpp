// Copyright (c) 2026 Cristian Mendoza <openbeta.finance>
// SPDX-License-Identifier: AGPL-3.0-or-later
//
// This file is part of regime-discovery-cpp. It is free software: you can
// redistribute it and/or modify it under the terms of the GNU Affero General
// Public License as published by the Free Software Foundation, either version
// 3 of the License, or (at your option) any later version. See LICENSE.

#include "mrd/standardize.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>

namespace mrd {

namespace {

// Variance threshold below which a feature is considered degenerate. Raw
// features live at ~1e-2 (returns) to ~1e0 (skewness); 1e-12 std is
// unambiguously "constant".
constexpr double kMinScale = 1e-12;

void check_fitted(std::size_t fitted_dim, const FeatureMatrix& raw) {
    if (fitted_dim == 0) {
        throw std::logic_error("Standardizer::transform called before fit");
    }
    if (raw.dim() != fitted_dim) {
        throw std::logic_error(
            "Standardizer::transform dim mismatch: fitted " +
            std::to_string(fitted_dim) + ", got " + std::to_string(raw.dim()));
    }
}

/// Applies out = (in - center) / scale element-wise per feature. Shared by
/// both implementations: the derived matrix copies the aligned dates through
/// and never touches the input.
FeatureMatrix affine_transform(const FeatureMatrix& raw,
                               std::span<const double> center,
                               std::span<const double> scale) {
    FeatureMatrix out(raw.n_rows(), raw.dim());
    out.dates() = raw.dates();
    for (std::size_t t = 0; t < raw.n_rows(); ++t) {
        const auto in = raw.row(t);
        const auto o = out.row(t);
        for (std::size_t j = 0; j < raw.dim(); ++j) {
            o[j] = (in[j] - center[j]) / scale[j];
        }
    }
    return out;
}

/// Quantile with linear interpolation over a SORTED span (numpy default
/// method): position h = (n-1)q, result = x[floor(h)] + frac(h) * (x[floor(h)+1]
/// - x[floor(h)]).
double quantile_sorted(std::span<const double> sorted, double q) {
    const double h = static_cast<double>(sorted.size() - 1) * q;
    const auto lo = static_cast<std::size_t>(h);
    if (lo + 1 >= sorted.size()) return sorted.back();
    const double frac = h - static_cast<double>(lo);
    return sorted[lo] + frac * (sorted[lo + 1] - sorted[lo]);
}

}  // namespace

void ZScoreGlobal::fit(const FeatureMatrix& raw) {
    if (raw.n_rows() < 2) {
        throw std::runtime_error("ZScoreGlobal::fit needs at least 2 rows");
    }
    const std::size_t d = raw.dim();
    const auto n = static_cast<double>(raw.n_rows());
    mu_.assign(d, 0.0);
    sigma_.assign(d, 0.0);

    for (std::size_t t = 0; t < raw.n_rows(); ++t) {
        const auto row = raw.row(t);
        for (std::size_t j = 0; j < d; ++j) mu_[j] += row[j];
    }
    for (std::size_t j = 0; j < d; ++j) mu_[j] /= n;

    for (std::size_t t = 0; t < raw.n_rows(); ++t) {
        const auto row = raw.row(t);
        for (std::size_t j = 0; j < d; ++j) {
            const double dev = row[j] - mu_[j];
            sigma_[j] += dev * dev;
        }
    }
    for (std::size_t j = 0; j < d; ++j) {
        sigma_[j] = std::sqrt(sigma_[j] / (n - 1.0));  // ddof=1
        if (sigma_[j] < kMinScale) {
            mu_.clear();
            sigma_.clear();
            throw std::runtime_error("ZScoreGlobal::fit: feature " +
                                     std::to_string(j) +
                                     " has ~zero variance (dead dimension)");
        }
    }
}

FeatureMatrix ZScoreGlobal::transform(const FeatureMatrix& raw) const {
    check_fitted(mu_.size(), raw);
    return affine_transform(raw, mu_, sigma_);
}

void ZScoreRobust::fit(const FeatureMatrix& raw) {
    if (raw.n_rows() < 2) {
        throw std::runtime_error("ZScoreRobust::fit needs at least 2 rows");
    }
    const std::size_t d = raw.dim();
    med_.assign(d, 0.0);
    scale_.assign(d, 0.0);

    std::vector<double> col(raw.n_rows());
    for (std::size_t j = 0; j < d; ++j) {
        for (std::size_t t = 0; t < raw.n_rows(); ++t) {
            col[t] = raw(t, static_cast<Feature>(j));
        }
        std::sort(col.begin(), col.end());
        med_[j] = quantile_sorted(col, 0.5);
        const double iqr =
            quantile_sorted(col, 0.75) - quantile_sorted(col, 0.25);
        scale_[j] = iqr / kIqrToSigma;
        if (scale_[j] < kMinScale) {
            med_.clear();
            scale_.clear();
            throw std::runtime_error("ZScoreRobust::fit: feature " +
                                     std::to_string(j) +
                                     " has ~zero IQR (dead dimension)");
        }
    }
}

FeatureMatrix ZScoreRobust::transform(const FeatureMatrix& raw) const {
    check_fitted(med_.size(), raw);
    return affine_transform(raw, med_, scale_);
}

std::unique_ptr<Standardizer> make_standardizer(std::string_view name) {
    if (name == "zscore") return std::make_unique<ZScoreGlobal>();
    if (name == "robust") return std::make_unique<ZScoreRobust>();
    throw std::invalid_argument("unknown standardizer '" + std::string(name) +
                                "' (expected: zscore | robust)");
}

}  // namespace mrd
