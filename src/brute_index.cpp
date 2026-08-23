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
#include <stdexcept>
#include <utility>

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

}  // namespace

void BruteForceIndex::build(const FeatureMatrix& data) {
    n_ = data.n_rows();
    d_ = data.dim();
    data_.assign(data.data(), data.data() + n_ * d_);
}

std::vector<Neighbor> BruteForceIndex::query(std::span<const double> q, int k,
                                             std::optional<std::size_t> exclude) const {
    if (n_ == 0) throw std::logic_error("BruteForceIndex::query before build");
    if (k <= 0) throw std::invalid_argument("k must be positive");
    if (q.size() != d_) throw std::logic_error("query dim mismatch");

    // (squared dist, index) pairs: pair ordering gives ascending distance
    // with ties broken by ascending index — deterministic by construction.
    std::vector<std::pair<double, std::size_t>> best;
    best.reserve(n_);
    for (std::size_t i = 0; i < n_; ++i) {
        if (exclude && *exclude == i) continue;
        best.emplace_back(dist2(q.data(), data_.data() + i * d_, d_), i);
    }

    const std::size_t kk = std::min(static_cast<std::size_t>(k), best.size());
    std::nth_element(best.begin(),
                     best.begin() + static_cast<std::ptrdiff_t>(kk), best.end());
    std::sort(best.begin(), best.begin() + static_cast<std::ptrdiff_t>(kk));

    std::vector<Neighbor> out;
    out.reserve(kk);
    for (std::size_t i = 0; i < kk; ++i) {
        out.push_back({best[i].second, std::sqrt(best[i].first)});
    }
    return out;
}

}  // namespace mrd
