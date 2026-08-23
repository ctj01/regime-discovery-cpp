// Copyright (c) 2026 Cristian Mendoza <openbeta.finance>
// SPDX-License-Identifier: AGPL-3.0-or-later
//
// This file is part of regime-discovery-cpp. It is free software: you can
// redistribute it and/or modify it under the terms of the GNU Affero General
// Public License as published by the Free Software Foundation, either version
// 3 of the License, or (at your option) any later version. See LICENSE.

#include "mrd/regime_validate.hpp"

#include <algorithm>
#include <cmath>
#include <map>
#include <stdexcept>

#include "mrd/features.hpp"

namespace mrd {

std::vector<RunStats> persistence(const std::vector<int>& labels) {
    std::map<int, RunStats> acc;
    std::size_t i = 0;
    while (i < labels.size()) {
        std::size_t j = i;
        while (j < labels.size() && labels[j] == labels[i]) ++j;
        auto& s = acc.try_emplace(labels[i], RunStats{labels[i], 0, 0, 0.0}).first->second;
        s.n_days += j - i;
        s.n_runs += 1;
        i = j;
    }
    std::vector<RunStats> out;
    out.reserve(acc.size());
    for (auto& [label, s] : acc) {
        (void)label;
        s.mean_run_length =
            static_cast<double>(s.n_days) / static_cast<double>(s.n_runs);
        out.push_back(s);
    }
    return out;  // std::map iteration => ascending by label
}

std::vector<EpisodeStats> recurrence(const std::vector<int>& labels,
                                     std::size_t gap) {
    std::map<int, EpisodeStats> acc;
    std::map<int, std::size_t> last_seen;
    for (std::size_t t = 0; t < labels.size(); ++t) {
        const int lab = labels[t];
        if (lab < 0) continue;
        auto [it, inserted] =
            acc.try_emplace(lab, EpisodeStats{lab, 0, 1, t, t});
        auto& s = it->second;
        if (!inserted && t - last_seen[lab] > gap) s.n_episodes += 1;
        s.n_days += 1;
        s.last_day = t;
        last_seen[lab] = t;
    }
    std::vector<EpisodeStats> out;
    out.reserve(acc.size());
    for (const auto& [label, s] : acc) {
        (void)label;
        out.push_back(s);
    }
    return out;
}

std::vector<ForwardVolStats> forward_vol_by_regime(
    const std::vector<int>& labels, const OhlcvSeries& series,
    std::size_t first_row_day, std::size_t horizon) {
    if (horizon < 2) throw std::invalid_argument("forward_vol: horizon must be >= 2");
    if (first_row_day + labels.size() > series.size()) {
        throw std::invalid_argument("forward_vol: labels/series misaligned");
    }

    // All log returns once; returns[i] = ln(close[i+1]/close[i]).
    std::vector<double> returns(series.size() - 1);
    for (std::size_t i = 0; i + 1 < series.size(); ++i) {
        returns[i] = std::log(series.close[i + 1] / series.close[i]);
    }

    std::map<int, std::vector<double>> vols;
    for (std::size_t t = 0; t < labels.size(); ++t) {
        const std::size_t T = t + first_row_day;
        if (T + horizon >= series.size()) break;  // incomplete forward window
        // forward returns r[T+1 .. T+horizon] == returns[T .. T+horizon-1]
        const std::span<const double> fwd(returns.data() + T, horizon);
        vols[labels[t]].push_back(realized_volatility(fwd));
    }

    std::vector<ForwardVolStats> out;
    out.reserve(vols.size());
    for (auto& [label, v] : vols) {
        std::sort(v.begin(), v.end());
        double mean = 0.0;
        for (const double x : v) mean += x;
        mean /= static_cast<double>(v.size());
        out.push_back({label, v.size(), mean, v[v.size() / 2]});
    }
    return out;
}

}  // namespace mrd
