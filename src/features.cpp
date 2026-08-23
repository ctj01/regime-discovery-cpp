// Copyright (c) 2026 Cristian Mendoza <openbeta.finance>
// SPDX-License-Identifier: AGPL-3.0-or-later
//
// This file is part of regime-discovery-cpp. It is free software: you can
// redistribute it and/or modify it under the terms of the GNU Affero General
// Public License as published by the Free Software Foundation, either version
// 3 of the License, or (at your option) any later version. See LICENSE.

#include "mrd/features.hpp"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <numeric>
#include <stdexcept>
#include <vector>

namespace mrd {

double realized_volatility(std::span<const double> returns) {
    assert(returns.size() >= 2);
    const auto n = static_cast<double>(returns.size());
    const double mean =
        std::accumulate(returns.begin(), returns.end(), 0.0) / n;
    double ss = 0.0;
    for (double r : returns) {
        const double d = r - mean;
        ss += d * d;
    }
    return std::sqrt(ss / (n - 1.0));
}

double momentum(std::span<const double> returns) {
    return std::accumulate(returns.begin(), returns.end(), 0.0);
}

double max_drawdown(std::span<const double> closes) {
    assert(!closes.empty());
    double peak = closes[0];
    double mdd = 0.0;  // log drawdown, <= 0
    for (double c : closes) {
        peak = std::max(peak, c);
        mdd = std::min(mdd, std::log(c / peak));
    }
    return mdd;
}

double skewness(std::span<const double> returns) {
    assert(returns.size() >= 3);
    const auto n = static_cast<double>(returns.size());
    const double mean =
        std::accumulate(returns.begin(), returns.end(), 0.0) / n;
    double m2 = 0.0, m3 = 0.0;
    for (double r : returns) {
        const double d = r - mean;
        m2 += d * d;
        m3 += d * d * d;
    }
    m2 /= n;
    m3 /= n;
    // Guard the degenerate (constant-returns) window; threshold is relative
    // to nothing in particular — daily log returns live at ~1e-2, so 1e-16
    // variance is unambiguously "no variation".
    if (m2 < 1e-16) return 0.0;
    const double g1 = m3 / std::pow(m2, 1.5);
    return g1 * std::sqrt(n * (n - 1.0)) / (n - 2.0);
}

double relative_volume(std::span<const double> win_vol,
                       std::span<const double> base_vol) {
    auto mean = [](std::span<const double> v) {
        return std::accumulate(v.begin(), v.end(), 0.0) /
               static_cast<double>(v.size());
    };
    const double w = mean(win_vol);
    const double b = mean(base_vol);
    if (w <= 0.0 || b <= 0.0) {
        throw std::invalid_argument("relative_volume: non-positive mean volume");
    }
    return std::log(w / b);
}

FeatureMatrix compute_features(const OhlcvSeries& series, std::size_t window) {
    return compute_features(series, FeatureParams{window, false, 252, 0});
}

FeatureMatrix compute_features(const OhlcvSeries& series,
                               const FeatureParams& params) {
    const std::size_t window = params.window;
    if (window < 3) {
        throw std::invalid_argument("window must be >= 3 (skewness needs 3 samples)");
    }

    // First day with full history for every column: `window` closes behind it
    // for the return window, `volume_baseline` volumes ending at it when the
    // volume feature is on, and never earlier than min_first_day.
    std::size_t first_day = window;
    if (params.include_volume) {
        if (params.volume_baseline <= window) {
            throw std::invalid_argument("volume_baseline must exceed window");
        }
        first_day = std::max(first_day, params.volume_baseline - 1);
    }
    first_day = std::max(first_day, params.min_first_day);
    if (series.size() < first_day + 1) {
        throw std::invalid_argument("series shorter than warm-up + 1 days");
    }

    const std::size_t n_days = series.size();
    const std::size_t n_rows = n_days - first_day;
    const std::size_t dim = params.include_volume ? kNumFeatures : kNumFeatures - 1;
    FeatureMatrix m(n_rows, dim);

    // Log returns for the whole series: returns[i] = ln(close[i+1]/close[i]),
    // so returns[i] is known at the end of day i+1.
    std::vector<double> returns(n_days - 1);
    for (std::size_t i = 0; i + 1 < n_days; ++i) {
        returns[i] = std::log(series.close[i + 1] / series.close[i]);
    }

    const std::span<const double> all_returns(returns);
    const std::span<const double> all_closes(series.close);
    const std::span<const double> all_volume(series.volume);

    for (std::size_t k = 0; k < n_rows; ++k) {
        const std::size_t t = first_day + k;  // day index of this row

        // NO-LOOKAHEAD: every span ends exactly at day t.
        // ret_win  = returns[t-window .. t-1]  (last one is ln(C_t/C_{t-1}))
        // close_win = close[t-window .. t]
        const auto ret_win   = all_returns.subspan(t - window, window);
        const auto close_win = all_closes.subspan(t - window, window + 1);
        assert(&ret_win.back() == &returns[t - 1]);
        assert(&close_win.back() == &series.close[t]);

        m(k, Feature::RealizedVol) = realized_volatility(ret_win);
        m(k, Feature::Momentum)    = momentum(ret_win);
        m(k, Feature::MaxDrawdown) = max_drawdown(close_win);
        m(k, Feature::Skewness)    = skewness(ret_win);
        if (params.include_volume) {
            // vol_win  = volume[t-window+1 .. t]        (window days)
            // vol_base = volume[t-baseline+1 .. t]      (baseline days)
            const auto vol_win = all_volume.subspan(t - window + 1, window);
            const auto vol_base =
                all_volume.subspan(t - params.volume_baseline + 1,
                                   params.volume_baseline);
            assert(&vol_win.back() == &series.volume[t]);
            assert(&vol_base.back() == &series.volume[t]);
            m(k, Feature::RelVolume) = relative_volume(vol_win, vol_base);
        }
        m.dates()[k] = series.dates[t];
    }
    return m;
}

}  // namespace mrd
