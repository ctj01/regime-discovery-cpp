// Copyright (c) 2026 Cristian Mendoza <openbeta.finance>
// SPDX-License-Identifier: AGPL-3.0-or-later
//
// This file is part of regime-discovery-cpp. It is free software: you can
// redistribute it and/or modify it under the terms of the GNU Affero General
// Public License as published by the Free Software Foundation, either version
// 3 of the License, or (at your option) any later version. See LICENSE.

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"

#include <cmath>
#include <cstdio>
#include <fstream>
#include <string>
#include <vector>

#include "mrd/csv_loader.hpp"
#include "mrd/features.hpp"

namespace {

// Reference fixture: 20 draws, N(-0.001, 0.015), numpy default_rng(42).
// Expected values computed with numpy/scipy (std ddof=1, skew bias=False).
const std::vector<double> kRefReturns = {
    0.0035707561963164705,   -0.016599761593607432,  0.010256767937096858,
    0.013108470745868208,    -0.030265527829807546,  -0.020532692602934773,
    0.00091760604750928043,  -0.005743638885153733,  -0.0012520173625643319,
    -0.013795658913603702,   0.012190969622942426,   0.010666879031434224,
    -9.5395365817593086e-06, 0.015908618104520492,   0.0060126401337806841,
    -0.013889386943248573,   0.0045312617612374827,  -0.015383239012434984,
    0.012176754519609086,    -0.0017488886647937935};
constexpr double kRefStd  = 0.013052667346011494;
constexpr double kRefSkew = -0.60418734748994118;
constexpr double kRefSum  = -0.029879627244415407;

// Deterministic synthetic OHLCV series (LCG-driven returns). Dates are
// synthetic monotonic strings — the core only needs strict ordering.
mrd::OhlcvSeries make_series(std::size_t n_days, std::uint64_t seed = 1) {
    mrd::OhlcvSeries s;
    std::uint64_t x = seed;
    double close = 100.0;
    for (std::size_t i = 0; i < n_days; ++i) {
        x = x * 6364136223846793005ULL + 1442695040888963407ULL;
        // uniform in [-0.03, 0.03)
        const double r =
            (static_cast<double>(x >> 11) / 9007199254740992.0 - 0.5) * 0.06;
        if (i > 0) close *= std::exp(r);
        char buf[32];
        std::snprintf(buf, sizeof(buf), "2020-%02zu-%02zu", 1 + i / 28, 1 + i % 28);
        s.dates.emplace_back(buf);
        s.open.push_back(close * 0.999);
        s.high.push_back(close * 1.005);
        s.low.push_back(close * 0.995);
        s.close.push_back(close);
        s.volume.push_back(1e6 + static_cast<double>(i));
    }
    return s;
}

}  // namespace

TEST_CASE("realized_volatility matches numpy std ddof=1") {
    CHECK(mrd::realized_volatility(kRefReturns) ==
          doctest::Approx(kRefStd).epsilon(1e-12));
    const std::vector<double> flat(20, 0.004);
    CHECK(mrd::realized_volatility(flat) == doctest::Approx(0.0).epsilon(1e-15));
}

TEST_CASE("momentum is the sum of window returns") {
    CHECK(mrd::momentum(kRefReturns) == doctest::Approx(kRefSum).epsilon(1e-12));
}

TEST_CASE("skewness matches scipy bias-corrected; constant window -> 0") {
    CHECK(mrd::skewness(kRefReturns) ==
          doctest::Approx(kRefSkew).epsilon(1e-12));
    const std::vector<double> flat(20, 0.004);
    CHECK(mrd::skewness(flat) == 0.0);
}

TEST_CASE("max_drawdown: log-space, <= 0, tracks running peak") {
    const std::vector<double> up = {100, 101, 102, 105, 110};
    CHECK(mrd::max_drawdown(up) == 0.0);

    const std::vector<double> dip = {100, 110, 99, 105, 120, 90};
    // worst: trough 90 against peak 120 (beats 99 vs 110)
    CHECK(mrd::max_drawdown(dip) ==
          doctest::Approx(std::log(90.0 / 120.0)).epsilon(1e-15));
}

TEST_CASE("compute_features: shape, alignment, momentum telescopes") {
    const std::size_t w = 20;
    const auto s = make_series(60);
    const auto m = mrd::compute_features(s, w);

    CHECK(m.n_rows() == s.size() - w);
    CHECK(m.dim() == 4);  // canonical d=4: kNumFeatures minus rel_volume
    for (std::size_t k = 0; k < m.n_rows(); ++k) {
        const std::size_t t = w + k;
        CHECK(m.dates()[k] == s.dates[t]);
        CHECK(m(k, mrd::Feature::Momentum) ==
              doctest::Approx(std::log(s.close[t] / s.close[t - w])).epsilon(1e-12));
        CHECK(m(k, mrd::Feature::MaxDrawdown) <= 0.0);
        CHECK(m(k, mrd::Feature::RealizedVol) > 0.0);
    }
}

TEST_CASE("no lookahead: truncating the series after day t leaves row t bit-identical") {
    // Both feature sets: the loop over dim() covers the rel_volume column
    // automatically once params include it.
    const mrd::FeatureParams param_sets[] = {
        {20, false, 252, 0},   // d=4, warm-up 20
        {20, true, 60, 0},     // d=5, short baseline so the test stays fast
    };
    for (const auto& fp : param_sets) {
        const auto full = make_series(100);
        const auto m_full = mrd::compute_features(full, fp);
        const std::size_t first = full.size() - m_full.n_rows();

        for (std::size_t t = first; t < full.size(); ++t) {
            mrd::OhlcvSeries trunc;
            trunc.dates.assign(full.dates.begin(), full.dates.begin() + (t + 1));
            trunc.open.assign(full.open.begin(), full.open.begin() + (t + 1));
            trunc.high.assign(full.high.begin(), full.high.begin() + (t + 1));
            trunc.low.assign(full.low.begin(), full.low.begin() + (t + 1));
            trunc.close.assign(full.close.begin(), full.close.begin() + (t + 1));
            trunc.volume.assign(full.volume.begin(), full.volume.begin() + (t + 1));

            const auto m_trunc = mrd::compute_features(trunc, fp);
            const std::size_t k = t - first;
            REQUIRE(m_trunc.n_rows() == k + 1);
            const auto row_full  = m_full.row(k);
            const auto row_trunc = m_trunc.row(k);
            REQUIRE(row_full.size() == row_trunc.size());
            for (std::size_t j = 0; j < row_full.size(); ++j) {
                CHECK(row_full[j] == row_trunc[j]);  // exact, not approx
            }
            CHECK(m_full.dates()[k] == m_trunc.dates()[k]);
        }
    }
}

TEST_CASE("relative_volume kernel: known ratios, contract violations") {
    const std::vector<double> flat(20, 5e6);
    const std::vector<double> base(60, 5e6);
    CHECK(mrd::relative_volume(flat, base) == 0.0);  // ln(1) exactly

    const std::vector<double> doubled(20, 1e7);
    CHECK(mrd::relative_volume(doubled, base) ==
          doctest::Approx(std::log(2.0)).epsilon(1e-15));

    const std::vector<double> zeros(20, 0.0);
    CHECK_THROWS_AS(mrd::relative_volume(zeros, base), std::invalid_argument);
    // one zero-volume day inside an otherwise live window is fine
    std::vector<double> one_zero(20, 5e6);
    one_zero[7] = 0.0;
    CHECK(mrd::relative_volume(one_zero, base) ==
          doctest::Approx(std::log(19.0 / 20.0)).epsilon(1e-12));
}

TEST_CASE("d=5 feature set: alignment, warm-up, canonical d=4 unchanged") {
    const auto s = make_series(320);

    const auto d4 = mrd::compute_features(s, 20);
    const auto d5 = mrd::compute_features(s, {20, true, 252, 0});

    CHECK(d5.dim() == 5);
    CHECK(d5.n_rows() == s.size() - 251);
    CHECK(d5.dates()[0] == s.dates[251]);

    // The 4 shared columns must be bit-identical on the common dates.
    const std::size_t off = d4.n_rows() - d5.n_rows();
    for (std::size_t k = 0; k < d5.n_rows(); ++k) {
        REQUIRE(d4.dates()[k + off] == d5.dates()[k]);
        for (std::size_t j = 0; j < 4; ++j) {
            CHECK(d4.row(k + off)[j] == d5.row(k)[j]);
        }
    }

    // min_first_day aligns a d=4 run onto the d=5 sample.
    const auto d4a = mrd::compute_features(s, {20, false, 252, 251});
    CHECK(d4a.n_rows() == d5.n_rows());
    CHECK(d4a.dates() == d5.dates());
    CHECK(d4a.dim() == 4);

    // rel_volume sanity on the synthetic series: volume is ~flat by
    // construction (1e6 + i), so the feature must hover near 0.
    for (std::size_t k = 0; k < d5.n_rows(); k += 7) {
        CHECK(std::abs(d5(k, mrd::Feature::RelVolume)) < 0.01);
    }

    CHECK_THROWS_AS(mrd::compute_features(s, {20, true, 20, 0}),
                    std::invalid_argument);  // baseline must exceed window
}

TEST_CASE("compute_features rejects short series and tiny windows") {
    const auto s = make_series(15);
    CHECK_THROWS_AS(mrd::compute_features(s, 20), std::invalid_argument);
    CHECK_THROWS_AS(mrd::compute_features(s, 2), std::invalid_argument);
}

TEST_CASE("csv loader: round trip and validation") {
    const std::string path = "test_ohlcv_tmp.csv";
    {
        std::ofstream out(path);
        out << "date,open,high,low,close,volume\n"
            << "2024-01-02,100.1,101.0,99.5,100.5,1000000\n"
            << "2024-01-03,100.6,102.0,100.2,101.8,1200000\n";
    }
    const auto s = mrd::load_ohlcv_csv(path);
    CHECK(s.size() == 2);
    CHECK(s.dates[1] == "2024-01-03");
    CHECK(s.close[0] == doctest::Approx(100.5));
    CHECK(s.volume[1] == doctest::Approx(1200000));

    {
        std::ofstream out(path);
        out << "date,open,high,low,close,volume\n"
            << "2024-01-03,100,101,99,100,1\n"
            << "2024-01-02,100,101,99,100,1\n";  // out of order
    }
    CHECK_THROWS(mrd::load_ohlcv_csv(path));

    {
        std::ofstream out(path);
        out << "date,open,high,low,close,volume\n"
            << "2024-01-02,100,101,99,-5,1\n";  // negative close
    }
    CHECK_THROWS(mrd::load_ohlcv_csv(path));

    std::remove(path.c_str());
}
