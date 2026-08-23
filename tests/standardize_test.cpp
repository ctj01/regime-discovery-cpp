// Copyright (c) 2026 Cristian Mendoza <openbeta.finance>
// SPDX-License-Identifier: AGPL-3.0-or-later
//
// This file is part of regime-discovery-cpp. It is free software: you can
// redistribute it and/or modify it under the terms of the GNU Affero General
// Public License as published by the Free Software Foundation, either version
// 3 of the License, or (at your option) any later version. See LICENSE.

#include "doctest.h"

#include <cmath>
#include <stdexcept>
#include <vector>

#include "mrd/standardize.hpp"

namespace {

// Same reference fixture as features_test.cpp: 20 draws, N(-0.001, 0.015),
// numpy default_rng(42). Expected values computed with numpy (17 digits).
const std::vector<double> kRefData = {
    0.0035707561963164705,   -0.016599761593607432,  0.010256767937096858,
    0.013108470745868208,    -0.030265527829807546,  -0.020532692602934773,
    0.00091760604750928043,  -0.005743638885153733,  -0.0012520173625643319,
    -0.013795658913603702,   0.012190969622942426,   0.010666879031434224,
    -9.5395365817593086e-06, 0.015908618104520492,   0.0060126401337806841,
    -0.013889386943248573,   0.0045312617612374827,  -0.015383239012434984,
    0.012176754519609086,    -0.0017488886647937935};
constexpr double kRefMean   = -0.0014939813622207704;
constexpr double kRefStd    = 0.013052667346011494;   // ddof=1
constexpr double kRefMedian = 0.00045403325546376056;  // numpy linear interp
constexpr double kRefIqr    = 0.024178386631696119;    // q75 - q25

// dim=2 matrix: column 0 = fixture, column 1 = affine image of the fixture
// (2x + 5, exercises per-feature parameters).
mrd::FeatureMatrix make_matrix() {
    mrd::FeatureMatrix m(kRefData.size(), 2);
    for (std::size_t t = 0; t < kRefData.size(); ++t) {
        m.row(t)[0] = kRefData[t];
        m.row(t)[1] = 2.0 * kRefData[t] + 5.0;
        m.dates()[t] = "2024-01-" + std::string(t < 9 ? "0" : "") +
                       std::to_string(t + 1);
    }
    return m;
}

std::vector<double> snapshot(const mrd::FeatureMatrix& m) {
    return {m.data(), m.data() + m.n_rows() * m.dim()};
}

}  // namespace

TEST_CASE("ZScoreGlobal: frozen params match numpy; output is exact z-scores") {
    const auto raw = make_matrix();
    mrd::ZScoreGlobal z;
    z.fit(raw);

    CHECK(z.mean()[0] == doctest::Approx(kRefMean).epsilon(1e-14));
    CHECK(z.stddev()[0] == doctest::Approx(kRefStd).epsilon(1e-14));
    CHECK(z.mean()[1] == doctest::Approx(2.0 * kRefMean + 5.0).epsilon(1e-14));
    CHECK(z.stddev()[1] == doctest::Approx(2.0 * kRefStd).epsilon(1e-14));

    const auto out = z.transform(raw);
    REQUIRE(out.n_rows() == raw.n_rows());
    REQUIRE(out.dim() == raw.dim());
    CHECK(out.dates() == raw.dates());

    // Per-feature mean 0, std 1 (ddof=1) after transform, and the two
    // columns must be identical z-scores (affine invariance).
    for (std::size_t j = 0; j < out.dim(); ++j) {
        double mean = 0.0, ss = 0.0;
        for (std::size_t t = 0; t < out.n_rows(); ++t) {
            mean += out(t, static_cast<mrd::Feature>(j));
        }
        mean /= static_cast<double>(out.n_rows());
        for (std::size_t t = 0; t < out.n_rows(); ++t) {
            const double d = out(t, static_cast<mrd::Feature>(j)) - mean;
            ss += d * d;
        }
        const double sd =
            std::sqrt(ss / (static_cast<double>(out.n_rows()) - 1.0));
        CHECK(mean == doctest::Approx(0.0).epsilon(1e-14));
        CHECK(sd == doctest::Approx(1.0).epsilon(1e-14));
    }
    for (std::size_t t = 0; t < out.n_rows(); ++t) {
        CHECK(out.row(t)[0] == doctest::Approx(out.row(t)[1]).epsilon(1e-12));
        CHECK(out.row(t)[0] ==
              doctest::Approx((kRefData[t] - kRefMean) / kRefStd).epsilon(1e-13));
    }
}

TEST_CASE("ZScoreRobust: median/IQR match numpy linear-interp quantiles") {
    const auto raw = make_matrix();
    mrd::ZScoreRobust z;
    z.fit(raw);

    CHECK(z.median()[0] == doctest::Approx(kRefMedian).epsilon(1e-14));
    CHECK(z.scale()[0] ==
          doctest::Approx(kRefIqr / mrd::kIqrToSigma).epsilon(1e-14));
    CHECK(z.median()[1] == doctest::Approx(2.0 * kRefMedian + 5.0).epsilon(1e-14));

    const auto out = z.transform(raw);
    for (std::size_t t = 0; t < out.n_rows(); ++t) {
        const double expected =
            (kRefData[t] - kRefMedian) / (kRefIqr / mrd::kIqrToSigma);
        CHECK(out.row(t)[0] == doctest::Approx(expected).epsilon(1e-13));
        CHECK(out.row(t)[0] == doctest::Approx(out.row(t)[1]).epsilon(1e-12));
    }
}

TEST_CASE("transform never mutates the raw matrix (bitwise)") {
    const auto raw = make_matrix();
    const auto before = snapshot(raw);
    const auto dates_before = raw.dates();

    for (const char* name : {"zscore", "robust"}) {
        auto s = mrd::make_standardizer(name);
        s->fit(raw);
        const auto out = s->transform(raw);
        (void)out;
        const auto after = snapshot(raw);
        REQUIRE(after.size() == before.size());
        for (std::size_t i = 0; i < before.size(); ++i) {
            CHECK(before[i] == after[i]);  // exact, not approx
        }
        CHECK(raw.dates() == dates_before);
    }
}

TEST_CASE("contract violations throw") {
    const auto raw = make_matrix();

    SUBCASE("transform before fit -> logic_error") {
        mrd::ZScoreGlobal z;
        CHECK_THROWS_AS((void)z.transform(raw), std::logic_error);
        mrd::ZScoreRobust r;
        CHECK_THROWS_AS((void)r.transform(raw), std::logic_error);
    }

    SUBCASE("dim mismatch -> logic_error") {
        mrd::ZScoreGlobal z;
        z.fit(raw);
        mrd::FeatureMatrix other(5, 3);
        for (std::size_t t = 0; t < 5; ++t) {
            for (std::size_t j = 0; j < 3; ++j) {
                other.row(t)[j] = static_cast<double>(t + j);
            }
        }
        CHECK_THROWS_AS((void)z.transform(other), std::logic_error);
    }

    SUBCASE("constant feature -> runtime_error (dead dimension)") {
        mrd::FeatureMatrix degenerate(10, 2);
        for (std::size_t t = 0; t < 10; ++t) {
            degenerate.row(t)[0] = static_cast<double>(t);
            degenerate.row(t)[1] = 42.0;  // constant column
        }
        mrd::ZScoreGlobal z;
        CHECK_THROWS_AS(z.fit(degenerate), std::runtime_error);
        mrd::ZScoreRobust r;
        CHECK_THROWS_AS(r.fit(degenerate), std::runtime_error);
    }

    SUBCASE("unknown factory name -> invalid_argument") {
        CHECK_THROWS_AS(mrd::make_standardizer("whiten"), std::invalid_argument);
    }
}
