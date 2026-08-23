// Copyright (c) 2026 Cristian Mendoza <openbeta.finance>
// SPDX-License-Identifier: AGPL-3.0-or-later
//
// This file is part of regime-discovery-cpp. It is free software: you can
// redistribute it and/or modify it under the terms of the GNU Affero General
// Public License as published by the Free Software Foundation, either version
// 3 of the License, or (at your option) any later version. See LICENSE.

#include "doctest.h"

#include <cmath>
#include <cstdint>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "mrd/csv_loader.hpp"
#include "mrd/features.hpp"
#include "mrd/nn_index.hpp"
#include "mrd/standardize.hpp"

namespace {

mrd::FeatureMatrix from_points(const std::vector<std::vector<double>>& pts) {
    mrd::FeatureMatrix m(pts.size(), pts[0].size());
    for (std::size_t t = 0; t < pts.size(); ++t) {
        for (std::size_t j = 0; j < pts[t].size(); ++j) m.row(t)[j] = pts[t][j];
        m.dates()[t] = "p" + std::to_string(t);
    }
    return m;
}

// Deterministic clustered synthetic data: `n` points, d=4, four Gaussian
// clusters (Box-Muller over an LCG) at well-separated centers.
mrd::FeatureMatrix make_clustered(std::size_t n, std::uint64_t seed = 9) {
    mrd::FeatureMatrix m(n, 4);
    std::uint64_t x = seed;
    auto uni = [&x]() {
        x = x * 6364136223846793005ULL + 1442695040888963407ULL;
        return (static_cast<double>(x >> 11) + 0.5) / 9007199254740992.0;
    };
    const double centers[4][4] = {
        {0, 0, 0, 0}, {6, 0, 0, 0}, {0, 6, 0, 0}, {3, 3, 6, 0}};
    for (std::size_t t = 0; t < n; ++t) {
        const auto& c = centers[t % 4];
        for (std::size_t j = 0; j < 4; ++j) {
            const double g = std::sqrt(-2.0 * std::log(uni())) *
                             std::cos(6.283185307179586 * uni());
            m.row(t)[j] = c[j] + g;
        }
        m.dates()[t] = "p" + std::to_string(t);
    }
    return m;
}

double mean_recall(const mrd::NNIndex& approx, const mrd::NNIndex& oracle,
                   const mrd::FeatureMatrix& data, int k, std::size_t stride) {
    double sum = 0.0;
    std::size_t queries = 0;
    for (std::size_t t = 0; t < data.n_rows(); t += stride) {
        const auto ex = oracle.query(data.row(t), k, t);
        const auto ap = approx.query(data.row(t), k, t);
        std::size_t hits = 0;
        for (const auto& a : ap) {
            for (const auto& e : ex) {
                if (a.index == e.index) {
                    ++hits;
                    break;
                }
            }
        }
        sum += static_cast<double>(hits) / static_cast<double>(ex.size());
        ++queries;
    }
    return sum / static_cast<double>(queries);
}

}  // namespace

TEST_CASE("BruteForce: hand-computed fixture") {
    // (0,0) (1,0) (0,2) (3,4): distances from origin are 0, 1, 2, 5.
    const auto m = from_points({{0, 0}, {1, 0}, {0, 2}, {3, 4}});
    mrd::BruteForceIndex idx;
    idx.build(m);

    SUBCASE("query row 0 excluding itself") {
        const auto nn = idx.query(m.row(0), 2, 0);
        REQUIRE(nn.size() == 2);
        CHECK(nn[0].index == 1);
        CHECK(nn[0].dist == doctest::Approx(1.0));
        CHECK(nn[1].index == 2);
        CHECK(nn[1].dist == doctest::Approx(2.0));
    }
    SUBCASE("no exclusion: the point itself comes first at distance 0") {
        const auto nn = idx.query(m.row(0), 2);
        REQUIRE(nn.size() == 2);
        CHECK(nn[0].index == 0);
        CHECK(nn[0].dist == 0.0);
        CHECK(nn[1].index == 1);
    }
    SUBCASE("external query vector excludes nothing") {
        const std::vector<double> q = {2.9, 4.1};
        const auto nn = idx.query(q, 1);
        REQUIRE(nn.size() == 1);
        CHECK(nn[0].index == 3);
    }
    SUBCASE("k larger than available clamps") {
        const auto nn = idx.query(m.row(0), 10, 0);
        CHECK(nn.size() == 3);
    }
}

TEST_CASE("BruteForce: ties broken by ascending index") {
    // Rows 1 and 2 are identical -> equidistant from row 0.
    const auto m = from_points({{0, 0}, {1, 1}, {1, 1}, {5, 5}});
    mrd::BruteForceIndex idx;
    idx.build(m);
    const auto nn = idx.query(m.row(0), 2, 0);
    CHECK(nn[0].index == 1);
    CHECK(nn[1].index == 2);
    CHECK(nn[0].dist == nn[1].dist);
}

TEST_CASE("contract violations throw (both implementations)") {
    const auto m = from_points({{0, 0}, {1, 0}, {0, 2}, {3, 4}});
    mrd::BruteForceIndex brute;
    mrd::HnswIndex hnsw;

    CHECK_THROWS_AS((void)brute.query(m.row(0), 1), std::logic_error);
    CHECK_THROWS_AS((void)hnsw.query(m.row(0), 1), std::logic_error);

    brute.build(m);
    hnsw.build(m);
    CHECK_THROWS_AS((void)brute.query(m.row(0), 0), std::invalid_argument);
    CHECK_THROWS_AS((void)hnsw.query(m.row(0), -1), std::invalid_argument);

    const std::vector<double> bad_dim = {1.0, 2.0, 3.0};
    CHECK_THROWS_AS((void)brute.query(bad_dim, 1), std::logic_error);
    CHECK_THROWS_AS((void)hnsw.query(bad_dim, 1), std::logic_error);

    CHECK_THROWS_AS(mrd::HnswIndex({1, 200, 64, 42}), std::invalid_argument);
}

TEST_CASE("HNSW: exact on a tiny set, self-exclusion honored") {
    const auto m = from_points({{0, 0}, {1, 0}, {0, 2}, {3, 4}, {10, 10}});
    mrd::HnswIndex idx;  // ef_search 64 > N -> effectively exhaustive
    idx.build(m);

    const auto nn = idx.query(m.row(0), 3, 0);
    REQUIRE(nn.size() == 3);
    CHECK(nn[0].index == 1);
    CHECK(nn[1].index == 2);
    CHECK(nn[2].index == 3);
    for (const auto& x : nn) CHECK(x.index != 0);
}

TEST_CASE("HNSW: recall@10 >= 0.95 on clustered synthetic data") {
    const auto data = make_clustered(2000);
    mrd::BruteForceIndex oracle;
    oracle.build(data);
    mrd::HnswIndex hnsw;
    hnsw.build(data);

    const double recall = mean_recall(hnsw, oracle, data, 10, 1);
    MESSAGE("synthetic recall@10 = ", recall);
    CHECK(recall >= 0.95);
}

TEST_CASE("HNSW: deterministic build and query for fixed seed") {
    const auto data = make_clustered(500);
    mrd::HnswIndex a, b;
    a.build(data);
    b.build(data);
    for (std::size_t t = 0; t < data.n_rows(); t += 10) {
        const auto ra = a.query(data.row(t), 10, t);
        const auto rb = b.query(data.row(t), 10, t);
        REQUIRE(ra.size() == rb.size());
        for (std::size_t i = 0; i < ra.size(); ++i) {
            CHECK(ra[i].index == rb[i].index);
            CHECK(ra[i].dist == rb[i].dist);  // exact
        }
    }
}

TEST_CASE("HNSW: recall@10 >= 0.95 on real SPY pipeline (skipped if no data)") {
    std::string path;
    for (const char* c : {"data/spy.csv", "../data/spy.csv"}) {
        if (std::ifstream(c).good()) {
            path = c;
            break;
        }
    }
    if (path.empty()) {
        MESSAGE("data/spy.csv not present -- skipping real-data recall test");
        return;
    }

    const auto series = mrd::load_ohlcv_csv(path);
    const auto raw = mrd::compute_features(series);
    mrd::ZScoreGlobal z;
    z.fit(raw);
    const auto std_m = z.transform(raw);

    mrd::BruteForceIndex oracle;
    oracle.build(std_m);
    mrd::HnswIndex hnsw;
    hnsw.build(std_m);

    // Every 10th row: 840+ queries is plenty to estimate recall in a test;
    // the bench CLI does the full sweep.
    const double recall = mean_recall(hnsw, oracle, std_m, 10, 10);
    MESSAGE("SPY recall@10 = ", recall);
    CHECK(recall >= 0.95);
}
