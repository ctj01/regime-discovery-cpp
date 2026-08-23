// Copyright (c) 2026 Cristian Mendoza <openbeta.finance>
// SPDX-License-Identifier: AGPL-3.0-or-later
//
// This file is part of regime-discovery-cpp. It is free software: you can
// redistribute it and/or modify it under the terms of the GNU Affero General
// Public License as published by the Free Software Foundation, either version
// 3 of the License, or (at your option) any later version. See LICENSE.

#include "doctest.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

#include "mrd/csv_loader.hpp"
#include "mrd/features.hpp"
#include "mrd/hdbscan.hpp"
#include "mrd/neighbor_graph.hpp"
#include "mrd/nn_index.hpp"
#include "mrd/regime_validate.hpp"
#include "mrd/standardize.hpp"

namespace {

// Deterministic LCG uniforms in (0,1).
struct Lcg {
    std::uint64_t x;
    double uni() {
        x = x * 6364136223846793005ULL + 1442695040888963407ULL;
        return (static_cast<double>(x >> 11) + 0.5) / 9007199254740992.0;
    }
    double gauss() {
        return std::sqrt(-2.0 * std::log(uni())) *
               std::cos(6.283185307179586 * uni());
    }
};

// Two tight Gaussian blobs + uniform background noise, d=2. Noise is
// rejection-sampled to stay > 15 away from both blob centers: background
// points near a blob legitimately chain into its cluster (see the labelling
// semantics note in the end-to-end test), so the fixture keeps noise
// unambiguous by construction.
mrd::FeatureMatrix blobs_with_noise(std::size_t per_blob, std::size_t n_noise,
                                    double blob_sep = 12.0,
                                    std::uint64_t seed = 5) {
    Lcg rng{seed};
    mrd::FeatureMatrix m(2 * per_blob + n_noise, 2);
    std::size_t t = 0;
    for (std::size_t b = 0; b < 2; ++b) {
        const double cx = b == 0 ? 0.0 : blob_sep;
        for (std::size_t i = 0; i < per_blob; ++i, ++t) {
            m.row(t)[0] = cx + rng.gauss();
            m.row(t)[1] = rng.gauss();
        }
    }
    for (std::size_t i = 0; i < n_noise; ++i, ++t) {
        double x = 0.0, y = 0.0;
        do {
            x = (rng.uni() - 0.5) * 60.0;
            y = (rng.uni() - 0.5) * 60.0;
        } while (std::min(std::hypot(x, y), std::hypot(x - blob_sep, y)) <= 15.0);
        m.row(t)[0] = x;
        m.row(t)[1] = y;
    }
    for (std::size_t j = 0; j < m.n_rows(); ++j) {
        m.dates()[j] = "p" + std::to_string(j);
    }
    return m;
}

double edist(const mrd::FeatureMatrix& m, std::size_t a, std::size_t b) {
    double s = 0.0;
    for (std::size_t j = 0; j < m.dim(); ++j) {
        const double t = m.row(a)[j] - m.row(b)[j];
        s += t * t;
    }
    return std::sqrt(s);
}

// In-test oracle, algorithmically independent from build_mst_exact (Prim):
// full Kruskal over all N(N-1)/2 mutual-reachability edges. Small N only.
double kruskal_full_weight(const mrd::FeatureMatrix& m,
                           const std::vector<double>& core) {
    const std::size_t n = m.n_rows();
    struct E {
        double w;
        std::uint32_t a, b;
    };
    std::vector<E> edges;
    edges.reserve(n * (n - 1) / 2);
    for (std::uint32_t a = 0; a < n; ++a) {
        for (std::uint32_t b = a + 1; b < n; ++b) {
            edges.push_back({mrd::hdbscan_detail::mutual_reachability(
                                 edist(m, a, b), core[a], core[b]),
                             a, b});
        }
    }
    std::sort(edges.begin(), edges.end(),
              [](const E& x, const E& y) { return x.w < y.w; });
    std::vector<std::uint32_t> parent(n);
    for (std::uint32_t i = 0; i < n; ++i) parent[i] = i;
    auto find = [&](std::uint32_t x) {
        while (parent[x] != x) x = parent[x] = parent[parent[x]];
        return x;
    };
    double total = 0.0;
    std::size_t used = 0;
    for (const auto& e : edges) {
        const auto ra = find(e.a), rb = find(e.b);
        if (ra == rb) continue;
        parent[rb] = ra;
        total += e.w;
        if (++used == n - 1) break;
    }
    return total;
}

// Adjusted Rand Index; noise (-1) treated as an ordinary label.
double ari(const std::vector<int>& x, const std::vector<int>& y) {
    REQUIRE(x.size() == y.size());
    std::map<std::pair<int, int>, double> cont;
    std::map<int, double> ax, by;
    for (std::size_t i = 0; i < x.size(); ++i) {
        cont[{x[i], y[i]}] += 1;
        ax[x[i]] += 1;
        by[y[i]] += 1;
    }
    auto c2 = [](double v) { return v * (v - 1.0) / 2.0; };
    double sum_ij = 0.0, sum_a = 0.0, sum_b = 0.0;
    for (const auto& [k, v] : cont) {
        (void)k;
        sum_ij += c2(v);
    }
    for (const auto& [k, v] : ax) {
        (void)k;
        sum_a += c2(v);
    }
    for (const auto& [k, v] : by) {
        (void)k;
        sum_b += c2(v);
    }
    const double total = c2(static_cast<double>(x.size()));
    const double expected = sum_a * sum_b / total;
    const double max_idx = 0.5 * (sum_a + sum_b);
    return (sum_ij - expected) / (max_idx - expected);
}

}  // namespace

TEST_CASE("neighbor graph: raw matches index, blocked excludes the band") {
    const auto m = blobs_with_noise(60, 20);
    mrd::BruteForceIndex idx;
    idx.build(m);

    const auto raw = mrd::build_neighbor_graph(idx, m, 5);
    REQUIRE(raw.n_rows() == m.n_rows());
    for (std::size_t t = 0; t < m.n_rows(); t += 7) {
        const auto direct = idx.query(m.row(t), 5, t);
        const auto row = raw.row(t);
        for (std::size_t i = 0; i < 5; ++i) {
            CHECK(row[i].index == direct[i].index);
            CHECK(row[i].dist == direct[i].dist);
        }
    }

    const std::size_t W = 10;
    const auto blocked = mrd::build_neighbor_graph(idx, m, 5, W);
    for (std::size_t t = 0; t < m.n_rows(); ++t) {
        for (const auto& nb : blocked.row(t)) {
            const std::size_t lag =
                t > nb.index ? t - nb.index : nb.index - t;
            CHECK(lag >= W);
        }
        // ascending distance preserved after filtering
        const auto row = blocked.row(t);
        for (std::size_t i = 1; i < row.size(); ++i) {
            CHECK(row[i].dist >= row[i - 1].dist);
        }
    }

    CHECK_THROWS_AS(mrd::build_neighbor_graph(idx, m, 200, 10),
                    std::invalid_argument);
}

TEST_CASE("hdbscan pieces: core distances and mutual reachability") {
    const auto m = blobs_with_noise(30, 10);
    mrd::BruteForceIndex idx;
    idx.build(m);
    const auto g = mrd::build_neighbor_graph(idx, m, 6);

    // min_samples counts the point itself: min_samples=4 -> 3rd other
    // neighbor -> row index 2.
    const auto core = mrd::hdbscan_detail::core_distances(g, 4);
    for (std::size_t t = 0; t < m.n_rows(); t += 5) {
        CHECK(core[t] == g.row(t)[2].dist);
    }
    CHECK_THROWS_AS(mrd::hdbscan_detail::core_distances(g, 8),
                    std::invalid_argument);
    CHECK_THROWS_AS(mrd::hdbscan_detail::core_distances(g, 1),
                    std::invalid_argument);

    CHECK(mrd::hdbscan_detail::mutual_reachability(1.0, 0.5, 0.2) == 1.0);
    CHECK(mrd::hdbscan_detail::mutual_reachability(0.1, 0.5, 0.2) == 0.5);
    CHECK(mrd::hdbscan_detail::mutual_reachability(0.1, 0.2, 0.5) == 0.5);
}

TEST_CASE("MST: exact Prim matches independent full-Kruskal oracle") {
    using namespace mrd::hdbscan_detail;
    for (const auto& [per_blob, n_noise] :
         {std::pair<std::size_t, std::size_t>{80, 40}, {50, 0}, {30, 25}}) {
        const auto m = blobs_with_noise(per_blob, n_noise);
        mrd::BruteForceIndex idx;
        idx.build(m);
        const auto g = mrd::build_neighbor_graph(idx, m, 10);
        const auto core = core_distances(g, 5);
        auto mst = build_mst_exact(m, core);
        REQUIRE(mst.size() == m.n_rows() - 1);
        double total = 0.0;
        for (const auto& e : mst) total += e.w;
        CHECK(total == doctest::Approx(kruskal_full_weight(m, core)).epsilon(1e-12));
    }
}

TEST_CASE("MST substrate variant: documented distortion vs exact") {
    using namespace mrd::hdbscan_detail;

    SUBCASE("disconnected substrate -> exact Boruvka bridging keeps it close") {
        const auto m = blobs_with_noise(50, 0, 40.0);  // far apart, no noise
        mrd::BruteForceIndex idx;
        idx.build(m);
        const auto g = mrd::build_neighbor_graph(idx, m, 3);
        const auto core = core_distances(g, 3);
        auto mst = build_mst(m, g, core);
        REQUIRE(mst.size() == m.n_rows() - 1);
        double total = 0.0;
        for (const auto& e : mst) total += e.w;
        const double exact = kruskal_full_weight(m, core);
        CHECK(total >= exact - 1e-9);  // restricted MST can never be lighter
        CHECK(total <= 1.05 * exact);
    }

    SUBCASE("noise-connected substrate misses portal edges (why it is not default)") {
        // Background chains connect the graph, so Boruvka never runs and the
        // true inter-blob bridge (not a k-NN edge) is missing: the substrate
        // MST is measurably heavier. This is the failure mode recorded in
        // HdbscanParams::substrate_mst.
        const auto m = blobs_with_noise(80, 40);
        mrd::BruteForceIndex idx;
        idx.build(m);
        const auto g = mrd::build_neighbor_graph(idx, m, 10);
        const auto core = core_distances(g, 5);
        auto mst = build_mst(m, g, core);
        REQUIRE(mst.size() == m.n_rows() - 1);
        double total = 0.0;
        for (const auto& e : mst) total += e.w;
        const double exact = kruskal_full_weight(m, core);
        CHECK(total >= exact - 1e-9);
        CHECK(total <= 1.05 * exact);  // heavier, but bounded on this fixture
    }
}

TEST_CASE("hdbscan end-to-end on blobs: 2 clusters, noise stays noise") {
    // 25 background points < min_cluster_size = 30: the background CANNOT
    // condense into a cluster by construction (sparse uniform noise CAN form
    // a legitimate cluster once it reaches mcs — that is HDBSCAN behavior,
    // not a bug — so the fixture rules it out to make the assertion crisp).
    const auto m = blobs_with_noise(150, 25);
    mrd::BruteForceIndex idx;
    idx.build(m);
    const auto g = mrd::build_neighbor_graph(idx, m, 10);

    const auto res = mrd::hdbscan(m, g, {30, 5});
    REQUIRE(res.n_clusters == 2);
    REQUIRE(res.stabilities.size() == 2);

    // Each blob overwhelmingly maps to one distinct label.
    for (std::size_t b = 0; b < 2; ++b) {
        std::map<int, std::size_t> counts;
        for (std::size_t t = b * 150; t < (b + 1) * 150; ++t) {
            counts[res.labels[t]] += 1;
        }
        int best_label = -1;
        std::size_t best = 0;
        for (const auto& [lab, c] : counts) {
            if (lab >= 0 && c > best) {
                best = c;
                best_label = lab;
            }
        }
        CHECK(best_label >= 0);
        CHECK(best >= 135);  // >= 90% of the blob
    }
    // Labelling semantics note: a point that falls out BELOW a selected
    // cluster's condensed subtree inherits its label at any lambda (matches
    // the reference library), so only background kept far from both blobs by
    // the fixture is guaranteed noise — and it must overwhelmingly be noise.
    std::size_t noise_total = 0;
    for (std::size_t t = 300; t < 325; ++t) noise_total += (res.labels[t] < 0);
    CHECK(noise_total * 10 >= 25 * 9);  // >= 90% of background is noise

    // Determinism: same inputs, same labels.
    const auto res2 = mrd::hdbscan(m, g, {30, 5});
    CHECK(res.labels == res2.labels);
}

TEST_CASE("persistence: hand fixture") {
    const std::vector<int> labels = {0, 0, 0, -1, 1, 1, 0, 0};
    const auto runs = mrd::persistence(labels);
    REQUIRE(runs.size() == 3);
    CHECK(runs[0].label == -1);
    CHECK(runs[0].n_runs == 1);
    CHECK(runs[1].label == 0);
    CHECK(runs[1].n_days == 5);
    CHECK(runs[1].n_runs == 2);
    CHECK(runs[1].mean_run_length == doctest::Approx(2.5));
    CHECK(runs[2].label == 1);
    CHECK(runs[2].mean_run_length == doctest::Approx(2.0));
}

TEST_CASE("recurrence: episodes split at gap threshold") {
    // label 0 at t = 0,1,2, 10,11, 100 -> gaps 8 and 89
    std::vector<int> labels(120, -1);
    for (const std::size_t t : {0u, 1u, 2u, 10u, 11u, 100u}) labels[t] = 0;

    auto ep = mrd::recurrence(labels, 5);
    REQUIRE(ep.size() == 1);
    CHECK(ep[0].n_episodes == 3);
    CHECK(ep[0].n_days == 6);
    CHECK(ep[0].first_day == 0);
    CHECK(ep[0].last_day == 100);

    ep = mrd::recurrence(labels, 50);
    CHECK(ep[0].n_episodes == 2);  // 8-day gap no longer splits
}

TEST_CASE("forward vol: known constant-vol segments separate by regime") {
    // Series: 61 days. Closes alternate +/-r so realized vol is exactly |r|,
    // with r doubling at day 40 (series index).
    mrd::OhlcvSeries s;
    double c = 100.0;
    for (std::size_t i = 0; i < 61; ++i) {
        const double r = (i < 40 ? 0.01 : 0.02) * (i % 2 == 0 ? 1.0 : -1.0);
        if (i > 0) c *= std::exp(r);
        char buf[16];
        std::snprintf(buf, sizeof(buf), "2020-%02zu-%02zu", 1 + i / 28,
                      1 + i % 28);
        s.dates.emplace_back(buf);
        s.open.push_back(c);
        s.high.push_back(c);
        s.low.push_back(c);
        s.close.push_back(c);
        s.volume.push_back(1.0);
    }
    // labels aligned at first_row_day = 20: rows 0..40 <-> days 20..60.
    // Row t=0 (day 20) forward window = returns of days 21..40: all |0.01|.
    // Row t=20 (day 40) forward window = returns of days 41..60: all |0.02|.
    std::vector<int> labels(41, -1);
    labels[0] = 0;
    labels[20] = 1;
    const auto fv = mrd::forward_vol_by_regime(labels, s, 20, 20);
    REQUIRE(fv.size() >= 2);
    const auto find = [&](int lab) {
        for (const auto& f : fv) {
            if (f.label == lab) return f;
        }
        REQUIRE(false);
        return fv[0];
    };
    // sample std of alternating +/-r with even n is slightly above |r|;
    // check tight brackets rather than exact closed forms
    CHECK(find(0).mean == doctest::Approx(0.01).epsilon(0.02));
    CHECK(find(1).mean == doctest::Approx(0.02).epsilon(0.02));
    CHECK(find(1).mean > 1.8 * find(0).mean);
}

TEST_CASE("hdbscan vs Python oracle on real SPY (skipped if files absent)") {
    std::string spy, oracle;
    for (const char* c : {"data/spy.csv", "../data/spy.csv"}) {
        if (std::ifstream(c).good()) spy = c;
    }
    for (const char* c :
         {"data/oracle_labels.csv", "../data/oracle_labels.csv"}) {
        if (std::ifstream(c).good()) oracle = c;
    }
    if (spy.empty() || oracle.empty()) {
        MESSAGE("spy.csv or oracle_labels.csv not present -- skipping oracle test");
        return;
    }

    // Oracle labels: index,label
    std::vector<int> ours, theirs;
    {
        std::ifstream in(oracle);
        std::string line;
        std::getline(in, line);  // header
        while (std::getline(in, line)) {
            const auto comma = line.find(',');
            REQUIRE(comma != std::string::npos);
            theirs.push_back(std::stoi(line.substr(comma + 1)));
        }
    }

    const auto series = mrd::load_ohlcv_csv(spy);
    const auto raw = mrd::compute_features(series);
    mrd::ZScoreGlobal z;
    z.fit(raw);
    const auto std_m = z.transform(raw);
    mrd::BruteForceIndex idx;
    idx.build(std_m);
    const auto blocked = mrd::build_neighbor_graph(idx, std_m, 10, 20);
    ours = mrd::hdbscan(std_m, blocked, {30, 10}).labels;

    REQUIRE(ours.size() == theirs.size());
    const double score = ari(ours, theirs);
    MESSAGE("ARI vs Python oracle = ", score);
    CHECK(score >= 0.85);
}
