// Copyright (c) 2026 Cristian Mendoza <openbeta.finance>
// SPDX-License-Identifier: AGPL-3.0-or-later
//
// This file is part of regime-discovery-cpp. It is free software: you can
// redistribute it and/or modify it under the terms of the GNU Affero General
// Public License as published by the Free Software Foundation, either version
// 3 of the License, or (at your option) any later version. See LICENSE.

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <fstream>
#include <memory>
#include <numeric>
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

[[noreturn]] void usage(const char* argv0) {
    std::fprintf(stderr,
                 "usage:\n"
                 "  %s features  <ohlcv.csv> [prefix] [--std=zscore|robust|none] "
                 "[--d=4|5] [--warmup=N]\n"
                 "  %s neighbors <ohlcv.csv> --date=YYYY-MM-DD [-k N] [--index=brute|hnsw]\n"
                 "  %s crisis    <ohlcv.csv> [-k N]\n"
                 "  %s bench     <ohlcv.csv> [-k N] [--M=N] [--ef=N]\n"
                 "  %s dump-nn   <ohlcv.csv> [prefix] [-k N]\n"
                 "  %s regimes   <ohlcv.csv> [prefix] [-k N] [--mcs=N] "
                 "[--min-samples=N] [--block=N] [--gap=N] [--mst=exact|substrate] "
                 "[--d=4|5] [--warmup=N]\n",
                 argv0, argv0, argv0, argv0, argv0, argv0);
    std::exit(2);
}

struct Args {
    std::vector<std::string> positional;
    std::string date;
    std::string std_name = "zscore";
    std::string index_name = "brute";
    int k = 10;
    std::size_t M = 12;
    std::size_t ef = 64;
    std::size_t mcs = 30;
    std::size_t min_samples = 10;
    std::size_t block = mrd::kDefaultWindow;  // W: derived from the feature window
    std::size_t gap = 60;
    std::string mst = "exact";
    std::size_t d = 4;       // 4 = canonical Phase 1-4 features; 5 adds rel_volume
    std::size_t warmup = 0;  // forces first feature row to this day index
};

Args parse_args(int argc, char** argv, const char* argv0) {
    Args a;
    for (int i = 2; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg.rfind("--std=", 0) == 0) {
            a.std_name = arg.substr(6);
        } else if (arg.rfind("--date=", 0) == 0) {
            a.date = arg.substr(7);
        } else if (arg.rfind("--index=", 0) == 0) {
            a.index_name = arg.substr(8);
        } else if (arg.rfind("--M=", 0) == 0) {
            a.M = std::stoul(arg.substr(4));
        } else if (arg.rfind("--ef=", 0) == 0) {
            a.ef = std::stoul(arg.substr(5));
        } else if (arg.rfind("--mcs=", 0) == 0) {
            a.mcs = std::stoul(arg.substr(6));
        } else if (arg.rfind("--min-samples=", 0) == 0) {
            a.min_samples = std::stoul(arg.substr(14));
        } else if (arg.rfind("--block=", 0) == 0) {
            a.block = std::stoul(arg.substr(8));
        } else if (arg.rfind("--gap=", 0) == 0) {
            a.gap = std::stoul(arg.substr(6));
        } else if (arg.rfind("--mst=", 0) == 0) {
            a.mst = arg.substr(6);
        } else if (arg.rfind("--d=", 0) == 0) {
            a.d = std::stoul(arg.substr(4));
        } else if (arg.rfind("--warmup=", 0) == 0) {
            a.warmup = std::stoul(arg.substr(9));
        } else if (arg == "-k" && i + 1 < argc) {
            a.k = std::stoi(argv[++i]);
        } else if (!arg.empty() && arg[0] == '-') {
            usage(argv0);
        } else {
            a.positional.push_back(arg);
        }
    }
    if (a.positional.empty()) usage(argv0);
    return a;
}

/// Loaded series + raw and standardized (global z-score) feature matrices.
/// The standardized matrix is the metric space; raw is kept for
/// interpretable display.
struct Pipeline {
    mrd::OhlcvSeries series;
    mrd::FeatureMatrix raw;
    mrd::FeatureMatrix std_m;
};

mrd::FeatureParams feature_params(const Args& a) {
    if (a.d != 4 && a.d != 5) throw std::invalid_argument("--d must be 4 or 5");
    return {mrd::kDefaultWindow, a.d == 5, 252, a.warmup};
}

Pipeline run_pipeline(const std::string& csv_path, const mrd::FeatureParams& fp) {
    mrd::OhlcvSeries series = mrd::load_ohlcv_csv(csv_path);
    mrd::FeatureMatrix raw = mrd::compute_features(series, fp);
    mrd::ZScoreGlobal z;
    z.fit(raw);
    mrd::FeatureMatrix std_m = z.transform(raw);
    return {std::move(series), std::move(raw), std::move(std_m)};
}

Pipeline run_pipeline(const std::string& csv_path) {
    return run_pipeline(csv_path, mrd::FeatureParams{});
}

/// Series day index of feature row 0 (warm-up), derived from the sizes.
std::size_t first_row_day(const Pipeline& p) {
    return p.series.size() - p.raw.n_rows();
}

std::size_t find_row(const mrd::FeatureMatrix& m, const std::string& date) {
    const auto& dates = m.dates();
    const auto it = std::lower_bound(dates.begin(), dates.end(), date);
    if (it == dates.end() || *it != date) {
        std::string msg = "date " + date + " not in feature matrix";
        if (it != dates.end()) msg += " (next trading day: " + *it + ")";
        throw std::runtime_error(msg);
    }
    return static_cast<std::size_t>(it - dates.begin());
}

std::unique_ptr<mrd::NNIndex> make_index(const Args& a) {
    if (a.index_name == "brute") return std::make_unique<mrd::BruteForceIndex>();
    if (a.index_name == "hnsw") {
        return std::make_unique<mrd::HnswIndex>(
            mrd::HnswIndex::Params{a.M, 200, a.ef, 42});
    }
    throw std::invalid_argument("unknown index '" + a.index_name +
                                "' (expected: brute | hnsw)");
}

void print_day_header(const Pipeline& p, std::size_t t) {
    const auto r = p.raw.row(t);
    std::printf("=== %s  (row %zu)   vol=%8.5f  mom=%+8.4f  mdd=%+8.4f  skew=%+7.3f\n",
                p.raw.dates()[t].c_str(), t, r[0], r[1], r[2], r[3]);
}

void print_neighbors(const Pipeline& p, std::size_t t,
                     const std::vector<mrd::Neighbor>& nn) {
    print_day_header(p, t);
    std::printf("  rank  date          dist    vol       mom       mdd       skew\n");
    for (std::size_t i = 0; i < nn.size(); ++i) {
        const auto r = p.raw.row(nn[i].index);
        std::printf("  %4zu  %s  %6.3f  %8.5f  %+8.4f  %+8.4f  %+7.3f\n", i + 1,
                    p.raw.dates()[nn[i].index].c_str(), nn[i].dist, r[0], r[1],
                    r[2], r[3]);
    }
}

int cmd_features(const Args& a) {
    const std::string prefix =
        a.positional.size() > 1 ? a.positional[1] : "features";
    const mrd::OhlcvSeries series = mrd::load_ohlcv_csv(a.positional[0]);
    const mrd::FeatureMatrix raw = mrd::compute_features(series, feature_params(a));

    mrd::dump_features_csv(raw, prefix + "_raw.csv");
    std::printf("loaded  %zu days  [%s .. %s]\n", series.size(),
                series.dates.front().c_str(), series.dates.back().c_str());
    std::printf("raw     %zu rows x %zu cols -> %s_raw.csv\n", raw.n_rows(),
                raw.dim(), prefix.c_str());

    if (a.std_name != "none") {
        const auto standardizer = mrd::make_standardizer(a.std_name);
        standardizer->fit(raw);
        const mrd::FeatureMatrix std_m = standardizer->transform(raw);
        mrd::dump_features_csv(std_m, prefix + "_std.csv");
        std::printf("std     %zu rows x %zu cols (%s) -> %s_std.csv\n",
                    std_m.n_rows(), std_m.dim(), a.std_name.c_str(),
                    prefix.c_str());
    }
    return 0;
}

int cmd_neighbors(const Args& a) {
    if (a.date.empty()) {
        throw std::invalid_argument("neighbors requires --date=YYYY-MM-DD");
    }
    const Pipeline p = run_pipeline(a.positional[0]);
    const auto index = make_index(a);
    index->build(p.std_m);
    const std::size_t t = find_row(p.std_m, a.date);
    print_neighbors(p, t, index->query(p.std_m.row(t), a.k, t));
    return 0;
}

int cmd_crisis(const Args& a) {
    // Known stress days: if their nearest neighbors are other crisis days,
    // the feature space encodes regime structure and Phase 1+2 is validated.
    static const char* kCrisisDays[] = {"2008-10-10", "2008-11-20",
                                        "2020-03-16", "2020-03-23"};
    const Pipeline p = run_pipeline(a.positional[0]);
    mrd::BruteForceIndex index;  // exact neighbors only for validation
    index.build(p.std_m);
    for (const char* day : kCrisisDays) {
        const std::size_t t = find_row(p.std_m, day);
        print_neighbors(p, t, index.query(p.std_m.row(t), a.k, t));
        std::printf("\n");
    }
    return 0;
}

int cmd_bench(const Args& a) {
    using Clock = std::chrono::steady_clock;
    const Pipeline p = run_pipeline(a.positional[0]);
    const std::size_t n = p.std_m.n_rows();
    const auto k = a.k;

    mrd::BruteForceIndex brute;
    const auto tb0 = Clock::now();
    brute.build(p.std_m);
    const auto tb1 = Clock::now();

    mrd::HnswIndex hnsw({a.M, 200, a.ef, 42});
    const auto th0 = Clock::now();
    hnsw.build(p.std_m);
    const auto th1 = Clock::now();

    auto us = [](auto d) {
        return std::chrono::duration<double, std::micro>(d).count();
    };

    // Every row is a query — brute is microseconds/query, so no sampling and
    // therefore no sampling noise in recall.
    std::vector<double> lat_brute(n), lat_hnsw(n);
    std::vector<std::vector<mrd::Neighbor>> exact(n);
    double recall_sum = 0.0;
    for (std::size_t t = 0; t < n; ++t) {
        const auto q0 = Clock::now();
        exact[t] = brute.query(p.std_m.row(t), k, t);
        lat_brute[t] = us(Clock::now() - q0);
    }
    for (std::size_t t = 0; t < n; ++t) {
        const auto q0 = Clock::now();
        const auto approx = hnsw.query(p.std_m.row(t), k, t);
        lat_hnsw[t] = us(Clock::now() - q0);

        std::size_t hits = 0;
        for (const auto& nb : approx) {
            for (const auto& ex : exact[t]) {
                if (ex.index == nb.index) {
                    ++hits;
                    break;
                }
            }
        }
        recall_sum +=
            static_cast<double>(hits) / static_cast<double>(exact[t].size());
    }

    auto stats = [](std::vector<double>& v) {
        std::sort(v.begin(), v.end());
        const double mean =
            std::accumulate(v.begin(), v.end(), 0.0) / static_cast<double>(v.size());
        return std::array<double, 3>{mean, v[v.size() / 2],
                                     v[v.size() * 99 / 100]};
    };
    const auto sb = stats(lat_brute);
    const auto sh = stats(lat_hnsw);

    std::printf("bench  N=%zu d=%zu k=%d  (HNSW M=%zu efC=200 efS=%zu)\n", n,
                p.std_m.dim(), k, a.M, a.ef);
    std::printf("build   brute %8.0f us   hnsw %8.0f us\n", us(tb1 - tb0),
                us(th1 - th0));
    std::printf("query   brute mean %6.2f us  p50 %6.2f  p99 %6.2f\n", sb[0],
                sb[1], sb[2]);
    std::printf("query   hnsw  mean %6.2f us  p50 %6.2f  p99 %6.2f\n", sh[0],
                sh[1], sh[2]);
    std::printf("recall@%d = %.4f\n", k, recall_sum / static_cast<double>(n));
    return 0;
}

int cmd_dump_nn(const Args& a) {
    const std::string prefix =
        a.positional.size() > 1 ? a.positional[1] : "features";
    const Pipeline p = run_pipeline(a.positional[0]);
    mrd::BruteForceIndex index;  // exact neighbors feed clustering/viz
    index.build(p.std_m);

    const std::string path = prefix + "_neighbors.csv";
    std::ofstream out(path);
    if (!out) throw std::runtime_error("cannot open " + path + " for writing");
    out << "index,date,rank,neighbor_index,neighbor_date,distance\n";
    out.precision(17);
    for (std::size_t t = 0; t < p.std_m.n_rows(); ++t) {
        const auto nn = index.query(p.std_m.row(t), a.k, t);
        for (std::size_t i = 0; i < nn.size(); ++i) {
            out << t << ',' << p.std_m.dates()[t] << ',' << i + 1 << ','
                << nn[i].index << ',' << p.std_m.dates()[nn[i].index] << ','
                << nn[i].dist << '\n';
        }
    }
    if (!out) throw std::runtime_error("write failed on " + path);
    std::printf("neighbors k=%d for %zu rows -> %s\n", a.k, p.std_m.n_rows(),
                path.c_str());
    return 0;
}

int cmd_regimes(const Args& a) {
    const std::string prefix =
        a.positional.size() > 1 ? a.positional[1] : "features";
    const Pipeline p = run_pipeline(a.positional[0], feature_params(a));
    const std::size_t k = static_cast<std::size_t>(a.k);

    mrd::BruteForceIndex index;  // exact substrate for the science
    index.build(p.std_m);
    const auto blocked =
        mrd::build_neighbor_graph(index, p.std_m, k, a.block);

    if (a.mst != "exact" && a.mst != "substrate") {
        throw std::invalid_argument("--mst must be exact or substrate");
    }
    const mrd::HdbscanParams params{a.mcs, a.min_samples, a.mst == "substrate"};
    const auto res = mrd::hdbscan(p.std_m, blocked, params);

    const std::size_t n = res.labels.size();
    std::size_t noise = 0;
    for (const int l : res.labels) noise += (l < 0);
    std::printf("regimes  N=%zu  d=%zu  k=%zu  W=%zu  mcs=%zu  min_samples=%zu"
                "  first=%s\n",
                n, p.std_m.dim(), k, a.block, a.mcs, a.min_samples,
                p.std_m.dates().front().c_str());
    std::printf("clusters: %zu   noise: %zu (%.1f%%)\n\n", res.n_clusters,
                noise, 100.0 * static_cast<double>(noise) / static_cast<double>(n));

    const auto runs = mrd::persistence(res.labels);
    const auto episodes = mrd::recurrence(res.labels, a.gap);
    const auto fwd = mrd::forward_vol_by_regime(res.labels, p.series,
                                                first_row_day(p));

    std::printf("validation  (persistence on final labels; episodes at gap>%zu; "
                "fwd vol horizon 20)\n", a.gap);
    std::printf("  id    days   runs  mean_run  episodes  span                      "
                "stability\n");
    for (const auto& r : runs) {
        const mrd::EpisodeStats* ep = nullptr;
        for (const auto& e : episodes) {
            if (e.label == r.label) ep = &e;
        }
        std::printf("  %2d  %6zu  %5zu  %8.2f", r.label, r.n_days, r.n_runs,
                    r.mean_run_length);
        if (ep != nullptr) {
            std::printf("  %8zu  %s..%s", ep->n_episodes,
                        p.std_m.dates()[ep->first_day].c_str(),
                        p.std_m.dates()[ep->last_day].c_str());
        } else {
            std::printf("  %8s  %-24s", "-", "-");
        }
        if (r.label >= 0) {
            std::printf("  %9.2f", res.stabilities[static_cast<std::size_t>(r.label)]);
        }
        std::printf("\n");
    }

    std::printf("\nforward 20d realized vol by regime (daily units):\n");
    std::printf("  id       n      mean    median\n");
    for (const auto& f : fwd) {
        std::printf("  %2d  %6zu  %8.5f  %8.5f\n", f.label, f.n, f.mean,
                    f.median);
    }

    const std::string path = prefix + "_regimes.csv";
    std::ofstream out(path);
    if (!out) throw std::runtime_error("cannot open " + path + " for writing");
    out << "index,date,label\n";
    for (std::size_t t = 0; t < n; ++t) {
        out << t << ',' << p.std_m.dates()[t] << ',' << res.labels[t] << '\n';
    }
    if (!out) throw std::runtime_error("write failed on " + path);
    std::printf("\nlabels -> %s\n", path.c_str());
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 3) usage(argv[0]);
    const std::string cmd = argv[1];
    try {
        const Args a = parse_args(argc, argv, argv[0]);
        if (cmd == "features") return cmd_features(a);
        if (cmd == "neighbors") return cmd_neighbors(a);
        if (cmd == "crisis") return cmd_crisis(a);
        if (cmd == "bench") return cmd_bench(a);
        if (cmd == "dump-nn") return cmd_dump_nn(a);
        if (cmd == "regimes") return cmd_regimes(a);
        usage(argv[0]);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
}
