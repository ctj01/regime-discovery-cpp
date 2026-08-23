// Copyright (c) 2026 Cristian Mendoza <openbeta.finance>
// SPDX-License-Identifier: AGPL-3.0-or-later
//
// This file is part of regime-discovery-cpp. It is free software: you can
// redistribute it and/or modify it under the terms of the GNU Affero General
// Public License as published by the Free Software Foundation, either version
// 3 of the License, or (at your option) any later version. See LICENSE.

#pragma once

#include <cstdint>
#include <optional>
#include <span>
#include <vector>

#include "mrd/types.hpp"

namespace mrd {

/// One nearest-neighbor hit. `dist` is true Euclidean (not squared) in the
/// space of the built matrix — for this project, the standardized feature
/// space. Metric choice is Euclidean deliberately (decision record):
/// magnitude matters (same vol AND same momentum), so no cosine.
struct Neighbor {
    std::size_t index;
    double      dist;
};

/// k-NN index over the rows of a FeatureMatrix, Euclidean metric.
///
/// build() COPIES the vectors (N*d doubles): the index owns its data and
/// cannot dangle if the source matrix goes away. Dates stay outside — the
/// index speaks row indices; callers map indices to dates.
///
/// query() contract: returns min(k, N - (exclude ? 1 : 0)) neighbors sorted
/// by ascending distance, ties broken by ascending index where the
/// implementation is exact (BruteForce). Self-exclusion is EXPLICIT: pass
/// exclude = row index of the query point; a query vector from outside the
/// matrix excludes nothing. (Implicit distance-0 exclusion was rejected: two
/// distinct days with identical vectors are regime structure, not artifacts.)
///
/// Throws std::logic_error if called before build() or if q.size() != dim;
/// std::invalid_argument on k <= 0.
class NNIndex {
public:
    virtual ~NNIndex() = default;

    virtual void build(const FeatureMatrix& data) = 0;

    [[nodiscard]] virtual std::vector<Neighbor>
    query(std::span<const double> q, int k,
          std::optional<std::size_t> exclude = std::nullopt) const = 0;
};

/// Exact k-NN by linear scan over all rows.
///
/// This is the oracle: ground truth for the HNSW recall benchmark, and the
/// default engine for neighbor inspection — approximation artifacts must not
/// contaminate validation of the feature pipeline. At N≈8.4k, d=4 a query is
/// microseconds; there is no curse of dimensionality at d=4, so exact search
/// is the scientifically correct default here.
class BruteForceIndex final : public NNIndex {
public:
    void build(const FeatureMatrix& data) override;

    [[nodiscard]] std::vector<Neighbor>
    query(std::span<const double> q, int k,
          std::optional<std::size_t> exclude = std::nullopt) const override;

private:
    std::vector<double> data_;  // row-major copy, n_ * d_
    std::size_t n_ = 0, d_ = 0;
};

/// Approximate k-NN: hierarchical navigable small world graph (Malkov &
/// Yashunin 2016). Included as measurable infrastructure — recall@k against
/// the BruteForce oracle is the metric that proves it works (see `mrd bench`).
///
/// Implementation notes:
///  - level assignment l = floor(-ln(U) * 1/ln(M)), seeded mt19937_64 —
///    build is DETERMINISTIC for fixed params/seed/insertion order.
///  - neighbor selection uses the diversity heuristic (alg. 4: a candidate is
///    kept only if it is closer to the base point than to every already
///    selected neighbor), with pruned-candidate fill-back to M — at d=4 the
///    heuristic prunes aggressively and fill-back preserves connectivity.
///  - layer-0 degree cap 2M, upper layers M; links are bidirectional with
///    heuristic re-pruning on overflow.
///  - adjacency is vector-of-vectors per node/level; flattening layer 0 into
///    a fixed-stride array is a later optimization, not needed at this N.
class HnswIndex final : public NNIndex {
public:
    struct Params {
        std::size_t   M = 12;                ///< out-degree target (upper layers)
        std::size_t   ef_construction = 200; ///< candidate list width at build
        std::size_t   ef_search = 64;        ///< candidate list width at query
        std::uint64_t seed = 42;             ///< level-assignment RNG seed
    };

    // Two constructors instead of one defaulted argument: a nested aggregate's
    // default member initializers are not usable in default arguments of the
    // enclosing class (they parse in outermost complete-class context).
    HnswIndex();  // default Params
    explicit HnswIndex(Params params);

    void build(const FeatureMatrix& data) override;

    [[nodiscard]] std::vector<Neighbor>
    query(std::span<const double> q, int k,
          std::optional<std::size_t> exclude = std::nullopt) const override;

    /// Query-time knob for the recall/latency curve; build is unaffected.
    void set_ef_search(std::size_t ef) noexcept { params_.ef_search = ef; }
    [[nodiscard]] const Params& params() const noexcept { return params_; }

private:
    using Cand = std::pair<double, std::uint32_t>;  // (squared dist, node)

    [[nodiscard]] const double* vec(std::uint32_t i) const {
        return data_.data() + static_cast<std::size_t>(i) * d_;
    }
    [[nodiscard]] std::vector<Cand> search_layer(const double* q,
                                                 std::uint32_t ep, double ep_d2,
                                                 std::size_t ef, std::size_t level,
                                                 std::vector<char>& visited) const;
    [[nodiscard]] std::vector<Cand> select_neighbors(const std::vector<Cand>& cand_sorted,
                                                     std::size_t m) const;
    void prune_links(std::uint32_t node, std::size_t level, std::size_t max_links);

    Params params_;
    double level_mult_ = 0.0;  // 1/ln(M)

    std::vector<double> data_;  // row-major copy, n_ * d_
    std::size_t n_ = 0, d_ = 0;

    // links_[node][level] -> neighbor ids; links_[node].size() == node level + 1
    std::vector<std::vector<std::vector<std::uint32_t>>> links_;
    std::uint32_t entry_ = 0;
    int max_level_ = -1;  // -1 == empty/unbuilt
};

}  // namespace mrd
