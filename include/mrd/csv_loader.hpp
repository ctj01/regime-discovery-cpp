// Copyright (c) 2026 Cristian Mendoza <openbeta.finance>
// SPDX-License-Identifier: AGPL-3.0-or-later
//
// This file is part of regime-discovery-cpp. It is free software: you can
// redistribute it and/or modify it under the terms of the GNU Affero General
// Public License as published by the Free Software Foundation, either version
// 3 of the License, or (at your option) any later version. See LICENSE.

#pragma once

#include <string>

#include "mrd/types.hpp"

namespace mrd {

/// Loads a daily OHLCV CSV with header "date,open,high,low,close,volume",
/// rows in chronological order.
///
/// Validates and establishes the OhlcvSeries invariants: strictly increasing
/// dates, prices > 0, no missing fields. Throws std::runtime_error with the
/// offending line number on any violation — a malformed input must never
/// produce a silently wrong feature matrix.
OhlcvSeries load_ohlcv_csv(const std::string& path);

/// Dumps the feature matrix as "date,<feature names...>" CSV, one row per
/// day, full double precision (round-trippable). Inspection/debug output.
void dump_features_csv(const FeatureMatrix& m, const std::string& path);

}  // namespace mrd
