// Copyright (c) 2026 Cristian Mendoza <openbeta.finance>
// SPDX-License-Identifier: AGPL-3.0-or-later
//
// This file is part of regime-discovery-cpp. It is free software: you can
// redistribute it and/or modify it under the terms of the GNU Affero General
// Public License as published by the Free Software Foundation, either version
// 3 of the License, or (at your option) any later version. See LICENSE.

#include "mrd/csv_loader.hpp"

#include <array>
#include <charconv>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string_view>

namespace mrd {

namespace {

[[noreturn]] void fail(const std::string& path, std::size_t line_no,
                       const std::string& what) {
    throw std::runtime_error(path + ":" + std::to_string(line_no) + ": " + what);
}

double parse_double(std::string_view field, const std::string& path,
                    std::size_t line_no) {
    double value{};
    const auto [ptr, ec] =
        std::from_chars(field.data(), field.data() + field.size(), value);
    if (ec != std::errc{} || ptr != field.data() + field.size()) {
        fail(path, line_no, "bad numeric field '" + std::string(field) + "'");
    }
    return value;
}

}  // namespace

OhlcvSeries load_ohlcv_csv(const std::string& path) {
    std::ifstream in(path);
    if (!in) throw std::runtime_error("cannot open " + path);

    std::string line;
    std::size_t line_no = 1;
    if (!std::getline(in, line)) fail(path, line_no, "empty file");
    // Tolerate a UTF-8 BOM and \r\n endings.
    if (line.size() >= 3 && line.compare(0, 3, "\xEF\xBB\xBF") == 0) line.erase(0, 3);
    if (!line.empty() && line.back() == '\r') line.pop_back();
    if (line != "date,open,high,low,close,volume") {
        fail(path, line_no,
             "expected header 'date,open,high,low,close,volume', got '" + line + "'");
    }

    OhlcvSeries s;
    while (std::getline(in, line)) {
        ++line_no;
        if (!line.empty() && line.back() == '\r') line.pop_back();
        if (line.empty()) continue;

        std::array<std::string_view, 6> fields;
        std::string_view rest = line;
        for (std::size_t i = 0; i < 6; ++i) {
            const auto comma = rest.find(',');
            if (i < 5) {
                if (comma == std::string_view::npos)
                    fail(path, line_no, "expected 6 fields");
                fields[i] = rest.substr(0, comma);
                rest.remove_prefix(comma + 1);
            } else {
                if (comma != std::string_view::npos)
                    fail(path, line_no, "expected 6 fields, got more");
                fields[i] = rest;
            }
        }

        Date date{fields[0]};
        if (date.empty()) fail(path, line_no, "empty date");
        if (!s.dates.empty() && date <= s.dates.back()) {
            fail(path, line_no, "dates not strictly increasing ('" + date +
                                    "' after '" + s.dates.back() + "')");
        }

        const double o = parse_double(fields[1], path, line_no);
        const double h = parse_double(fields[2], path, line_no);
        const double l = parse_double(fields[3], path, line_no);
        const double c = parse_double(fields[4], path, line_no);
        const double v = parse_double(fields[5], path, line_no);
        if (o <= 0.0 || h <= 0.0 || l <= 0.0 || c <= 0.0) {
            fail(path, line_no, "non-positive price");
        }

        s.dates.push_back(std::move(date));
        s.open.push_back(o);
        s.high.push_back(h);
        s.low.push_back(l);
        s.close.push_back(c);
        s.volume.push_back(v);
    }
    if (s.size() == 0) fail(path, line_no, "no data rows");
    return s;
}

void dump_features_csv(const FeatureMatrix& m, const std::string& path) {
    std::ofstream out(path);
    if (!out) throw std::runtime_error("cannot open " + path + " for writing");

    out << "date";
    // Header follows the matrix dim, not kNumFeatures: a d=4 matrix (no
    // volume feature) must not claim a rel_volume column.
    for (std::size_t j = 0; j < m.dim(); ++j) out << ',' << kFeatureNames[j];
    out << '\n';

    out.precision(17);
    for (std::size_t t = 0; t < m.n_rows(); ++t) {
        out << m.dates()[t];
        for (double x : m.row(t)) out << ',' << x;
        out << '\n';
    }
    if (!out) throw std::runtime_error("write failed on " + path);
}

}  // namespace mrd
