#!/usr/bin/env python3
# Copyright (c) 2026 Cristian Mendoza <openbeta.finance>
# SPDX-License-Identifier: AGPL-3.0-or-later
#
# This file is part of regime-discovery-cpp, licensed under the GNU Affero
# General Public License v3 or later. See LICENSE.
"""Builds the self-contained regime visualization.

Merges the engine's output CSVs (price, d=5 raw features, default and probe
regime labels) into a compact JSON blob and inlines it into
viz/regimes.src.html, writing viz/regimes.html — a single file that works on
double-click, no server, no external requests.

Usage:  python3 scripts/build_viz.py
"""

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
SRC = ROOT / "viz" / "regimes.src.html"
OUT = ROOT / "viz" / "regimes.html"


def main() -> None:
    spy = pd.read_csv(DATA / "spy.csv")
    raw = pd.read_csv(DATA / "features_spy5_raw.csv")
    lab_def = pd.read_csv(DATA / "features_spy5_regimes.csv")
    lab_probe = pd.read_csv(DATA / "features_spy5_probe_regimes.csv")

    dates = spy["date"].tolist()
    first = dates.index(raw["date"].iloc[0])
    assert dates[first:] == raw["date"].tolist(), "spy/raw date misalignment"
    assert raw["date"].tolist() == lab_def["date"].tolist() == lab_probe["date"].tolist()

    blob = {
        "dates": dates,
        "close": [round(c, 2) for c in spy["close"]],
        "first": first,
        "feat": [
            [round(v, 6), round(m, 5), round(d, 5), round(s, 4), round(rv, 4)]
            for v, m, d, s, rv in zip(raw["realized_vol"], raw["momentum"],
                                      raw["max_drawdown"], raw["skewness"],
                                      raw["rel_volume"])
        ],
        "labs": {
            "default": lab_def["label"].tolist(),
            "probe": lab_probe["label"].tolist(),
        },
    }
    payload = json.dumps(blob, separators=(",", ":"))

    html = SRC.read_text(encoding="utf-8")
    marker = "/*__DATA__*/null"
    assert marker in html, "data marker not found in template"
    OUT.write_text(html.replace(marker, payload), encoding="utf-8")
    print(f"{OUT.name}: {OUT.stat().st_size / 1024:.0f} KB "
          f"({len(dates)} days, features from {dates[first]})")


if __name__ == "__main__":
    main()
