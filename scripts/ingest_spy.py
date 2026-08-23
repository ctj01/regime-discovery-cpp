#!/usr/bin/env python3
# Copyright (c) 2026 Cristian Mendoza <openbeta.finance>
# SPDX-License-Identifier: AGPL-3.0-or-later
#
# This file is part of regime-discovery-cpp, licensed under the GNU Affero
# General Public License v3 or later. See LICENSE.
"""Download full daily OHLCV history for SPY from Twelve Data into data/spy.csv.

Run-once ingestion scaffolding for the C++ regime engine. Produces exactly the
format mrd::load_ohlcv_csv expects: header `date,open,high,low,close,volume`,
strictly increasing dates, positive prices.

Usage:
    export TWELVEDATA_API_KEY=<your key>        # PowerShell: $env:TWELVEDATA_API_KEY = "..."
    python3 scripts/ingest_spy.py

The API key is read from the environment only; it is never written to the CSV
and is redacted from any error output.
"""

import os
import sys
import time
from datetime import date, datetime
from pathlib import Path

import requests

SYMBOL = "SPY"
INTERVAL = "1day"
CHUNK = 5000  # max bars per request on the free plan (~19.8y of trading days)
BASE_URL = "https://api.twelvedata.com/time_series"
OUT_PATH = Path(__file__).resolve().parent.parent / "data" / "spy.csv"

API_KEY = os.environ.get("TWELVEDATA_API_KEY", "")


def die(msg: str) -> "None":
    # Belt and braces: never let the key leak through an error message
    # (requests exceptions can embed the full request URL).
    if API_KEY:
        msg = msg.replace(API_KEY, "<redacted>")
    print(f"error: {msg}", file=sys.stderr)
    sys.exit(1)


def fetch_chunk(end_date: "str | None") -> list:
    """One time_series request, ascending, ending at end_date (inclusive).

    Returns the list of bar dicts (possibly empty when walking past the start
    of history). Retries on rate limit (429); dies on any other API error.
    """
    params = {
        "symbol": SYMBOL,
        "interval": INTERVAL,
        "outputsize": CHUNK,
        "order": "ASC",
        "apikey": API_KEY,
    }
    if end_date is not None:
        params["end_date"] = end_date

    for attempt in range(4):
        try:
            resp = requests.get(BASE_URL, params=params, timeout=30)
            data = resp.json()
        except requests.RequestException as e:
            die(f"request failed: {e}")
        except ValueError:
            die(f"non-JSON response (HTTP {resp.status_code})")

        if data.get("status") == "error" or "values" not in data:
            code = data.get("code")
            message = data.get("message", "unknown API error")
            if code == 429:  # per-minute rate limit; free plan is 8 req/min
                wait = 15 * (attempt + 1)
                print(f"rate limited, retrying in {wait}s ...", file=sys.stderr)
                time.sleep(wait)
                continue
            if "no data" in str(message).lower():
                return []  # walked past the start of available history
            die(f"Twelve Data error (code {code}): {message}")

        return data["values"]

    die("still rate limited after retries")


def download_full_history() -> list:
    """Paginate backwards from the present until history is exhausted.

    Each chunk is ascending; chunks are collected newest-first and reversed at
    the end. The next request uses end_date = earliest bar already received —
    deliberately overlapping by up to one bar, because Twelve Data's end_date
    inclusivity is not reliable (an exclusive interpretation with earliest-1
    was observed to drop the boundary trading day). The overlap is deduped in
    the merge, and mismatching duplicates are a hard error.
    """
    chunks = []
    end_date = None
    while True:
        values = fetch_chunk(end_date)
        if not values:
            break
        earliest = values[0]["datetime"]
        print(f"  fetched {len(values):5d} bars  [{earliest} .. {values[-1]['datetime']}]")
        if end_date is not None and earliest >= end_date:
            # no progress (chunk is only the overlap bar) -> history exhausted
            if len(values) == 1:
                break
            die(f"pagination made no progress at end_date {end_date}")
        chunks.append(values)
        if len(values) < CHUNK:
            break
        end_date = earliest

    rows = []
    for chunk in reversed(chunks):
        for bar in chunk:
            if rows and bar["datetime"] == rows[-1]["datetime"]:
                if bar != rows[-1]:
                    die(f"overlap bar mismatch at {bar['datetime']}: {rows[-1]} vs {bar}")
                continue  # boundary overlap, identical -> drop
            rows.append(bar)
    return rows


def validate(rows: list) -> "None":
    """Fail loudly on anything the C++ loader would reject (or worse, accept
    silently wrong): duplicate/non-increasing dates, non-positive prices,
    malformed numbers. A bad download must never reach the C++ side."""
    if not rows:
        die("no data returned")
    prev = ""
    for i, r in enumerate(rows):
        d = r["datetime"]
        try:
            datetime.strptime(d, "%Y-%m-%d")
        except ValueError:
            die(f"row {i}: bad date '{d}'")
        if d <= prev:
            die(f"row {i}: dates not strictly increasing ('{d}' after '{prev}')")
        prev = d
        try:
            o, h, l, c = (float(r[k]) for k in ("open", "high", "low", "close"))
            v = float(r["volume"])
        except (KeyError, ValueError) as e:
            die(f"row {i} ({d}): malformed field: {e}")
        if min(o, h, l, c) <= 0.0:
            die(f"row {i} ({d}): non-positive price")
        if v < 0.0:
            die(f"row {i} ({d}): negative volume")


def main() -> "None":
    if not API_KEY:
        die("TWELVEDATA_API_KEY is not set")

    print(f"downloading {SYMBOL} {INTERVAL} history from Twelve Data ...")
    rows = download_full_history()
    validate(rows)

    first, last = rows[0]["datetime"], rows[-1]["datetime"]
    years = (date.fromisoformat(last) - date.fromisoformat(first)).days / 365.25
    if years < 20:
        print(f"warning: only {years:.1f} years of history "
              f"({first} .. {last}) — plan may cap history depth", file=sys.stderr)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w", newline="\n") as f:
        f.write("date,open,high,low,close,volume\n")
        for r in rows:
            f.write(f"{r['datetime']},{r['open']},{r['high']},"
                    f"{r['low']},{r['close']},{r['volume']}\n")

    print(f"{len(rows)} rows, {first} .. {last} -> {OUT_PATH}")


if __name__ == "__main__":
    main()
