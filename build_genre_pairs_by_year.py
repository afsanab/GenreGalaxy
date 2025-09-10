#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
build_genre_pairs_by_year.py
----------------------------
Creates a yearly co-occurrence table:
    genre_pairs_by_year.csv  (columns: Genre1, Genre2, Year, Count)

It will:
1) Use cleaned_goodreads_data.csv if available (expects columns: BookID, Book, Genres, Year).
2) Otherwise, load goodreads_data.csv, parse Genres, infer/clean a Year column, and proceed.

This is safe to run alongside your current pipeline; it doesn't overwrite your other outputs.
"""

import os
import ast
from itertools import combinations
from typing import Optional

import numpy as np
import pandas as pd


OUT_PAIRS_BY_YEAR = "genre_pairs_by_year.csv"
CLEANED = "cleaned_goodreads_data.csv"
RAW = "goodreads_data.csv"


def infer_year_column(df: pd.DataFrame) -> Optional[str]:
    candidates = [
        "Year", "Publication Year", "Original Publication Year",
        "Original_Publication_Year", "Published", "publication_year", "published_year"
    ]
    lower_map = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand in df.columns:
            return cand
        key = cand.lower().replace(" ", "_")
        if key in lower_map:
            return lower_map[key]

    # heuristic fallback: look for mostly 4-digit numeric values
    for c in df.columns:
        col = df[c]
        # try numeric cast on a sample
        num = pd.to_numeric(col, errors="coerce")
        num = num[(num >= 1400) & (num <= 2100)]
        if num.notna().sum() >= max(10, 0.5 * len(col)):
            return c
    return None


def coerce_year(series: pd.Series) -> pd.Series:
    def parse_one(x):
        if pd.isna(x):
            return None
        s = str(x).strip()
        # direct int
        try:
            y = int(float(s))
            if 1400 <= y <= 2100:
                return y
        except Exception:
            pass
        # find first 4-digit substring
        for i in range(len(s) - 3):
            chunk = s[i:i+4]
            if chunk.isdigit():
                y = int(chunk)
                if 1400 <= y <= 2100:
                    return y
        return None
    return series.map(parse_one).astype("Int64")


def load_cleaned_or_raw() -> pd.DataFrame:
    """
    Returns a DataFrame with columns: BookID, Book, Genres (list), Year (Int64 or NA)
    """
    if os.path.exists(CLEANED):
        df = pd.read_csv(CLEANED)
        # Ensure minimal columns exist
        if "BookID" not in df.columns:
            df = df.reset_index(drop=False).rename(columns={"index": "BookID"})
        if "Book" not in df.columns:
            df["Book"] = df["BookID"].astype(str)
        if "Genres" in df.columns and df["Genres"].dtype == object:
            # If genres are serialized lists, parse them
            try:
                df["Genres"] = df["Genres"].apply(lambda v: v if isinstance(v, list) else ast.literal_eval(v))
            except Exception:
                pass
        if "Year" not in df.columns:
            df["Year"] = pd.Series([pd.NA] * len(df), dtype="Int64")
        return df[["BookID", "Book", "Genres", "Year"]]

    # Fallback: load raw and produce minimal cleaned frame
    if not os.path.exists(RAW):
        raise FileNotFoundError(
            f"Neither {CLEANED} nor {RAW} was found. Please place your data in the working directory."
        )

    raw = pd.read_csv(RAW)
    # drop obvious noise columns if present
    for col in ["URL", "Description", "Unnamed: 0"]:
        if col in raw.columns:
            raw = raw.drop(columns=[col])

    # parse genres
    if "Genres" not in raw.columns:
        raise ValueError("Expected a 'Genres' column in the input data.")
    raw = raw.dropna(subset=["Genres"]).copy()
    raw["Genres"] = raw["Genres"].apply(ast.literal_eval)

    # infer & coerce year
    yc = infer_year_column(raw)
    if yc:
        raw["Year"] = coerce_year(raw[yc])
    else:
        raw["Year"] = pd.Series([pd.NA] * len(raw), dtype="Int64")

    # ensure identifiers
    raw = raw.reset_index(drop=False).rename(columns={"index": "BookID"})
    if "Book" not in raw.columns:
        # try to find a title-like column
        for alt in ["Title", "title", "book_title", "Name"]:
            if alt in raw.columns:
                raw = raw.rename(columns={alt: "Book"})
                break
        if "Book" not in raw.columns:
            raw["Book"] = raw["BookID"].astype(str)

    return raw[["BookID", "Book", "Genres", "Year"]]


def build_pairs_by_year(df: pd.DataFrame) -> pd.DataFrame:
    """
    From per-book Genres + Year -> yearly co-occurrence counts per (Genre1, Genre2).
    """
    records = []
    for _, row in df.iterrows():
        genres = row["Genres"]
        if not isinstance(genres, (list, tuple)) or len(genres) < 2:
            continue
        # unique & sorted within book to avoid duplicate edges from repeated genres
        gset = sorted(set(map(str, genres)))
        for a, b in combinations(gset, 2):
            # store normalized (sorted) pair
            if a == b:
                continue
            year = row["Year"]
            records.append((a, b, year))

    pairs = pd.DataFrame(records, columns=["Genre1", "Genre2", "Year"])
    # count only rows with a known year
    pairs = pairs.dropna(subset=["Year"]).copy()
    pairs["Year"] = pairs["Year"].astype(int)

    out = (
        pairs.groupby(["Genre1", "Genre2", "Year"])
             .size().rename("Count").reset_index()
             .sort_values(["Genre1", "Genre2", "Year"])
    )
    return out


def main():
    df = load_cleaned_or_raw()
    out = build_pairs_by_year(df)
    out.to_csv(OUT_PAIRS_BY_YEAR, index=False)
    print(f"[OK] Wrote {OUT_PAIRS_BY_YEAR} with {len(out)} rows.")


if __name__ == "__main__":
    main()
