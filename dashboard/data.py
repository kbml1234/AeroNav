"""Loads every one-time test's flight CSV into one table for the dashboard."""
from pathlib import Path

import pandas as pd

RESULTS = Path(__file__).resolve().parent.parent / "results"

EXPERIMENTS = {
    "Main test: random gusts (8 × 8)": ("03_main_test/final_test_flights.csv", 8, "30,000–30,299", "gust"),
    "Known wind (8 × 8)": ("04_wind/wind_test_flights.csv", 8, "40,000–40,299", "wind"),
    "Diamond vision (8 × 8)": ("05_vision/vision_test_flights.csv", 8, "50,000–50,299", "wind"),
    "16 × 16 probe 1: old reward": ("06_scale16/scale_test_flights.csv", 16, "60,000–60,299", None),
    "16 × 16 probe 2: reward fix": ("07_reward16/reward_test_flights.csv", 16, "62,000–62,299", None),
    "16 × 16 probe 3: buffer 5M": ("08_buffer16/buffer_test_flights.csv", 16, "63,000–63,299", None),
    "16 × 16 probe 4: three buffer sizes": ("09_middle16/middle_test_flights.csv", 16, "64,000–64,299", None),
}


def load_flights():
    tables = []
    for name, (filename, size, maps, kind) in EXPERIMENTS.items():
        df = pd.read_csv(RESULTS / filename, dtype={"train_seed": "string"})
        df = df.rename(columns={"agent": "drone"})
        df["train_seed"] = df["train_seed"].fillna("")
        if kind == "gust":
            df["condition"] = df["gust_%"].astype(int).astype(str) + "% gusts"
            df["strength"] = df["gust_%"].astype(int)
            df["direction"] = ""
        elif kind == "wind":
            df["condition"] = df["strength_%"].astype(int).astype(str) + "% wind"
            df["strength"] = df["strength_%"].astype(int)
        else:
            df["condition"] = "calm air"
            df["strength"] = 0
            df["direction"] = ""
        df["experiment"] = name
        df["success"] = df["ending"] == "goal"
        tables.append(df[["experiment", "drone", "train_seed", "condition", "strength", "direction",
                          "map_seed", "ending", "success", "moves", "efficiency"]])
    return pd.concat(tables, ignore_index=True)
