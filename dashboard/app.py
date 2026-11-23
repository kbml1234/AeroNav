
import itertools
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import altair as alt
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from dashboard.data import EXPERIMENTS, load_flights
from dashboard.draw import draw_flight, draw_vision
from dashboard.fly import DIRECTIONS, DRONES, fly, input_labels
from src.astar import astar
from src.stats import paired_counts, paired_p_value, seed_spread, wilson_interval

st.set_page_config(page_title="AeroNav", layout="wide")

MAIN = "Main test: random gusts (8 × 8)"


@st.cache_data
def flights():
    return load_flights()


def summarise(df):
    rows = []
    for (drone, cond), g in df.groupby(["drone", "condition"], sort=False):
        n, wins = len(g), int(g["success"].sum())
        lo, hi = wilson_interval(wins, n)
        seeds = [s for s in g["train_seed"].unique() if s != ""]
        per_seed = [100 * g[g["train_seed"] == s]["success"].mean() for s in seeds]
        _, smin, smax = seed_spread(per_seed) if per_seed else (None, None, None)
        goals = g[g["success"]]
        rows.append({"drone": drone, "condition": cond, "strength": int(g["strength"].iloc[0]), "n": n,
                     "success %": round(100 * wins / n, 1), "95% CI low": round(100 * lo, 1),
                     "95% CI high": round(100 * hi, 1),
                     "seed min %": None if smin is None else round(smin, 1),
                     "seed max %": None if smax is None else round(smax, 1),
                     "crash %": round(100 * (g["ending"] == "collision").mean(), 1),
                     "timeout %": round(100 * (g["ending"] == "timeout").mean(), 1),
                     "mean moves (successes)": round(goals["moves"].mean(), 2) if len(goals) else None,
                     "route efficiency (successes)": round(goals["efficiency"].mean(), 3) if len(goals) else None})
    return pd.DataFrame(rows)


def compare(df, a, b, cond):
    da = df[(df["drone"] == a) & (df["condition"] == cond)]
    db = df[(df["drone"] == b) & (df["condition"] == cond)]
    keys = ["map_seed", "direction"]
    if (da["train_seed"] != "").all() and (db["train_seed"] != "").all():
        keys.append("train_seed")
    m = da.merge(db, on=keys, suffixes=("_a", "_b"))
    only_a, only_b = paired_counts(list(m["success_a"]), list(m["success_b"]))

    def seed_range(d):
        per_seed = [100 * g["success"].mean() for _, g in d.groupby("train_seed")]
        return min(per_seed), max(per_seed)

    (a_lo, a_hi), (b_lo, b_hi) = seed_range(da), seed_range(db)
    return {"A": a, "B": b, "condition": cond, "pairs": len(m),
            "A success %": round(100 * m["success_a"].mean(), 1), "B success %": round(100 * m["success_b"].mean(), 1),
            "only A succeeded": only_a, "only B succeeded": only_b,
            "p": paired_p_value(only_a, only_b), "p (exact McNemar)": f"{paired_p_value(only_a, only_b):.3g}",
            "A seed range %": f"{a_lo:.1f}–{a_hi:.1f}", "B seed range %": f"{b_lo:.1f}–{b_hi:.1f}",
            "seed ranges overlap": not (a_lo > b_hi or b_lo > a_hi)}


def verdict(row):
    if row["p"] >= 0.05:
        return "no clear difference"
    better = "A" if row["only A succeeded"] > row["only B succeeded"] else "B"
    if row["seed ranges overlap"]:
        return f"{better} higher on these maps (p < 0.05), but seed ranges overlap: not clear"
    return f"{better} better (p < 0.05, seed ranges do not overlap)"


def rate(df, drone, cond):
    g = df[(df["drone"] == drone) & (df["condition"] == cond)]
    return 100 * g["success"].mean(), len(g)


data = flights()
exp = {name: data[data["experiment"] == name] for name in EXPERIMENTS}
main = exp[MAIN]

st.title("AeroNav")
st.write("Learned (DQN) versus planned (A\\*) navigation for a simulated drone on grid maps it has never seen. "
         f"All results below come from {len(data):,} test flights on held-out maps, each test run once under a "
         "protocol written before the run.")

key = []
for label, e, drone, cond in [
        ("Calm DQN, unseen 8 × 8 maps, calm air", MAIN, "calm DQN", "0% gusts"),
        ("A* (full map), unseen 8 × 8 maps, calm air", MAIN, "A*", "0% gusts"),
        ("Gust DQN, 20% gusts", MAIN, "gust DQN", "20% gusts"),
        ("Calm DQN, 20% gusts", MAIN, "calm DQN", "20% gusts"),
        ("16 × 16, diamond vision, original reward", "16 × 16 probe 1: old reward", "S2 diamond", "calm air"),
        ("16 × 16, diamond vision, reward fix", "16 × 16 probe 2: reward fix", "R2 diamond", "calm air")]:
    g = exp[e][(exp[e]["drone"] == drone) & (exp[e]["condition"] == cond)]
    lo, hi = wilson_interval(int(g["success"].sum()), len(g))
    key.append({"result": label, "success %": round(100 * g["success"].mean(), 1), "n": len(g),
                "95% CI": f"{100 * lo:.1f}–{100 * hi:.1f}"})
st.dataframe(pd.DataFrame(key), hide_index=True, width="content")

tab1, tab2, tab3, tab4, tab5 = st.tabs(["Question", "Drones", "Replay", "Results", "Limitations"])

with tab1:
    st.subheader("Research question")
    st.write("How does a DQN drone that senses only the squares next to it perform on new, never-seen maps, "
             "compared with A\\*, which sees the whole map? Twist: does training the drone in random gusts make it "
             "better in gusts, and does that cost anything in calm air?")
    st.write("Setup: 8 × 8 grid, 15% walls, start (0, 0), goal (7, 7), 4 moves. Reward +100 goal, −30 crash, "
             "−1 per move, 200-move limit. 300 held-out test maps (seeds 30,000–30,299). DQN results pool 3 "
             "training seeds (n = 900 flights per condition); A\\* n = 300.")

    st.subheader("Hypotheses")
    rows = []
    h1 = compare(main, "calm DQN", "A*", "0% gusts")
    rows.append({"hypothesis": "H1: the DQN succeeds on most unseen maps, below A*", **h1})
    for cond in ["5% gusts", "10% gusts", "20% gusts"]:
        rows.append({"hypothesis": "H2a: the gust DQN succeeds more in gusts", **compare(main, "gust DQN", "calm DQN", cond)})
    rows.append({"hypothesis": "H2b: the gust DQN succeeds slightly less in calm air",
                 **compare(main, "gust DQN", "calm DQN", "0% gusts")})
    hyp = pd.DataFrame(rows)
    hyp["result"] = hyp.apply(verdict, axis=1)
    st.dataframe(hyp[["hypothesis", "A", "B", "condition", "pairs", "A success %", "B success %", "only A succeeded",
                      "only B succeeded", "p (exact McNemar)", "A seed range %", "B seed range %", "result"]],
                 hide_index=True, width="stretch")
    st.write("H1 supported: 91.9% vs 100%, and in calm air all 24 failures of the seed-0 DQN were loops (12 into a dead "
             "end it could not sense). H2a not supported at any gust level. H2b: the gust DQN was lower in calm air on "
             "these maps (p = 0.015), but one calm-trained and one gust-trained seed overlap, so training luck could explain it.")
    fail = main[(main["ending"] != "goal")].groupby(["drone", "condition", "ending"]).size().unstack(fill_value=0)
    st.write("How the failed flights ended (all seeds):")
    st.dataframe(fail.reset_index(), hide_index=True, width="content")

    st.subheader("Follow-up experiments")
    fu = [("Known wind: do wind inputs help?", "Known wind (8 × 8)", "A wind-aware", "B wind-blind", "30% wind"),
          ("Diamond vision: does seeing 2 squares help? (with wind inputs)", "Diamond vision (8 × 8)", "A2 diamond+wind", "A wind-aware", "30% wind"),
          ("Diamond vision: does seeing 2 squares help? (no wind inputs)", "Diamond vision (8 × 8)", "B2 diamond", "B wind-blind", "30% wind"),
          ("Diamond vision: do wind inputs help now?", "Diamond vision (8 × 8)", "A2 diamond+wind", "B2 diamond", "30% wind"),
          ("16 × 16: plus vs diamond vision, reward fix", "16 × 16 probe 2: reward fix", "R2 diamond", "R1 plus", "calm air"),
          ("16 × 16: buffer 50,000 vs 1,500,000", "16 × 16 probe 4: three buffer sizes", "R2 diamond", "M2 diamond", "calm air"),
          ("16 × 16: buffer 50,000 vs 5,000,000", "16 × 16 probe 4: three buffer sizes", "R2 diamond", "B2 diamond", "calm air")]
    rows = []
    for q, e, a, b, cond in fu:
        r = compare(exp[e], a, b, cond)
        rows.append({"question": q, **r, "result": verdict(r)})
    st.dataframe(pd.DataFrame(rows).drop(columns=["condition", "p", "seed ranges overlap"]), hide_index=True, width="stretch")
    s2, _ = rate(exp["16 × 16 probe 1: old reward"], "S2 diamond", "calm air")
    r2, _ = rate(exp["16 × 16 probe 2: reward fix"], "R2 diamond", "calm air")
    st.write(f"16 × 16 reward fix (different map sets, so not paired): diamond-vision success {s2:.1f}% with the original "
             f"reward (every flight crashed on move 1) vs {r2:.1f}% with the fix (0 crashes). "
             "Pairing note: where both drones have training seeds, seed 0 is paired with seed 0, 1 with 1, 2 with 2.")

with tab2:
    st.subheader("Drones")
    calm_rate = lambda e, d, c: f"{rate(exp[e], d, c)[0]:.1f}"
    st.dataframe(pd.DataFrame([
        {"drone": "A*", "inputs": "whole map", "training": "none (search)", "map": "8 × 8 / 16 × 16",
         "test success, calm air %": calm_rate(MAIN, "A*", "0% gusts")},
        {"drone": "Calm DQN", "inputs": "12: position (2), goal (2), 4 wall flags, last move (4)", "training": "500,000 moves, calm",
         "map": "8 × 8", "test success, calm air %": calm_rate(MAIN, "calm DQN", "0% gusts")},
        {"drone": "Gust DQN", "inputs": "12 (same)", "training": "500,000 moves, 10% gusts", "map": "8 × 8",
         "test success, calm air %": calm_rate(MAIN, "gust DQN", "0% gusts")},
        {"drone": "Wind-aware DQN (A)", "inputs": "17: + wind direction (4) and strength (1)", "training": "500,000 moves, wind schedule",
         "map": "8 × 8", "test success, calm air %": calm_rate("Known wind (8 × 8)", "A wind-aware", "0% wind")},
        {"drone": "Diamond DQN (B2)", "inputs": "20: 12 wall flags instead of 4", "training": "500,000 moves, wind schedule",
         "map": "8 × 8", "test success, calm air %": calm_rate("Diamond vision (8 × 8)", "B2 diamond", "0% wind")},
        {"drone": "16 × 16 DQN, reward fix (R2)", "inputs": "20 (diamond)", "training": "5,000,000 moves, buffer 50,000",
         "map": "16 × 16", "test success, calm air %": calm_rate("16 × 16 probe 2: reward fix", "R2 diamond", "calm air")},
    ]), hide_index=True, width="stretch")
    st.write("Network for every DQN: inputs → 64 → 64 → 4 Q-values (one per move). Learning rate 0.0003, discount 0.99, "
             "batch 64, replay buffer 50,000 moves unless stated, exploration falling from 1.0 to 0.05 over the first 30% "
             "of training. The final model is always the last checkpoint.")
    c1, c2, c3 = st.columns([1, 1, 3])
    with c1:
        st.caption("Plus vision: 4 squares")
        fig = draw_vision("plus"); st.pyplot(fig, width="content"); plt.close(fig)
    with c2:
        st.caption("Diamond vision: 12 squares")
        fig = draw_vision("diamond"); st.pyplot(fig, width="content"); plt.close(fig)
    with c3:
        st.caption("Reward per move")
        st.dataframe(pd.DataFrame([
            {"event": "move closer to the goal", "original": "−1", "reward fix (16 × 16)": "+9"},
            {"event": "move farther from the goal", "original": "−1", "reward fix (16 × 16)": "−11"},
            {"event": "reach the goal", "original": "+100", "reward fix (16 × 16)": "+110"},
            {"event": "crash", "original": "−30", "reward fix (16 × 16)": "−401"},
            {"event": "run out of moves (whole flight)", "original": "−200 (8 × 8) / −400 (16 × 16)", "reward fix (16 × 16)": "about −400"},
        ]), hide_index=True, width="content")

with tab3:
    st.subheader("Replay a flight")
    st.write("Flies the saved model (training seed 0) in the simulator. The same map and conditions always give the same flight.")
    c1, c2, c3, c4 = st.columns([2, 1, 1, 1])
    drone = c1.selectbox("Drone", list(DRONES))
    size = DRONES[drone]["size"]
    map_seed = c2.number_input("Map seed", min_value=0, max_value=10**7, value=30000 if size == 8 else 62000, step=1,
                               help="8 × 8 test maps: 30,000–30,299. 16 × 16 test maps: 62,000–62,299.")
    weather = c3.selectbox("Conditions", ["calm", "random gusts", "steady wind"] if size == 8 else ["calm"])
    gust_pct, wind_dir, wind_pct = 0, None, 0
    if weather == "random gusts":
        gust_pct = c4.selectbox("Gust probability %", [0, 5, 10, 20], index=2)
    elif weather == "steady wind":
        wind_dir = DIRECTIONS.index(c4.selectbox("Wind pushes", DIRECTIONS))
        wind_pct = c4.selectbox("Wind strength %", [0, 10, 20, 30], index=2)
    run_col, anim_col = st.columns([1, 5])
    animate = anim_col.checkbox("Animate", value=True)
    if run_col.button("Run", type="primary"):
        st.session_state["flight"] = fly(drone, int(map_seed), gust_pct, wind_dir, wind_pct)
        st.session_state["animate"] = animate

    f = st.session_state.get("flight")
    if f:
        n_steps = len(f["steps"]) - 1
        route = astar(f["grid"], f["start"], f["goal"])
        shortest = len(route) - 1 if route else None
        left, right = st.columns([3, 2])
        picture = left.empty()
        if st.session_state.get("animate"):
            for i in range(n_steps + 1):
                fig = draw_flight(f, upto=i)
                picture.pyplot(fig, width="content"); plt.close(fig)
                time.sleep(0.12 if f["size"] == 8 else 0.05)
            st.session_state["animate"] = False
        with right:
            pushes = sum(s["event"] in ("gust", "wind replace", "wind drift", "gust crash", "wind crash") for s in f["steps"])
            st.dataframe(pd.DataFrame([
                {"measure": "ending", "value": f["ending"]},
                {"measure": "moves flown", "value": n_steps},
                {"measure": "shortest route (A*, no gusts)", "value": shortest},
                {"measure": "route efficiency", "value": round(shortest / n_steps, 3) if f["ending"] == "goal" else "n/a (not a success)"},
                {"measure": "moves changed by gust or wind", "value": pushes},
                {"measure": "total reward", "value": round(sum(s.get("reward", 0) for s in f["steps"]), 1)},
            ]).astype(str), hide_index=True, width="stretch")
            step = st.slider("Step", 0, n_steps, n_steps)
        fig = draw_flight(f, upto=step)
        picture.pyplot(fig, width="content"); plt.close(fig)
        left.caption("S start, G goal, line = path, dot = drone, outlined squares = squares the drone senses now, "
                     "orange squares = move changed by a gust or wind, x = crash.")
        with right:
            s = f["steps"][step]
            names = ["N", "E", "S", "W"]
            if step > 0:
                st.write(f"Step {step}: requested {names[s['requested']]}, executed {names[s['executed']]}, "
                         f"event: {s['event']}, reward {s['reward']:+.0f}.")
            st.dataframe(pd.DataFrame({"input": input_labels(f["vision"], f["wind_inputs"]),
                                       "value": [round(v, 3) for v in s["obs"]]}),
                         hide_index=True, height=320, width="stretch")

with tab4:
    st.subheader("Results")
    name = st.selectbox("Experiment", list(EXPERIMENTS))
    _, size, maps, kind = EXPERIMENTS[name]
    df = exp[name]
    table = summarise(df).sort_values(["strength", "drone"])
    st.write(f"{size} × {size} maps, test seeds {maps}, run once. n = flights (maps × training seeds"
             f"{' × 4 wind directions' if kind == 'wind' else ''}).")

    order = list(dict.fromkeys(table["condition"]))
    base = alt.Chart(table).encode(x=alt.X("condition:N", sort=order, title=None, axis=alt.Axis(labelAngle=0)),
                                   xOffset=alt.XOffset("drone:N"),
                                   color=alt.Color("drone:N", title="drone", scale=alt.Scale(scheme="tableau10")))
    bars = base.mark_bar().encode(
        y=alt.Y("success %:Q", title="success rate (%)", scale=alt.Scale(domain=[0, 100])),
        tooltip=["drone", "condition", "n", "success %", "95% CI low", "95% CI high", "seed min %", "seed max %"])
    errs = base.mark_rule(color="black").encode(y="95% CI low:Q", y2="95% CI high:Q")
    st.altair_chart((bars + errs).properties(height=320), width="stretch")
    st.caption("Bars: success rate, all training seeds pooled. Black lines: 95% Wilson confidence interval.")
    st.dataframe(table.drop(columns=["strength"]), hide_index=True, width="stretch")
    st.caption("Moves and route efficiency (shortest route ÷ moves flown) are over successful flights only. "
               "Seed min/max: success rate of the worst and best of the 3 training seeds.")

    st.subheader("Same-map comparisons")
    cond = st.selectbox("Condition", order)
    drones = list(table["drone"].unique())
    pairs = pd.DataFrame([compare(df, a, b, cond) for a, b in itertools.combinations(drones, 2)])
    pairs["result"] = pairs.apply(verdict, axis=1)
    st.dataframe(pairs.drop(columns=["condition", "p", "seed ranges overlap"]), hide_index=True, width="stretch")

    st.subheader("Methods")
    st.write("""
**95% confidence interval (Wilson score).** For k successes in n flights, the range that would contain the true
success rate in 95% of repeated tests on new maps. Example: 276 of 300 = 92.0%, interval 88.4–94.6%.

**Same-map comparison (exact McNemar test).** All drones flew the same maps with the same gusts or wind. Only maps
where exactly one drone succeeded are informative. If the drones were equally good, each such map would go either way
with probability 1/2; p is the two-sided binomial probability of a split at least as uneven. p < 0.05 is reported as a
clear difference.

**Seed spread.** Each DQN was trained with 3 random seeds. The interval and the McNemar test cover variation from
the choice of maps; the seed min/max covers variation from training.

**Rule for "clear".** A difference is reported as clear only if p < 0.05 **and** the 3-seed ranges do not overlap.
The McNemar test alone treats the 900 flights as independent, so it can show that these particular trained models
differ on these maps while saying nothing about whether training them again would give the same order.
""")

with tab5:
    st.subheader("Limitations")
    st.write("""
- 2D grid simulation with 4 moves. No physics, altitude, speed, battery or sensor noise.
- Small maps: 8 × 8 for the main study, 16 × 16 for the exploratory probes.
- A\\* sees the whole map and the DQNs do not, so A\\* is a reference, not an equal competitor.
- With 4 wall flags, 46.6% of situations look identical to a situation on another map that needs a different move.
- Gusts and wind are simple models (a swapped move, a fixed push direction).
- 3 training seeds per DQN.
- 16 × 16 learning was unstable (validation success swung between about 0% and 90% between checkpoints). Replay
  buffer size was ruled out as the cause; learning rate and target-network updates were not tested.
- The 16 × 16 reward fix changed two things at once (crash cost and closer/farther reward).
""")
