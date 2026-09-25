"""Logic invariants for src/safety, dyno, reward, rl on real runs (no VLM, no writes).

    python tests/test_logic.py [GOOD_RUN] [FIXTURE ...]
or set CLIMB_GOOD_RUN / CLIMB_FIXTURES (os.pathsep-separated). Missing paths are skipped.
"""
import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src import dyno, reward, rl, safety  # noqa: E402

CACHE = ROOT / "data" / "cache"
SCRATCH = ROOT / "data" / "fixtures"   # local, gitignored: runs kept as edge cases
GOOD = os.environ.get("CLIMB_GOOD_RUN", str(ROOT / "data/output/20260924-201405"))
FIXTURES = os.environ.get("CLIMB_FIXTURES", os.pathsep.join(str(SCRATCH / n) for n in ("IMG_0508_green", "IMG_0508_yellow"))).split(os.pathsep)


def pipeline(d: Path):
    """Mirror safety_report.py steps 2, 4, 4b without VLM or file writes."""
    r = safety.analyze(d, CACHE, None)
    if r.get("floor_edge_y"):
        edge = r["floor_edge_y"]
        poly = [[float(x), float(y)] for x, y in zip(np.linspace(0, 1, len(edge)), edge)] + [[1, 1], [0, 1]]
        r = safety.analyze(d, CACHE, [poly])
    t = safety.load_track(d)
    holds = safety.Holds(json.loads((d / "holds.json").read_text()), t.size)
    sc = safety.body_scale(t)
    j = r["judge"]
    a0 = j["start"]["frame"] if j.get("start") else 0
    a1 = int((j["top"]["t"] or t.n / t.fps) * t.fps)
    r["dyno"] = dyno.analyze(t, holds, sc, (a0, min(t.n - 1, a1 + int(0.5 * t.fps))))
    for c in r["dyno"]["candidates"]:
        c.pop("_frame", None)
    r["reward"] = reward.score_demo(r, t, holds, sc)
    env, human = rl.build(r, holds, sc, t)
    r["rl"] = rl.train(env, human=human)
    return r, t, holds, sc, env, human


def common(name, r, t, holds, sc, env, human):
    json.dumps(r)                                   # JSON-safe
    j, rw, q = r["judge"], r["reward"], r["rl"]
    assert j["result"] in ("TOP", "ZONE", "NO SCORE")
    assert j["attempts"] >= 1
    assert j["zone"]["hold"] in holds.ids and j["top"]["hold"] in holds.ids
    if len(holds.ids) >= 3:
        assert j["zone"]["hold"] != j["top"]["hold"], "zone must sit below the top"
    if j["result"] == "TOP":
        assert j["top"]["reached"] and j["top"]["controlled"]
    for f in r["falls"]:
        assert f["kind"] in ("fall", "jump_off", "downclimb")
        assert f["t_start"] <= f["t_impact"] and 0 <= f["frame_impact"] < t.n
        assert f["drop_body_lengths"] > 0, "y-down: a descent must drop the hips"
        assert 0 <= f["landing_score"] <= 100
        if f["kind"] == "jump_off":
            assert j["top"]["controlled"]
    fe = r["feet"]
    if fe["score"] is not None:
        assert 0 <= fe["score"] <= 100 and fe["grade"] in "ABCDE"
        for p in fe["placements"]:
            assert p["t_on"] <= p["t_off"] and 0 <= p["precision"] <= 1 and p["hold"] in holds.ids
    # reward: total == sum(terms), curve ends at total, signs
    assert abs(rw["total"] - sum(rw["terms"].values())) < 1e-2, (rw["total"], rw["terms"])
    assert abs(rw["curve"][-1][1] - rw["total"]) < 0.05, (rw["curve"][-1], rw["total"])
    for k in ("reach", "dyno_risk", "campus", "footwork", "fall", "time"):
        assert rw["terms"][k] <= 0, k
    for k in ("zone", "top"):
        assert rw["terms"][k] >= 0, k
    assert (rw["terms"]["top"] > 0) == (j["result"] == "TOP")
    assert rw["terms"]["fall"] == reward.WEIGHTS["fall"] * sum(f["kind"] == "fall" for f in r["falls"])
    # dyno classify: direction matters (y grows down), monotone in gap
    e = {"static_span_bl": r["dyno"]["static_span_bl"], "dyno_gain_bl": r["dyno"]["dyno_gain_bl"]}
    g = (e["static_span_bl"] + 0.5 * e["dyno_gain_bl"]) * sc["body"]
    up, down = dyno.classify(np.array([0, -g]), e, sc["body"]), dyno.classify(np.array([0, g]), e, sc["body"])
    assert up[1] > down[1], "upward gap must get more dyno margin than downward"
    assert dyno.classify(np.array([0, -0.1 * sc["body"]]), e, sc["body"])[0] == "static"
    # RL: legality of the learned beta, deterministic training, human return consistency
    for b in q["learned_beta"]:
        assert b["needed"] != "out_of_reach"
    q2 = rl.train(rl.build(r, holds, sc, t)[0], human=human)
    assert q2["learned_return"] == q["learned_return"] and q2["policy_top_rate"] == q["policy_top_rate"], "seeded RL must be deterministic"
    hret, _ = rl.replay(env, *human)
    assert hret == q["human_return"]
    if not j["top"]["controlled"]:
        assert q["human_return"] < reward.WEIGHTS["top"] + reward.WEIGHTS["progress"] + reward.WEIGHTS["zone"], "no top bonus without a controlled top"
    print(f"  ok {name}: {j['result']} att={j['attempts']} falls={[f['kind'] for f in r['falls']]} "
          f"feet={fe['score']} {fe['grade']} reward={rw['total']:+.2f} rl top={q['policy_top_rate']:.2f} "
          f"learned={q['learned_return']:+.2f} human={q['human_return']:+.2f}")


def main():
    args = sys.argv[1:]
    good = Path(args[0]) if args else Path(GOOD)
    fixtures = [Path(a) for a in args[1:]] if len(args) > 1 else [Path(f) for f in FIXTURES if f]
    ran = 0
    if (good / "poses.json").exists():
        r, t, holds, sc, env, human = pipeline(good)
        j = r["judge"]
        assert j["result"] == "TOP" and j["attempts"] == 1, j
        assert abs(j["top"]["t"] - 19.8) < 0.5, j["top"]
        assert [f["kind"] for f in r["falls"]] == ["jump_off"], r["falls"]
        assert abs(r["falls"][0]["t_impact"] - 25.06) < 0.2
        assert r["rl"]["policy_top_rate"] > 0.9 and r["rl"]["learned_topped"]
        assert r["reward"]["terms"]["top"] > 0 and r["reward"]["terms"]["fall"] == 0
        common("good", r, t, holds, sc, env, human)
        ran += 1
    else:
        print(f"  skip good run (missing): {good}")
    for fx in fixtures:
        if not (fx / "poses.json").exists():
            print(f"  skip fixture (missing): {fx}")
            continue
        r, t, holds, sc, env, human = pipeline(fx)
        assert r["judge"]["result"] != "TOP", r["judge"]["result"]
        common(fx.name, r, t, holds, sc, env, human)
        ran += 1
    print(f"PASS ({ran} run(s))")


if __name__ == "__main__":
    main()
