"""A hands-only route model and a tabular Q-learning agent that climbs it.

``RouteEnv`` abstracts the recorded route to what the reach envelope can
judge: the state is (left-hand hold, right-hand hold), an action moves one
hand to any hold, and a move is legal only if the target is within
static_span + dyno_gain of the OTHER hand's hold (dyno.classify is not
"out_of_reach"). Feet, body position, momentum and the third dimension are
abstracted away. Step rewards are the same terms as src.reward (progress,
zone, top, reach, dyno_risk, time at ~1.5 s per move), plus a stochastic
failure on deadpoints / dynos so the agent must trade risk against speed.
"""
from __future__ import annotations

import numpy as np

from src import dyno, reward
from src.reward import WEIGHTS

MOVE_S = 1.5             # seconds of wall time per hand move
MAX_STEPS = 40
FAIL_P = {"deadpoint": 0.15, "dyno": 0.35}   # x (1 - margin_norm)


class RouteEnv:
    def __init__(self, holds, env: dict, body: float, start: tuple[int, int], top_id: int,
                 zone_id: int | None, seed: int = 0):
        self.ids = list(holds.ids)
        self.idx = {h: i for i, h in enumerate(self.ids)}
        self.n = len(self.ids)
        self.env, self.body = env, body
        self.top, self.zone = self.idx[top_id], (self.idx[zone_id] if zone_id in self.idx else None)
        self.start = (self.idx[start[0]], self.idx[start[1]])
        H = reward.heights(holds, top_id)
        self.h = np.array([H[i] for i in self.ids])
        # per (anchor, target) pair: class, margin, gap, reach/dyno terms, failure p
        n = self.n
        self.need = np.empty((n, n), dtype=object)
        self.gap = np.zeros((n, n))
        self.margin = np.zeros((n, n))
        self.move_r = np.zeros((n, n))
        self.fail_p = np.zeros((n, n))
        for a, ha in enumerate(self.ids):
            for b, hb in enumerate(self.ids):
                vec = holds.centroid[hb] - holds.centroid[ha]
                need, margin = dyno.classify(vec, env, body)
                self.need[a, b], self.margin[a, b] = need, margin
                self.gap[a, b] = float(np.linalg.norm(vec)) / body
                mt = reward.move_terms(self.gap[a, b], need, margin, env)
                self.move_r[a, b] = mt["reach"] + mt["dyno_risk"]
                if need in FAIL_P:
                    self.fail_p[a, b] = FAIL_P[need] * (1 - float(np.clip(margin / env["dyno_gain_bl"], 0, 1)))
        self.legal_to = self.need != "out_of_reach"          # [anchor, target]
        self.rng = np.random.default_rng(seed)
        self.n_actions = 2 * n

    # state = (l, r) indices; action a = hand * n + target (hand 0 = left)
    def sid(self, s):
        return s[0] * self.n + s[1]

    def legal(self, s) -> np.ndarray:
        l, r = s
        m = np.zeros(self.n_actions, bool)
        m[:self.n] = self.legal_to[r]          # left hand moves, right anchors
        m[self.n:] = self.legal_to[l]
        m[l] = False                            # no-op
        m[self.n + r] = False
        return m

    def reset(self):
        self.s, self.steps, self.zone_hit = self.start, 0, self.zone in self.start
        return self.s

    def step(self, a, stochastic: bool = True):
        hand, tgt = divmod(a, self.n)
        l, r = self.s
        anchor = r if hand == 0 else l
        rew = self.move_r[anchor, tgt] + WEIGHTS["time"] * MOVE_S
        self.steps += 1
        if stochastic and self.fail_p[anchor, tgt] > 0 and self.rng.random() < self.fail_p[anchor, tgt]:
            return self.s, rew + WEIGHTS["fall"], True, {"fell": True}
        ns = (tgt, r) if hand == 0 else (l, tgt)
        rew += WEIGHTS["progress"] * (self.h[list(ns)].max() - self.h[list(self.s)].max())
        if not self.zone_hit and self.zone is not None and tgt == self.zone:
            self.zone_hit = True
            rew += WEIGHTS["zone"]
        self.s = ns
        topped = ns[0] == self.top and ns[1] == self.top
        if topped:
            rew += WEIGHTS["top"]
        return ns, rew, topped or self.steps >= MAX_STEPS, {"fell": False, "topped": topped}

    def describe(self, s, a, r) -> dict:
        hand, tgt = divmod(a, self.n)
        anchor = s[1] if hand == 0 else s[0]
        return {"hand": "left" if hand == 0 else "right", "from": int(self.ids[s[hand]]),
                "to": int(self.ids[tgt]), "anchor": int(self.ids[anchor]),
                "needed": str(self.need[anchor, tgt]), "gap_bl": round(float(self.gap[anchor, tgt]), 3),
                "reward": round(float(r), 3)}


def _rollout(env: RouteEnv, policy, stochastic: bool):
    s, done, ret, beta, info = env.reset(), False, 0.0, [], {}
    while not done:
        a = policy(s)
        ns, r, done, info = env.step(a, stochastic)
        beta.append(env.describe(s, a, r))
        ret += r
        s = ns
    return ret, beta, info


def replay(env: RouteEnv, start_holds, moves: list[dict], top_controlled: bool) -> tuple[float, list[dict]]:
    """Score a recorded hand sequence with the env's reward (no stochastic failure)."""
    env.reset()
    env.s = (env.idx[start_holds[0]], env.idx[start_holds[1]])
    env.zone_hit = env.zone in env.s
    ret, beta, steps = 0.0, [], []
    for m in moves:
        steps.append(((0 if m["hand"] == "left" else 1) * env.n + env.idx[m["to"]], m.get("t")))
    # the finish hold can sit off the frame edge; if the judge saw a controlled top,
    # finish with the match the pose could not see
    l, r = env.s
    for a, _ in steps:
        hand, tgt = divmod(a, env.n)
        l, r = (tgt, r) if hand == 0 else (l, tgt)
    if top_controlled:
        if l != env.top:
            steps.append((env.top, None))
            l = env.top
        if r != env.top:
            steps.append((env.n + env.top, None))
    for a, tt in steps:
        s = env.s
        _, rew, done, _ = env.step(a, stochastic=False)
        d = env.describe(s, a, rew)
        d["t"] = tt
        if tt is None:
            d["inferred"] = True
        beta.append(d)
        ret += rew
    return round(float(ret), 3), beta


def train(env: RouteEnv, episodes: int = 4000, alpha: float = 0.2, gamma: float = 0.97, seed: int = 7,
          human: tuple | None = None) -> dict:
    """Tabular Q-learning, epsilon-greedy decaying 1.0 -> 0.05 over 80% of episodes.
    ``human`` = (start_holds, moves, top_controlled) to score the human beta in the same env."""
    rng = np.random.default_rng(seed)
    env.rng = np.random.default_rng(seed + 1)
    nS = env.n * env.n
    Q = np.zeros((nS, env.n_actions))
    masks = {}

    def mask(s):
        k = env.sid(s)
        if k not in masks:
            masks[k] = env.legal(s)
        return masks[k]

    def greedy(s):
        m = mask(s)
        q = np.where(m, Q[env.sid(s)], -np.inf)
        return int(np.argmax(q))

    returns = []
    for ep in range(episodes):
        eps = max(0.05, 1.0 - ep / (0.8 * episodes))
        s, done, ret = env.reset(), False, 0.0
        while not done:
            m = mask(s)
            a = int(rng.choice(np.flatnonzero(m))) if rng.random() < eps else greedy(s)
            ns, r, done, _ = env.step(a)
            target = r if done else r + gamma * np.max(np.where(mask(ns), Q[env.sid(ns)], -np.inf))
            Q[env.sid(s), a] += alpha * (target - Q[env.sid(s), a])
            s, ret = ns, ret + r
        returns.append(ret)

    win = max(1, episodes // 60)
    curve = [[i + win, round(float(np.mean(returns[i:i + win])), 3)] for i in range(0, episodes - win + 1, win)]
    learned_ret, beta, info = _rollout(env, greedy, stochastic=False)
    tops, rets = 0, []
    for _ in range(200):
        r, _, inf = _rollout(env, greedy, stochastic=True)
        tops += bool(inf.get("topped"))
        rets.append(r)
    out = {"algo": "tabular Q-learning", "episodes": episodes, "seed": seed,
           "alpha": alpha, "gamma": gamma, "epsilon": "1.0 -> 0.05 linear over 80% of episodes",
           "state_space": f"{env.n}x{env.n} (left-hand hold, right-hand hold) = {nS}",
           "action_space": f"2 hands x {env.n} holds = {env.n_actions} (masked to reachable)",
           "move_s": MOVE_S, "max_steps": MAX_STEPS, "fail_p": dict(FAIL_P),
           "training_curve": curve,
           "learned_beta": [{k: b[k] for k in ("hand", "from", "to", "needed", "reward")} for b in beta],
           "learned_return": round(float(learned_ret), 3), "learned_moves": len(beta),
           "learned_topped": bool(info.get("topped")),
           "policy_top_rate": round(tops / 200, 3), "policy_mean_return": round(float(np.mean(rets)), 3),
           "human_beta": [], "human_return": None, "human_moves": None}
    if human:
        start_holds, moves, top_ok = human
        hret, hbeta = replay(env, start_holds, moves, top_ok)
        out["human_beta"] = [{k: b.get(k) for k in ("hand", "from", "to", "needed", "reward", "t")}
                             | ({"inferred": True} if b.get("inferred") else {}) for b in hbeta]
        out["human_return"], out["human_moves"] = hret, len(hbeta)
    return out


def build(data: dict, holds, scale: dict, t=None, seed: int = 0) -> tuple[RouteEnv, tuple]:
    """RouteEnv from a safety.json-style dict (+ holds); also the human hand sequence."""
    j, dy = data["judge"], data["dyno"]
    env = {"static_span_bl": dy["static_span_bl"], "dyno_gain_bl": dy["dyno_gain_bl"]}
    rw = data.get("reward") or {}
    if rw.get("hand_start"):
        start, moves = tuple(rw["hand_start"]), rw["hand_moves"]
    elif t is not None:
        t0 = j["start"]["t"] if j.get("start") else 0.0
        t1 = j["top"]["t"] or t.n / t.fps
        start, moves = reward.hand_moves(t, holds, scale, (int(t0 * t.fps), min(t.n - 1, int((t1 + 0.5) * t.fps))),
                                         env, (j.get("start") or {}).get("holds"))
    else:
        sh = (j.get("start") or {}).get("holds") or [min(holds.ids)]
        start, moves = (sh[0], sh[-1]), [c | {"from": c["from_hold"], "to": c["to_hold"]}
                                         for c in dy["candidates"] if c["climbed"]]
    sh = (j.get("start") or {}).get("holds") or list(start)
    renv = RouteEnv(holds, env, scale["body"], (sh[0], sh[-1]), j["top"]["hold"], j["zone"]["hold"], seed)
    top_ok = bool(j["top"].get("reached") and j["top"].get("controlled"))
    return renv, (start, moves, top_ok)
