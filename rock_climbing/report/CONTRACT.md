# safety.json contract (Climb Vision Judge: judge + landing safety + silent feet)

Produced by `safety_report.py <run_dir>`; written to `<run_dir>/safety.json` and inlined
into `<run_dir>/report.html` by replacing the literal token `/*__SAFETY_DATA__*/null`
in `report/template.html` with the JSON.

All `*_xy` are normalized [0..1] image coords, origin top-left, y grows DOWN.
Times are seconds from clip start.

```jsonc
{
  "clip": "IMG_0508",
  "video": "IMG_0508_climb.mp4",        // relative to run_dir (side-by-side render)
  "holds_image": "holds.png",           // relative to run_dir
  "fps": 29.97, "frame_count": 965, "size": [608, 1080],
  "scale_px": {"torso": 97.7, "thigh": 82.6, "shin": 75.1, "body": 289.7},   // pixels; body = 1.35*torso + thigh + shin
  "route": {"color": "blue", "grade": "VB", "holds": 17},
  "holds": [{"id": 1, "centroid_xy": [0.23, 0.55], "polygon": [[x,y],...]}],

  "judge": {
    "result": "TOP",                    // "TOP" | "ZONE" | "NO SCORE"
    "attempts": 1,
    "start": {"t": 5.97, "frame": 179, "holds": [7, 8]},
    "zone":  {"hold": 9, "reached": true, "t": 9.1},
    "top":   {"hold": 17, "reached": true, "controlled": true, "t": 20.4, "hold_s": 1.2,
             "note": null},   // set when control was borrowed from the route read (top hold at the
                              // frame edge): the page then shows "controlled (from route read; ...)"
                              // instead of the hold_s seconds, and the coach is told hold_s was not measured
    "events": [{"t": 5.97, "label": "Start: both hands on start holds, feet off the mat", "kind": "start"}]
  },

  "pad": {                              // crash pad / landing zone, from SAM 3.1 (or floor fallback)
    "source": "sam3.1",                 // "sam3.1" | "floor-fallback"
    "prompt": "crash pad",
    "polygons": [[[x,y],...]]
  },

  "falls": [{
    "id": 1,
    "kind": "fall",                     // "fall" | "jump_off" | "downclimb"
    "t_start": 20.9, "t_impact": 21.5, "frame_impact": 644,
    "drop_body_lengths": 1.4,
    "impact_speed_bl_s": 3.1,           // body-lengths per second at impact
    "knee_angle_min": {"left": 112, "right": 118},   // degrees, lower = more bend
    "feet_on_pad": [true, true],        // [left, right]; null if no pad
    "hands_posted": false,              // wrists below hips / reaching to the ground
    "landing_score": 86,                // 0..100
    "flags": ["Stiff left knee"],       // human-readable issues
    "keyframe": "fall_1.jpg",           // relative to run_dir, may be null
    "ankle_xy": [[0.4, 0.62], [0.47, 0.62]],
    "verdict": "Solid two-foot landing ...",  // from a VLM on the gateway, may be null
    "vlm": {"on_pad": true, "posture": "...", "agrees_with_metrics": true}
                                              // optional; agrees_with_metrics=false -> "VLM disagrees" tag
  }],

  "feet": {
    "score": 78, "grade": "B",          // silent-feet score, 0..100
    "placements": [{
      "foot": "left", "hold": 1, "t_on": 5.9, "t_off": 7.8,
      "readjusts": 0, "jitter": 0.004,  // jitter in body-lengths (std of ankle after settling)
      "impact_speed": 0.6,              // body-lengths/s in the 3 frames before contact
      "precision": 0.9,                 // 1 = ankle lands at the hold centre
      "score": 88
    }],
    "per_foot": {"left": {"score": 80, "placements": 4, "readjusts": 1},
                 "right": {"score": 76, "placements": 4, "readjusts": 2}},
    "worst": [3, 5]                     // indices into placements
  },

  "dyno": {                             // "is a dyno possible here?" per gap; may be null
    "assumptions": {"height_m": 1.75, "takeoff_speed_m_s": 2.2},
    "static_span_bl": 0.62,             // longest hand-to-hand span seen on this climb, body-lengths
    "dyno_gain_bl": 0.31,               // extra reach a dyno buys (leg drive + flight), body-lengths
    "candidates": [{
      "id": 1,
      "from_hold": 12, "to_hold": 15, "hand": "left",
      "t": 18.2,                        // when the climber was set up on from_hold
      "skip": [13, 14],                 // holds skipped (empty = a move actually climbed)
      "climbed": false,                 // true if this is a move the climber actually made
      "gap_bl": 0.78, "dx_bl": 0.05, "dy_bl": -0.77,   // dy < 0 = target is above
      "needed": "dyno",                 // "static" | "deadpoint" | "dyno" | "out_of_reach"
      "margin_bl": 0.15,                // reach left over (negative = short)
      "target_size": "small",           // "small" | "medium" | "large" catch surface
      "frame": "dyno_1.jpg",            // annotated frame (from/to holds + arrow), may be null
      "vlm": {"feasible": true, "style": "dyno", "confidence": 0.7,
              "reason": "<=30 words", "agrees_with_metrics": true}
                                        // false -> "VLM disagrees" tag; gateway VLM read of the annotated frame, may be null
    }]
  },

  "reward": {                           // src/reward.py applied to the recorded climb; may be null
    "weights": {"progress": 10.0, "zone": 3.0, "top": 10.0, "reach": -0.5, "dyno_risk": -1.0,
                "campus": -0.2, "footwork": -0.1, "fall": -10.0, "landing": 1.0, "time": -0.01},
    "total": 18.79,
    "terms": {"progress": 6.334, "zone": 3.0, "top": 10.0, "reach": -0.406, "dyno_risk": 0.0,
              "campus": 0.0, "footwork": -1.0, "fall": 0.0, "landing": 1.0, "time": -0.138},
    "curve": [[6.01, -0.1], [6.21, -0.102]],          // [t, cumulative reward] at 5 Hz, start -> end of events
    "events": [{"t": 6.97, "term": "progress", "value": 0.854,
                "label": "right hand #7→#9: height 0.45"}],   // every non-zero scored event, time order
    "hand_start": [7, 7],                             // [left, right] start holds
    "hand_moves": [{"hand": "right", "from": 7, "to": 9, "anchor": 7, "t": 6.97,
                    "gap_bl": 0.188, "needed": "static", "margin_bl": 0.839}]   // the human hand sequence
  },

  "rl": {                               // src/rl.py: Q-learning on a hands-only route model; may be null
    "algo": "tabular Q-learning", "episodes": 4000, "seed": 7, "alpha": 0.2, "gamma": 0.97,
    "epsilon": "1.0 -> 0.05 linear over 80% of episodes",
    "state_space": "17x17 (left-hand hold, right-hand hold) = 289",
    "action_space": "2 hands x 17 holds = 34 (masked to reachable)",
    "move_s": 1.5, "max_steps": 40, "fail_p": {"deadpoint": 0.15, "dyno": 0.35},   // x (1 - margin/dyno_gain)
    "training_curve": [[66, -7.852], [132, -6.005], [3960, 18.563]],   // [episode, mean return over the window], ~60 points
    "learned_beta": [{"hand": "right", "from": 7, "to": 11, "needed": "static", "reward": 4.853}],
    "learned_return": 18.747, "learned_moves": 4, "learned_topped": true,
    "policy_top_rate": 1.0,             // greedy policy, 200 rollouts with the stochastic failure
    "policy_mean_return": 18.747,
    "human_beta": [{"hand": "left", "from": 16, "to": 17, "needed": "static", "reward": 9.985,
                    "t": 19.79}],       // "inferred": true on a top match the pose could not see
    "human_return": 18.691, "human_moves": 16   // the same env reward along the human's hand sequence
  },

  "coach": "One-paragraph overall coaching note from the gateway VLM (may be null)",
  "cost": {"sam_pad": 0.012, "vlm": 0.004},   // vlm includes fall + dyno verdicts and the coach note
  "generated_at": "2026-09-24T20:40:00"
}
```
