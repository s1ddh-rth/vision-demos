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
    "top":   {"hold": 17, "reached": true, "controlled": true, "t": 20.4, "hold_s": 1.2},
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
    "verdict": "Solid two-foot landing ..."   // from a VLM on the gateway, may be null
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
              "reason": "<=30 words"}   // gateway VLM read of the annotated frame, may be null
    }]
  },

  "coach": "One-paragraph overall coaching note from the gateway VLM (may be null)",
  "cost": {"sam_pad": 0.012, "vlm": 0.004},
  "generated_at": "2026-09-24T20:40:00"
}
```
