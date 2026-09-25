# Rock Climbing + Computer Vision

Film a boulder problem from a still phone and this reads the route back: which
holds were used, in what order, and how long the send took. On top of that it
judges the attempt IFSC-style, scores every landing and every foot placement,
checks which moves could be dynos, and writes a short coaching note. It all runs
on the [VLM Run Gateway](https://www.vlm.run/gateway):
[`sam3.1`](https://vlm.run/gateway/models/facebook-sam3.1) for the holds, the
floor and the crash pad,
[`vitpose-plus-large`](https://vlm.run/gateway/models/usyd-community-vitpose-plus-large)
for the climber, and `qwen/qwen3.8-27b` for the commentary. A full run costs
about three cents.

<p align="center">
  <img src="readme_images/rock_climbing_demo_thumbnail.jpg" width="600" alt="A completed boulder problem: the left panel holds a card comparing this attempt's sequence of holds against three others, the right panel the segmented route with the holds used lit in order">
</p>

The original demo
([LinkedIn video](https://www.linkedin.com/posts/jeremyipark_computervision-ai-ml-activity-7505763952740478976-cgKm))
shows each hold light up as a hand or foot uses it, then the sequence of holds
and the total time once the climb is done.

## What you can do

| Feature | What you get | Models / tech | Cost |
|---|---|---|---|
| **Route reading** | Holds of one colour segmented, lit in the order used; timings, per-limb contact, `_climb.mp4` render | SAM 3.1 + ViTPose+ Large | ~$0.025 per clip |
| **Attempt comparison** | Several clips of one route aligned hold for hold, with a comparison card | local (sequence alignment) | free |
| **Auto judge** | Attempts, start, zone, top, and whether the top was controlled: `TOP` / `ZONE` / `NO SCORE` | local, from pose | free |
| **Landing safety** | Each descent classed fall / jump-off / downclimb and scored 0–100 (knees, feet on pad, hand posting, feet sync, drop), plus a VLM verdict | SAM 3.1 (pad) + Qwen (verdict) | ~$0.004 pad + fractions of a cent |
| **Silent feet** | Score per foot placement and per foot, graded A–E, worst three flagged | local, from pose | free |
| **Dyno lab** | Reach envelope; every gap classed static / deadpoint / dyno / out of reach; VLM check on the interesting ones | local + Qwen | fractions of a cent |
| **Reward & RL** | One reward function scoring the recorded climb, and a Q-learning agent that finds its own beta on the same route | local (NumPy) | free |
| **Coach note** | A few lines of coaching from the numbers above | Qwen, text only | fractions of a cent |
| **Wall scout** | From wall photos (no climber): which colours separate cleanly and how many holds each route has | SAM 3.1 | ~$0.001 per photo per colour |
| **Dashboard** | Upload, pick clip and colour, run, watch progress, browse / re-run / delete past runs; a **How to run** page at `/guide` | `app.py` (stdlib HTTP server) | the runs it starts |
| **Sharing** | A public link to a report (or, while demoing, the dashboard) | `cloudflared` | free, but see the warning below |

Everything except the gateway calls runs locally, and `--no-vlm` skips the
Qwen and pad calls entirely.

## Quick start

### Setup

1. Clone the [vision-demos](https://github.com/jeremyipark/vision-demos) repo.
2. Sign up at [VLM Run](https://app.vlm.run/sign-in) and copy your API key from
   the [Overview dashboard](https://app.vlm.run/dashboard). It is only shown
   once, when created, so copy it right away (or generate a new one).
3. Add it to a `.env` at the vision-demos root. From `rock_climbing/`:

   ```bash
   cp ../.env.example ../.env      # then paste the key after VLMRUN_API_KEY=
   conda env create -f environment.yml
   conda activate rock_climbing
   ```

### The dashboard (recommended)

```bash
python app.py                  # --port 8000 --host 127.0.0.1 are the defaults
```

Open http://127.0.0.1:8000.

1. **Pick a clip.** Upload one, or drop it in `data/input/current/` (uploads go
   to `data/input/uploads/`). The newest clip is selected for you.
2. **Pick the hold colour** of the route you climbed, and the grade.
3. Choose whether to spend VLM credits, then press **RUN**. The button names the
   clip, colour and grade it is about to run (`Run IMG_8842.mov · blue · VB`),
   so you can check before you spend.

RUN runs `main.py` and then `safety_report.py`, with live progress. If that clip
and colour already have a run, it warns first and offers the existing report. Past runs sit in a
gallery below, each with **Open report**, **Re-run analysis** and **Delete**
(which asks first). The **How to run** page is at http://127.0.0.1:8000/guide.

### The command line

`main.py` reads its settings from [`config.py`](config.py), and the dashboard
drives it through `CVJ_*` environment variables, which you can set yourself:

| Variable | Default | Meaning |
|---|---|---|
| `CVJ_HOLD_COLOR` | `blue` | the route's hold colour |
| `CVJ_ROUTE_GRADE` | `VB` | metadata; recorded, not drawn |
| `CVJ_BATCH_MODE` | `1` | `1`: every clip in the input folder is an attempt; `0`: one clip |
| `CVJ_INPUT_VIDEO` | `data/input/climbing.mov` | the clip, when batch mode is off |
| `CVJ_INPUT_DIR` | `data/input/current` | the folder, when batch mode is on |

```bash
# one clip, blue route (bash; in PowerShell use $env:CVJ_HOLD_COLOR="blue"; ...)
CVJ_BATCH_MODE=0 CVJ_INPUT_VIDEO=data/input/current/IMG_8842.mov CVJ_HOLD_COLOR=blue python main.py

python safety_report.py                     # newest run under data/output/
python safety_report.py <run_dir>           # a specific run (or clip subfolder in batch mode)
python safety_report.py <run_dir> --no-vlm  # no gateway calls: no pad, verdicts, dyno check or coach note

python scout.py                                   # wall photos in data/input/photos/jpg
python scout.py <photos_dir> --colors blue,green,pink
```

Then open `report.html` in the run folder. It plays the `_climb.mp4` beside it,
so if the video won't load from `file://`, serve the folder:

```bash
cd data/output/<run>
python -m http.server 8080                        # http://localhost:8080/report.html
cloudflared tunnel --url http://localhost:8080    # a public link, to share it
```

> **Sharing warning.** Tunnelling port 8080 shares one finished report. If you
> tunnel the dashboard itself (port 8000), **anyone with the link can press RUN
> and spend your VLM credits**. Only do that while you are demoing, and stop the
> tunnel afterwards.

## Filming guide

**tl;dr: keep your phone still using a tripod or a water bottle, and have the
entire route in frame.**

Everything assumes a still camera. For best results:

* Have the entire route visible, from the first hold to the last. **The finish
  hold must be fully in frame**, with room above it for your hands on the match.
* Pick an easy route. For these demo videos, I typically go with a VB or V0.
* Use a tripod, or simply set your phone on a water bottle. Someone can also
  hold the camera, but they should stay as still as possible.
* Try to avoid recording other people, and watch out for anyone walking through
  the frame.
* **One route colour per run.** SAM segments the colour you name, and the
  judge, dyno lab and RL all work on those holds. The wrong colour gives garbage:
  our green and yellow re-runs of a blue route read `NO SCORE` and `ZONE`.

<p align="center">
  <img src="readme_images/rock_climbing_video_setup.jpg" width="320" alt="A climber at the start of a green route, filmed in portrait from a still phone; every green hold from the floor to the finish is in frame">
  <br>
  <em>Example framing of the rock climbing video. Note that all holds are in frame.</em>
</p>

The minimal reproducible example:

1. Go to an easy VB/V0 route.
2. Put your phone on a water bottle, with the entire route in frame.
3. Record yourself completing the route.
4. Trim the video so it starts right before you begin and ends right after you
   finish (this saves on inference time 🙂).
5. Send it to your computer and run it from the dashboard.

## How each analysis works

### Route reading

The hold colour becomes a SAM 3.1 prompt:

```python
HOLD_COLOR = "blue"                      # config.py, or CVJ_HOLD_COLOR
HOLD_PROMPT = "{color} climbing hold"
ROUTE_GRADE = "VB"                       # metadata; recorded, not drawn
```

Point it at another colour and the demo follows a different problem up the same
wall, with no retraining and no new model. That is what SAM 3.1 buys over a
detector fine-tuned on one gym's holds. ViTPose+ tracks the climber, and a hold
counts as used when a hand or foot stays on it. Every other knob lives in
`config.py`, and each run snapshots the ones it used into `run.json`. The full
walk-through, and the knob for each symptom, is in
[route-reading-explained.md](route-reading-explained.md).

### Comparing several attempts

With batch mode on, every clip in `data/input/current/` is read as another
attempt at the same route. One clip behaves like a single run; four clips each
end on a card comparing its holds against the other three. Holds are detected
once per clip and the clips are compared hold for hold, so the camera should not
move within a take or between takes.

### Judge, landings, silent feet

These are heuristics taken from coaching and injury writing, tuned on **one test
clip**. Treat the thresholds as starting points; they live at the top of
[`src/safety.py`](src/safety.py).

**Judge (IFSC-style start / zone / top).**
- An **attempt** is a lift-off: both feet clear of the floor for at least 0.5 s,
  with a hand on a route hold for at least 0.3 s of it. Lift-offs split by less
  than 0.3 s back on the floor merge into one. Pose dropouts under 0.7 s keep the
  last floor state, so a lost frame high on the wall neither ends an attempt nor
  fakes a fall.
- **Start** is the first attempt's lift-off, with the holds under the hands.
- **Zone** is the hold 60% of the way up the numbered route, touched by a hand
  for at least 0.3 s.
- **Top** is both hands on the final hold. It is **controlled** if held for 1 s
  with the hips still (median speed under 0.6 torso-lengths/s). If the hands
  leave the frame on the match, the judge falls back on the route read's own
  top-out and says so in `judge.top.note`.

**Landing safety.** Every lift-off that ends back on the floor is examined.
- **Kind:** hips moving down faster than 1.5 body-lengths/s in the last 0.35 s
  is free flight; slower is a `downclimb`. Free flight after a controlled top is
  a `jump_off`, otherwise a `fall`. Drops under 0.18 body-lengths are ignored.
- **Drop** is hip height from the last hand-on-hold frame to impact, in
  body-lengths; **impact speed** is the peak hip speed in the 3 frames before it.
- The landing score starts at 100:

| Check | Rule | Penalty |
|---|---|---|
| Knees | smallest hip-knee-ankle angle in the 0.35 s after impact; over 150° is stiff, 120°–150° is partial | −20 stiff, up to −10 partial |
| Feet on pad | each toe inside the pad polygon (2.5% of frame height tolerance) | −20 per foot off |
| Hand posting | a wrist below the hips and at the floor, around impact | −20 |
| Feet sync | the two feet touch down more than 3 frames (~100 ms) apart | −10 |
| Drop | over 1 body-length and faster than 3 body-lengths/s | −5 |

A downclimb gets +5. The crash pad comes from SAM 3.1 on one clean frame; each
landing also gets a Qwen verdict from 4 frames around impact, and
`agrees_with_metrics` flags when it disagrees with the numbers.

**Silent feet.** Each foot placement on a hold (toe within 0.3 shin-lengths of
the outline, held at least 0.25 s) is scored from 100:
- **Readjusts**, −15 each, up to 3: re-placing on the same hold within 1 s, or a
  wiggle faster than 0.8 shins/s after a 0.2 s settle.
- **Jitter**, up to −25: how much the ankle drifts once settled.
- **Impact speed**, up to −20: foot speed over 0.5 body-lengths/s in the 3 frames
  before contact.
- **Precision**, up to −20: how far from the hold's centre the foot lands.

The score is the mean over placements, graded A (90+) to E (under 60), with
per-foot scores and the three worst placements.

**Sources:**
[injury patterns in bouldering](https://www.frontiersin.org/journals/sports-and-active-living/articles/10.3389/fspor.2025.1609133/full),
[how to fall and land](https://www.climbing.com/skills/boulder-safely/),
[what "controlled" means on a top](https://www.8a.nu/news/ifsc-needs-to-define-controlled-in-bouldering-top-outs),
[IFSC top/zone/attempt scoring](https://gripped.com/indoor-climbing/boulder-world-cup-scoring-explained/),
[silent feet](https://climbskill.rocks/footwork/silent-feet/).

### Dyno lab

[`src/dyno.py`](src/dyno.py) answers "is a dyno possible between these holds,
for this climber?"
- **Reach envelope**, in body-lengths from the pose. Static reach is the larger
  of the longest hand-to-hand span actually held on the climb and 75% of the
  pose-measured wingspan. A dyno adds flight (projectile height `v²/2g` at an
  assumed 2.2 m/s take-off for a 1.75 m climber) plus leg drive (35% of
  thigh + shin). The assumptions are in `ASSUME` at the top of the file.
- **Classification.** The dyno gain is scaled by how upward the gap points (at
  least 30% of it: a jump buys height, not much sideways reach). Within static
  reach is `static`; within static + 40% of the gain, `deadpoint`; within
  static + the full gain, `dyno`; beyond that, `out_of_reach`.
- **Candidates:** every move actually climbed, plus up to 6 skipped moves (from
  a two-hand set-up, 2 to 6 holds higher), dynos and deadpoints first.
- **VLM check:** up to 4 are drawn onto a frame with an arrow and the static and
  dyno reach rings (`dyno_*.jpg`), and Qwen says whether it would commit.
- **On the test clip:** static span 0.73 body-lengths, dyno gain +0.33. Every
  climbed move was static; the skips #9→#15, #10→#16 and #11→#17 came out
  `deadpoint`, and the VLM agreed (confidence 0.85–0.92).

### Reward and RL

[`src/reward.py`](src/reward.py) is one reward function used two ways: it
scores the recorded climb (`reward` in `safety.json`), and it is the step reward
of a small route model that a Q-learning agent learns to climb (`rl`,
[`src/rl.py`](src/rl.py)). Height `h` runs from 0 at the lowest route hold to 1
at the top hold; distances are in body-lengths (bl). Weights are in `WEIGHTS`.

| Term | Weight | When |
|---|---|---|
| progress | +10 × Δh | the higher hand's hold gets higher (or lower) |
| zone | +3 | first time on the zone hold |
| top | +10 | controlled top |
| reach | −0.5 × (gap / static span)² | each hand move, gap from the other hand's hold |
| dyno_risk | −1 × (1 − margin / dyno gain) | a move classed deadpoint or dyno |
| campus | −0.2 per second | no foot on a hold while climbing (recorded climb only) |
| footwork | −0.1 | each silent-feet readjust (recorded climb only) |
| fall | −10 | each uncontrolled fall |
| landing | +(landing score − 50) / 50 | each descent |
| time | −0.01 per second | on the wall |

**The route model.** The state is (left-hand hold, right-hand hold). An action
moves one hand to any hold within static span + dyno gain of the other hand's
hold. It starts on the judge's start holds and ends with both hands on the top
hold, or after 40 moves. Each move costs 1.5 s. Deadpoints and dynos fail with
probability 0.15 or 0.35 × (1 − margin / dyno gain): a fall, −10, and the
episode ends. Feet, balance, momentum and depth are left out.

**The agent.** Tabular Q-learning, masked to reachable moves: ε-greedy decaying
from 1.0 to 0.05, α = 0.2, γ = 0.97, 4000 episodes with a fixed seed, about a
second to train. The human's hand sequence is replayed through the same env, so
both are scored on the same reward.

**On the test clip:** the recorded climb scores +18.8. The agent tops in 4
static moves for +18.75; the human's 16 moves score +18.69 through the env, and
the greedy policy tops in 100% of 200 rollouts. With a 0.73 bl static span no
dyno ever pays for its risk, so the learned beta is just longer static reaches.

### Wall scout

Scout a wall before filming: one SAM call per photo per colour, an overlay per
photo and `scout.json` in `data/output/scout/<timestamp>/`. iPhone `.HEIC`
photos need converting to JPG first (ffmpeg decodes them). On 7 test photos the
colours separated cleanly; "black" also picks up black volumes, and "white"
finds nothing on pale grey or mint holds.

## Output

One timestamped directory per run under `data/output/` (in batch mode, one
subfolder per clip plus `comparison.json` and `sequences.txt` at the top):

```
20260914-231204/
├── climbing_climb.mp4   # the pair, side by side, with the original audio
├── climbing_route.mp4   # the right panel alone
├── holds.png            # the detected route on a clean frame; check this first
├── climb.json           # order, timings, per-hold contact, utilization
├── hold_times.csv       # one row per hold: order, limbs, timings
├── limb_usage.csv       # one row per (limb, hold): seconds and share
├── summary.txt          # the climb, human-readable
├── sequence.json        # the sequence, the moves, and the other attempts
├── holds.json           # every hold's mask polygon and bbox, normalized
├── poses.json           # the raw gateway response
├── metrics.txt          # throughput and cost, with metrics.json beside it
├── run.json             # config + provenance
│   # written by safety_report.py:
├── safety.json          # judge, landings, feet, dyno, reward, rl, coach, cost (report/CONTRACT.md)
├── report.html          # self-contained report
├── fall_1.jpg           # keyframe at each landing, fall_1_0..3.jpg around it
└── dyno_1.jpg           # annotated frame per dyno candidate
```

Converted MP4s, segmentation results and pose responses are cached in
`data/cache/`, keyed on the source file *and* the settings that shaped them, so
a changed prompt never reuses a stale route. Both directories are gitignored.

If a run comes out wrong, look at `holds.png` first: nearly everything
downstream is a consequence of it.

## Models and cost

| Step | Model on the gateway |
|---|---|
| Holds and floor (`main.py`) | `facebook/sam3.1` |
| Climber pose (`main.py`) | `usyd-community/vitpose-plus-large` |
| Crash pad, one clean frame | `facebook/sam3.1` |
| Fall verdict, 4 frames around impact | `qwen/qwen3.8-27b` |
| Dyno check, one annotated frame | `qwen/qwen3.8-27b` |
| Coaching note, text only | `qwen/qwen3.8-27b` |

About **$0.025 for `main.py`** and **$0.006 for the report**. `main.py` writes
its cost to `metrics.txt`; the report's is in `safety.json` under `cost`
(`sam_pad`, `vlm`), taken from the gateway's reported cost or estimated from
token counts, and printed at the end of `safety_report.py`.

## Gotchas

- **SAM pad prompts fall back in order:** `crash pad` → `gym mat` → `floor mat`
  → `gray floor`. On wall-to-wall matting the whole floor is the pad and only
  `gray floor` finds it. With `--no-vlm`, or when every prompt misses, the floor
  segmented by `main.py` is used as the landing zone.
- **Qwen needs thinking off**: `extra_body={"chat_template_kwargs":
  {"enable_thinking": False}}`. With it on, qwen3.8 spends its whole token budget
  reasoning on image prompts and returns empty content after ~110 s.
- **The VLM is commentary, not an independent judge.** Its prompts include our
  measured metrics, so its verdicts are conditioned on them. The
  `agrees_with_metrics` flag is there to surface the cases where it still
  disagrees.
- **Wrong colour, wrong everything.** A colour that isn't the route you climbed
  still finds *some* holds, and every analysis runs on them without complaint.
- **HDR iPhone clips on Windows:** conda-forge ffmpeg has no `libplacebo`, so HDR
  is not tone-mapped and the render looks washed out. Detection was still fine.

## Limits

- One still camera: no panning, and the whole route must stay in frame.
- 2D pose with no depth, so knee angles and distances are foreshortened.
- Body-length units come from the median torso, thigh and shin in pixels, not
  from the climber's real height.
- The thresholds are tuned on a handful of clips, not calibrated against
  labelled falls or real judges.
- The route model is hands only; its beta is a sketch, not advice.

## Where this goes next

- **Live, in the browser.** MediaPipe pose on the phone, an Agent SDK coach on
  the laptop, browser speech for the calls, tunnelled with `cloudflared`.
- **Gym-level fall-zone heatmaps**, landings pooled across many clips per wall.
- **Auto-judging comps**: attempts, zone and top per competitor, from a fixed
  camera per problem.

## Changelog

**2026-09-25**
- Dashboard: the RUN button names the clip and colour, the newest clip is
  auto-selected, re-running the same clip and colour warns first, and runs can
  be deleted.
- Logic cross-check of the judge, landings, dyno lab, reward and RL, with a
  repeatable test: `python tests/test_logic.py <good_run> [fixture_runs...]`
  (or `CLIMB_GOOD_RUN` / `CLIMB_FIXTURES`). Fixed: RL farming the zone bonus,
  uncontrolled tops scored as tops, campus penalty with no hand on the route,
  zone/top holds picked by id instead of height, and each run now reads its
  own floor (it used to take the newest cached floor, i.e. another clip's).
- "Read with care" warnings in the report: finish hold at the frame edge,
  climber lost after the top, other people in frame, implausible arm length.
- Reach is capped at a 1.06 wingspan-to-height ratio when a low camera angle
  inflates the arm reading.
- VLM fall and dyno verdicts carry `agrees_with_metrics`, so disagreements are
  visible.
- Second test clip, IMG_0510 (black route): TOP, silent feet 64 (D), with
  warnings (top out of frame, climber lost after the top, a passer-by).
- Cost accounting: gateway cost per report in `safety.json` and the report.
- A **How to run** page at `/guide`.

**2026-09-24**
- `scout.py`: per-colour hold inventory on wall photos.
- `safety_report.py`: IFSC-style judge (attempts, start, zone, top, control).
- Landing safety: fall / jump-off / downclimb, knees, feet on pad, hand posting,
  feet sync, drop, with a VLM verdict per landing.
- Silent-feet score per placement and per foot, graded A–E.
- Crash pad segmented with SAM 3.1, falling back to the floor.
- Dyno lab (`src/dyno.py`): reach envelope and static / deadpoint / dyno /
  out-of-reach per gap.
- `report.html`: a self-contained report, plus a coaching note.
- `app.py`: a local dashboard to upload, run and browse runs.
- Reward function (`src/reward.py`) and a tabular Q-learning agent
  (`src/rl.py`) on a hands-only route model, compared with the human's beta.

## License

[Apache-2.0](../LICENSE).
