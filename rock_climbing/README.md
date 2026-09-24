# Rock Climbing + Computer Vision

Segments a boulder problem off the wall, follows the climber up it, and reads the
route back: which holds were used, in what order, and how long the send took.
Two models run on the [VLM Run Gateway](https://www.vlm.run/gateway):
[`sam3.1`](https://vlm.run/gateway/models/facebook-sam3.1) for the holds and
the floor,
[`vitpose-plus-large`](https://vlm.run/gateway/models/usyd-community-vitpose-plus-large)
for the climber.

<p align="center">
  <img src="readme_images/rock_climbing_demo_thumbnail.jpg" width="600" alt="A completed boulder problem: the left panel holds a card comparing this attempt's sequence of holds against three others, the right panel the segmented route with the holds used lit in order">
</p>

## tl;dr: how to run this demo

1. Clone the [vision-demos](https://github.com/jeremyipark/vision-demos) repo.
2. Get an API key at [VLM Run](https://app.vlm.run/sign-in) and add it to your `.env`.
3. Record a video of yourself climbing, with your phone still and the whole route in frame.
4. Send the video to your computer and drop it in `data/input/current/`.
5. Update `HOLD_COLOR` in [`config.py`](config.py) to match your route.
6. Create the conda environment and run `python main.py`.
7. View the output in the latest timestamped folder under `data/output/`. The video ends in `_climb.mp4`.

The details for each step are below.

## Resources

The rock climbing + computer vision demo
([LinkedIn video](https://www.linkedin.com/posts/jeremyipark_computervision-ai-ml-activity-7505763952740478976-cgKm)):

* Uses [ViTPose+ Large](https://vlm.run/gateway/models/usyd-community-vitpose-plus-large) for pose estimation
* Uses [SAM 3.1](https://vlm.run/gateway/models/facebook-sam3.1) to segment the bouldering holds
* Shows each hold light up as a hand or foot uses it
* Shows the sequence of holds used and the total time once the climb is done

Both models run on the VLM Run [Gateway](https://vlm.run/gateway), whose Model
Catalog gives you access to 22 vision models.

## Setup instructions

1. Clone the [vision-demos](https://github.com/jeremyipark/vision-demos) repo.
2. Sign up at [VLM Run](https://app.vlm.run/sign-in) to get an API key.
3. Copy your API key from the [main Overview dashboard](https://app.vlm.run/dashboard).
   The key is only shown once, when it's created, and then it's hidden, so copy
   it right away. You can also generate a new API key.
4. Add your API key to a `.env` file at the vision-demos root. From the
   `rock_climbing` folder, run:

   ```bash
   cp ../.env.example ../.env
   ```

   Then paste the key after `VLMRUN_API_KEY=`.

## Data collection

**tl;dr: keep your phone still using a tripod or a water bottle, and have the
entire route in frame.**

This rock climbing demo was based on a video taken with a tripod, so it assumes a
still camera. For best results:

* Have the entire route visible in frame, from the first hold to the last hold.
* Pick an easy route. For these demo videos, I typically go with a VB or V0.
* Use a tripod, or simply set your phone on a water bottle. Someone can also
  hold the camera, but they should try to stay as still as possible.
* Try to avoid recording other people, and watch out for anyone who might walk
  through the frame.

<p align="center">
  <img src="readme_images/rock_climbing_video_setup.jpg" width="320" alt="A climber at the start of a green route, filmed in portrait from a still phone; every green hold from the floor to the finish is in frame">
  <br>
  <em>Example framing of the rock climbing video. Note that all holds are in frame.</em>
</p>

In short, to create the minimal reproducible example of my demo:

1. Go to an easy VB/V0 route.
2. Put your phone on a water bottle.
3. Have the entire route in frame.
4. Record yourself completing the route.
5. Trim the video so it starts right before you begin the route and ends right
   after you finish it (this saves on inference time 🙂).
6. Send the video to your computer.

## Code instructions

**tl;dr: clone the vision-demos repo, point Claude or another coding agent at the
repo, and describe how you want to update the project.**

1. Add your trimmed video to the input folder:
   `vision-demos/rock_climbing/data/input/current/`
2. Create the conda environment (from the `rock_climbing` folder):

   ```bash
   conda env create -f environment.yml
   conda activate rock_climbing
   ```

3. Run the program:

   ```bash
   python main.py
   ```

4. View the rendered output in the latest timestamped folder under
   `data/output/`. Each clip gets its own subfolder, and the video will have the
   suffix `_climb.mp4`.

**NOTE:** the hold color is currently a variable in [`config.py`](config.py).
Update `HOLD_COLOR` to match your route so that SAM 3.1 knows which holds to
segment (or ask your coding agent to do this).

```python
HOLD_COLOR = "green"                     # config.py
HOLD_PROMPT = "{color} climbing hold"
ROUTE_GRADE = "VB"                       # metadata; recorded, not drawn
```

Point it at another colour and the demo follows a different problem up the same
wall, with no retraining and no new model. That is what SAM 3.1 buys over a
detector fine-tuned on one gym's holds.

Every other knob lives in [`config.py`](config.py), and each run snapshots the
ones it used into `run.json`.

### Comparing several attempts

Every clip in `data/input/current/` is read as another attempt at the same route.
Drop one in and it behaves like a single run; drop four in and each render ends
on a card comparing its holds against the other three. Set `BATCH_MODE = False`
to run one clip, named by `INPUT_VIDEO`.

This is another reason to keep the camera still: holds are detected once per clip
and reused for every frame of it, and in batch mode the clips are compared to each
other hold for hold, so the camera should not move within a take or between takes.

For how the route is read and the attempts are aligned, see
[route-reading-explained.md](route-reading-explained.md).

## Output

In batch mode, one timestamped directory holding a subdirectory per clip:

```
20260915-153000/
├── comparison.json      # every attempt, aligned onto one numbering
├── sequences.txt        # the same thing, human-readable
├── IMG_8842/            # everything below, per clip
└── …
```

Otherwise one timestamped directory per run under `data/output/`:

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
└── run.json             # config + provenance
```

Converted MP4s, segmentation results and pose responses are cached in
`data/cache/`, keyed on the source file *and* the settings that shaped them, so a
changed prompt never reuses a stale route. Both directories are gitignored.

If a run comes out wrong, look at `holds.png` first: nearly everything
downstream is a consequence of it. The knob for each symptom is in
[route-reading-explained.md](route-reading-explained.md).

## Climb Vision Judge: judging, landing safety, silent feet, dyno lab

**tl;dr: `python app.py`, open http://127.0.0.1:8000, press Run.**

```bash
python app.py        # then open http://127.0.0.1:8000
```

The page lets you upload a clip, pick the hold colour and grade, and choose
whether to spend VLM credits. **Run** runs `main.py` and then
`safety_report.py`, with live progress. Past runs sit in a gallery below, each
with **Open report** and **Re-run analysis**.

**From the command line** instead: run `main.py`, then `safety_report.py` on its
output, and open `report.html`.

```bash
python main.py                              # the route read, as above
python safety_report.py                     # newest run under data/output/
python safety_report.py <run_dir>           # a specific run (or clip subfolder in batch mode)
python safety_report.py <run_dir> --no-vlm  # no gateway calls: no pad, verdicts or coach note
```

It works on a finished run folder and writes beside it:

```
safety.json      # everything below; schema in report/CONTRACT.md
report.html      # self-contained report, the JSON inlined into report/template.html
fall_1.jpg       # keyframe at each landing, fall_1_0..3.jpg around it
dyno_1.jpg       # annotated frame per dyno candidate
```

Open `report.html` in a browser. It plays the `_climb.mp4` beside it, so if the
video won't load from `file://`, serve the folder:

```bash
cd data/output/<run>
python -m http.server 8080                        # http://localhost:8080/report.html
cloudflared tunnel --url http://localhost:8080    # a public link, to share it
```

### What it measures

These are heuristics taken from coaching and injury writing, tuned on **one test
clip**. Treat the thresholds as starting points and tune them on your own
footage; they all live at the top of [`src/safety.py`](src/safety.py).

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
  with the hips still (median speed under 0.6 torso-lengths/s).
- Result: `TOP`, `ZONE` or `NO SCORE`.

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

A downclimb gets +5.

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

**Dyno lab** ([`src/dyno.py`](src/dyno.py)). "Is a dyno possible between these
holds, for this climber?"
- **Reach envelope**, in body-lengths from the pose. Static reach is the larger
  of the longest hand-to-hand span actually held on the climb and 75% of the
  pose-measured wingspan (arm length from the 90th percentile of the pose, since
  2D pose foreshortens a bent arm). A dyno adds flight (projectile height `v²/2g` at an
  assumed 2.2 m/s take-off for a 1.75 m climber) plus leg drive (35% of
  thigh + shin).
- **Classification.** The dyno gain is scaled by how upward the gap points (at
  least 30% of it: a jump buys height, not much sideways reach). A gap within
  static reach is `static`; within static + 40% of the gain, `deadpoint`; within
  static + the full gain, `dyno`; beyond that, `out_of_reach`. The margin left
  over is kept, and each target is sized small / medium / large as a catch.
- **Candidates**: every move actually climbed (a hand reaching a new hold while
  the other anchors), plus up to 6 skipped moves (from a two-hand set-up, 2 to 6
  holds higher in one), dynos and deadpoints first.
- **VLM check.** Up to 4 of them (the skipped dynos/deadpoints and the biggest
  climbed move) are drawn onto a frame with the from/to holds, an arrow and the
  static and dyno reach rings (`dyno_*.jpg`), and `qwen/qwen3.8-27b` says
  whether it would commit.
- **On the test clip:** static span 0.73 body-lengths, dyno gain +0.33. Every
  climbed move was static; the skips #9→#15, #10→#16 and #11→#17 came out
  `deadpoint`, and the VLM agreed (confidence 0.85–0.92).

The assumptions are explicit, in `ASSUME` at the top of the file.

**Sources:**
[injury patterns in bouldering](https://www.frontiersin.org/journals/sports-and-active-living/articles/10.3389/fspor.2025.1609133/full),
[how to fall and land](https://www.climbing.com/skills/boulder-safely/),
[what "controlled" means on a top](https://www.8a.nu/news/ifsc-needs-to-define-controlled-in-bouldering-top-outs),
[IFSC top/zone/attempt scoring](https://gripped.com/indoor-climbing/boulder-world-cup-scoring-explained/),
[silent feet](https://climbskill.rocks/footwork/silent-feet/).

### Models and cost

| Step | Model on the gateway |
|---|---|
| Crash pad, one clean frame | `facebook/sam3.1` |
| Fall verdict, 4 frames around impact | `qwen/qwen3.8-27b` |
| Dyno check, one annotated frame | `qwen/qwen3.8-27b` |
| Coaching note, text only | `qwen/qwen3.8-27b` |

About **$0.006 per report**, on top of about $0.025 for `main.py`. The test clip
cost $0.004 for the pad and $0.0015 for the Qwen calls.

### Gotchas

- **SAM pad prompts fall back in order:** `crash pad` → `gym mat` → `floor mat`
  → `gray floor`. On a gym with wall-to-wall matting the whole floor is the pad,
  and only `gray floor` finds it. With `--no-vlm`, or when every prompt misses,
  the floor segmented by `main.py` is used as the landing zone.
- **Qwen needs thinking off**: `extra_body={"chat_template_kwargs":
  {"enable_thinking": False}}`. With it on, qwen3.8 spends its whole token budget
  reasoning on image prompts and returns empty content after ~110 s.
- **The VLM is commentary, not an independent judge.** Its prompts include our
  measured metrics, so its verdicts are conditioned on them.
- **A top hold at the frame edge** takes the hands out of frame on the match. The
  judge then takes control from the route read's own top-out, and says so in
  `judge.top.note`.
- **HDR iPhone clips on Windows:** conda-forge ffmpeg has no `libplacebo`, so HDR
  is not tone-mapped and the render looks washed out. Detection was still fine.

### Limits

- One still camera: no panning, and the whole route must stay in frame.
- 2D pose with no depth, so knee angles and distances are foreshortened.
- Body-length units come from the median torso, thigh and shin in pixels, not
  from the climber's real height.
- Validated on one clip. The thresholds are not calibrated against labelled
  falls or judges.

### Wall scout (photos, no climber)

Scout a wall before filming: which colours SAM 3.1 separates cleanly, and how
many holds each route has.

```bash
python scout.py                                   # data/input/photos/jpg
python scout.py <photos_dir> --colors blue,green,pink
```

One SAM call per photo per colour (~$0.001 each). Writes an overlay per photo
and `scout.json` to `data/output/scout/<timestamp>/`. iPhone `.HEIC` photos need
converting to JPG first (ffmpeg decodes them). On the 7 test photos, colours
separated cleanly. "Black" also picks up black volumes, and "white" finds
nothing on pale grey or mint holds.

### Changelog

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

## Where this goes next

- **Live, in the browser.** MediaPipe pose on the phone, an Agent SDK coach on
  the laptop, browser speech for the calls, tunnelled with `cloudflared`.
- **Gym-level fall-zone heatmaps**, landings pooled across many clips per wall.
- **Auto-judging comps**: attempts, zone and top per competitor, from a fixed
  camera per problem.

## License

[Apache-2.0](../LICENSE).
