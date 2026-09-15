# NanoTrack V3 with ONNX Runtime

A small multi-object, arbitrary-ROI, axis-aligned bounding-box tracker. Runtime dependencies are
NumPy, OpenCV and ONNX Runtime. No `cv2.TrackerNano`, detection, segmentation or pose code.

## Run

From this directory in PowerShell (Python 3.10+):

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python main.py --source 0
python main.py --source "path/to/video.mp4"
```

If activation is restricted, use `.venv\Scripts\python.exe` directly. On macOS/Linux,
activate with `source .venv/bin/activate`.

Drag a rectangle over the live feed and release the left mouse button to start
tracking. Press `a` and drag again to add another object. The feed and existing
trackers keep updating while you select, at startup and after `a` or `r`.
The template comes from the displayed frame at release. Escape/C cancels selection;
existing targets keep tracking. Click a box or press Tab to select a target. The
selected box is thicker and its ID has a `*`. IDs are stable and never reused.

| Key | Action |
| --- | --- |
| `q` | Quit |
| `a` | Add another object with a live drag selection |
| Tab / click box | Select a target (Tab cycles through overlapping boxes) |
| `r` | Draw a replacement ROI for the selected ID; adds a target if none exists |
| `t` | Reseed only the selected target using its current predicted box |
| `u` | Toggle continuous template updates for all targets |
| `[` / `]` | Decrease/increase template low-pass Fc by 0.5 Hz (0 bypasses) |
| `d` | Remove the selected target |
| `l` | Show/hide target center trails |
| `v` | Start/stop recording (red REC indicator while active) |
| Escape / `c` | Cancel the current drag selection |

`--debug` shows template/search crops and confidence/bbox for the selected target.
Target switching, `t`, and `d` are available outside drag selection. There is
no frame-rate cap. Automatic template updating is off by default. `waitKey(1)` pumps the GUI.
Tracking milliseconds sum inference/tracking across all objects and exclude
decode/display; smoothed FPS includes them. Each box shows its own confidence.
One camera capture and two ONNX sessions are shared by all targets. Processing
is sequential and its cost grows approximately with the number of targets.

Repeat `--bbox` for multiple initial targets in repeatable/headless runs:

```powershell
python main.py --source reference/girl_dance.mp4 --bbox 200 165 185 300 --bbox 440 158 197 310
python -m unittest test_multiobject.py
```

## Continuous template experiment

Press `u` to toggle template updating for **all targets**, including subsequently
added targets. The on-screen indicator shows its state. Start enabled with:

```powershell
python main.py --source 0 --auto-template --debug
```

After each prediction, this mode refreshes the target's template with a fresh
127x127 context crop from the clean current frame at the predicted center/size.
That template is used to search the next frame. There is no confidence gating
or update interval; only invalid network outputs skip the update. The app applies
the pixel filter described below unless Fc is set to 0.
Turning it off keeps the most recent template, rather than restoring the original.
Use `r` to select the target again if it drifts.

Template refresh preserves the bounding box, confidence, search debug image and
trail. It does not repeatedly call `init()`, which would introduce its half-pixel
center offset. Sessions remain shared and unchanged. The extra template backbone
inference and crop cost are included in tracking milliseconds/FPS. The debug
template window shows the freshly updated template. Recordings log the mode in
each frame's `auto_template` field. This intentionally naive mode may reinforce
drift; it is provided to experiment with that behavior.

API: construct with `auto_update_template=True`, or change that attribute at
runtime. `refresh_template(frame)` performs one template-only refresh manually.

### Pixel-wise template low-pass filter

The app defaults to **Fc = 2 Hz**. Use `--template-fc 0` for the previous unfiltered
replacement experiment, or e.g. `--auto-template --template-fc 1 --debug` for a
slower update. `[` and `]` adjust Fc live for all targets. Lower positive Fc retains
more history; higher Fc responds faster. **0 means bypass**, not a frozen template.
Filtering only runs when the template is refreshed; `u` still controls automatic
updates. Turning automatic updates off retains the latest filtered template.

For each BGR pixel/channel on the resized 127x127 template crop:

```text
alpha = 1 - exp(-2*pi*Fc*dt)
template = previous_template + alpha * (new_crop - previous_template)
```

This is an exponential discretization of a first-order temporal low-pass with
time constant `1/(2*pi*Fc)`; Fc is the continuous-time cutoff parameter (the sampled
filter's exact -3 dB point differs near the sampling limit). Filtering uses float32
pixels **before backbone inference**, with no spatial blur or feature blending.
Initialization, ROI replacement and manual `t` reseeding reset the pixel history.
The debug template window shows the actual filtered pixels, rounded for display.
Motion within the aligned crop can produce ghosting; no pixel registration is added.

For video files `dt` is one source-FPS interval, independent of processing speed.
For cameras, or files without valid FPS, it is elapsed capture-receipt time (the
first camera sample uses zero). Source-FPS timing assumes constant-frame-rate video.
Recordings include `template_fc_hz` and `template_dt_s` on every frame.

The reusable API defaults to `template_fc_hz=0.0` for backward compatibility.
Set `tracker.template_fc_hz = 2.0` and call `update(frame, dt=1/30)` for explicit
sample timing; omitted `dt` uses monotonic time between calls. Each target keeps
its own pixel history while sharing the ONNX sessions.

## Trails and recording

Trails retain the last 60 center positions per object. Use `--trail-length 120`
for longer trails or `--trail-length 0` to disable storage. Replacing an ROI clears
that target's trail; removing a target removes its trail. Reseeding keeps the trail.

Press `v` to start/stop recording, or launch with `--record`. Each recording gets
a new timestamped folder under `recordings/` (change with `--record-dir`). It contains:

- `video.avi`: annotated MJPEG video with boxes, IDs, confidence and visible trails.
- `tracking.jsonl`: one JSON record per saved frame, including target IDs, xywh
  boxes, confidence, success, tracking time, UTC capture-receipt timestamp, elapsed
  time since app startup, and video-source position in milliseconds when available.
- `metadata.json`: source, video dimensions and playback FPS.

Recording closes on stop, quit, video end or an application exception. A toggle
starts saving with the next frame. Tracking data describes the predictions drawn
on each saved frame, before any selection/reseed action made on that frame.
Frame indices are zero-based. Frames without targets have an empty targets list.

AVI uses constant playback FPS (source FPS, or 30 if unavailable); override with
`--record-fps 30`. It writes each processed frame once without slowing the feed.
If processing/camera timing varies, playback duration can differ from wall time;
use the JSON timestamps for timing analysis. No audio is recorded.

```powershell
python main.py --source 0 --record --trail-length 90
python -m unittest test_multiobject.py test_recording.py
```

## Models and the V3 export discrepancy

Prepared V3 files are included in `models/nanotrack_backbone_sim.onnx` and
`models/nanotrack_head_sim.onnx`. **Upstream does not supply V3 `_sim.onnx` files**
at the inspected commit; files with those names under V1/V2 are different models.
The actual sources are [V3 backbone](https://raw.githubusercontent.com/HonglinChu/SiamTrackers/248663fde6bf7c40190cf10ee396d5662919ecd3/NanoTrack/models/nanotrackv3/nanotrack_backbone.onnx)
and [V3 head](https://raw.githubusercontent.com/HonglinChu/SiamTrackers/248663fde6bf7c40190cf10ee396d5662919ecd3/NanoTrack/models/nanotrackv3/nanotrack_head.onnx).

The upstream backbone declares a fixed 255x255 input and 16x16 output, despite
the tracker requiring both 127x127 and 255x255 crops. `prepare_models.py` makes
the backbone's spatial input/output metadata dynamic, then simplifies both
graphs. Weights are unchanged. The fully convolutional backbone successfully
produces 8x8 and 16x16 features in ORT. The head remains fixed-shape.
Source and prepared SHA-256 hashes are recorded in `models/provenance.json`.

To regenerate/download the models (these two extra packages are preparation-only):

```powershell
python -m pip install onnx onnxsim
python prepare_models.py
```

The script compares prepared inference to the unsimplified graph at both input
sizes and checks the ONNX graphs. Runtime needs neither `onnx` nor `onnxsim`.
Alternative model paths can be supplied with `--backbone` and `--head`.
Fixed-size backbones and incompatible V2 interfaces produce explicit errors.

## Verified algorithm

Source revision: `248663fde6bf7c40190cf10ee396d5662919ecd3` in
[HonglinChu/SiamTrackers](https://github.com/HonglinChu/SiamTrackers).
Inspected files are retained under `reference/` for reproducible comparison.

| Item | Verified V3 behavior |
| --- | --- |
| Template / search | 127x127 / 255x255 |
| Context | 0.5 times width + height, added to each dimension; geometric mean gives crop side |
| Pixel input | OpenCV BGR, uint8 crop then float32 NCHW, raw 0–255; no channel swap or normalization |
| Padding / resize | Initialization-frame BGR mean, truncated to uint8; bilinear OpenCV resize; original floor/round conventions |
| Backbone | `input` `[1,3,H,W]` → `output` `[1,96,Hf,Wf]`; 127→8 and 255→16 verified by inference |
| Head template | `input1` `[1,96,8,8]` |
| Head search | `input2` `[1,96,16,16]` |
| Classification | `output1` `[1,2,15,15]`; softmax across two channels, foreground channel 1 |
| Localization | `output2` `[1,4,15,15]`; left/top/right/bottom distances, **already exponentiated inside head** |
| Grid | Anchor-free 15x15 points; stride 16, coordinates −112…112; row-major x/y meshgrid |
| Decode | point minus left/top and plus right/bottom; corners → center/width/height |
| Scale / ratio | `change(r)=max(r,1/r)`; padded-size and aspect-ratio changes multiplied |
| Penalty | `exp(-(scale_change*ratio_change-1)*0.138)` |
| Hann window | Outer product of `np.hanning(15)`; ranking = `0.545*penalty*score + 0.455*window` |
| Interpolation | Selected displacement / search scale added directly to center; size LR = `penalty*score*0.348` |
| Clipping | Center clipped to `[0,W]`, `[0,H]`; width/height to `[10,W]`, `[10,H]` |

Clipping deliberately matches upstream: returned corners can extend outside the
frame. Initialization uses `x+(w-1)/2`; updates return `cx-w/2`, including the
original half-pixel convention. Search scale uses the unrounded context size;
only pixel crop size is rounded. No extra neck is applied: although the YAML
declares an adjustment layer, `ModelBuilder.template/track` bypass it. The V3
head handles its own internal template feature cropping/correlation.

Relevant primary sources: [V3 config](https://github.com/HonglinChu/SiamTrackers/blob/248663fde6bf7c40190cf10ee396d5662919ecd3/NanoTrack/models/config/configv3.yaml),
[tracker](https://github.com/HonglinChu/SiamTrackers/blob/248663fde6bf7c40190cf10ee396d5662919ecd3/NanoTrack/nanotrack/tracker/nano_tracker.py),
[crop implementation](https://github.com/HonglinChu/SiamTrackers/blob/248663fde6bf7c40190cf10ee396d5662919ecd3/NanoTrack/nanotrack/tracker/base_tracker.py),
[V3 head](https://github.com/HonglinChu/SiamTrackers/blob/248663fde6bf7c40190cf10ee396d5662919ecd3/NanoTrack/nanotrack/models/head/ban_v3.py),
[model builder](https://github.com/HonglinChu/SiamTrackers/blob/248663fde6bf7c40190cf10ee396d5662919ecd3/NanoTrack/nanotrack/models/model_builder.py),
[export script](https://github.com/HonglinChu/SiamTrackers/blob/248663fde6bf7c40190cf10ee396d5662919ecd3/NanoTrack/pytorch2onnx.py).

## Reusable API and providers

```python
from nanotrack import NanoTrackORT

tracker = NanoTrackORT(
    backbone_path="models/nanotrack_backbone_sim.onnx",
    head_path="models/nanotrack_head_sim.onnx",
    providers=["CPUExecutionProvider"],
)
tracker.init(frame, (x, y, width, height))
success, bbox, confidence = tracker.update(frame)
tracker.reseed(frame, bbox)  # Same sessions; recomputes template, mean, geometry.

other = tracker.new_target()  # Shares sessions; independent, uninitialized state.
other.init(frame, another_bbox)
# In each capture iteration, call update(frame) on both trackers.
```

Sessions are constructed once. Constructor probes both crop sizes and head outputs
to validate interfaces and warm up inference. Ordinary updates run one search
backbone and one head call in the default mode; init/reseed run one template
backbone call. Continuous mode adds one template backbone call after each prediction.
Each tracker instance holds one target's state. The app uses `new_target()` to
share sessions while keeping templates, geometry and debug images independent.
All updates run in one calling thread. These independent trackers can drift onto
each other during overlap; IDs label tracker instances, not guaranteed identities.

`confidence` is the raw foreground softmax probability at the candidate selected
*after* penalties/window ranking. It is not the windowed score or necessarily the
maximum raw score. Upstream has no lost-target threshold. `success` defaults to
true for a finite prediction; optional `confidence_threshold=0.5` flags low
scores without freezing or changing the original tracking algorithm.

For another provider install its appropriate ONNX Runtime distribution, then:

```powershell
python main.py --providers CUDAExecutionProvider CPUExecutionProvider
```

Use `onnxruntime.get_available_providers()` to see available providers. Unavailable
requested providers are rejected. The Python API also accepts ORT provider-option
tuples. CPU is the default, with one intra-op thread to avoid thread overhead on
these small graphs; try `--threads 0` or another count on your machine.

## Validation and practical limits

```powershell
python validate.py
python main.py --source reference/girl_dance.mp4
python main.py --source reference/girl_dance.mp4 --bbox 440 158 197 310 --headless --max-frames 300
```

Validation uses the upstream sample video stored locally. It executes the original
tracker source with a NumPy tensor adapter and the same ORT model outputs (not a
PyTorch-versus-ONNX weights comparison). Results on this Windows host:

- 32 crop cases, including padding and fractional centers: pixel-exact match.
- 300 real-video updates: maximum bbox difference from upstream **0 pixels**;
  classification score parity also passes.
- Three controlled moving/scaling textured targets at different sizes and
  locations: mean IoU **0.901–0.903**, minimum **0.871–0.877**.
- Target-present confidence near 1; mean confidence over ten absent frames
  **0.205–0.456** in controlled tests.
- Reseed test forbids session construction and checks one 127 template inference
  followed by one 255 search inference.
- CPU tracking approximately **158–162 FPS** (6.2–6.3 ms), headless video loop
  **132 FPS**. A 30-frame GUI smoke run measured **54 FPS** including display.
  These short runs are illustrative, not hardware-independent guarantees.
- Tested with Python 3.13, NumPy 2.5.3, OpenCV 5.0.0 and ORT 1.29.0.

`validation_output/` contains JSON metrics, an annotated AVI and a contact sheet.
Video inspection shows the box follows the selected dancer through substantial
motion; during close overlap it can include both dancers despite high confidence.
No camera or physical 6DoF experiment was performed. The GUI display was exercised;
interactive ROI selection and physical keyboard input still need your hands-on check.

NanoTrack is a local single-template tracker. Large rotations, perspective changes,
occlusion or fast motion outside the search area can cause drift. Its saturated
classification confidence is not a calibrated correctness/lost-target probability.
Manual `t` can help appearance adaptation but can also reinforce a drifted box;
`r` selects the intended target again. Continuous updating is opt-in via `u` or
`--auto-template`.
