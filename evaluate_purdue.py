"""Live Purdue tracking evaluation with optional, explicitly logged GT recovery."""
import argparse
from collections import defaultdict, deque
from datetime import datetime
import json
import math
from pathlib import Path
import re
from time import perf_counter

import cv2
import numpy as np

from nanotrack import NanoTrackORT
from video_reader import VideoReader

ROOT = Path(__file__).resolve().parent
COLORS = [(0, 255, 0), (255, 100, 255), (0, 170, 255), (255, 160, 80)]


def load_annotations(path):
    """Refined MOT: one-based frame, ID, x,y,w,h, annotation confidence, ...

    Pixel coordinates are used directly, matching the dataset's supplied renderer.
    Keep positive annotation confidence, including interpolated rows below 1.
    """
    frames = defaultdict(dict)
    for number, line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), 1):
        if not line.strip():
            continue
        values = [float(v) for v in line.split(",")]
        if len(values) < 7 or not all(math.isfinite(v) for v in values):
            raise ValueError(f"{path}:{number}: invalid MOT row")
        frame, identity = map(int, values[:2])
        if frame < 1 or identity < 1 or values[:2] != [frame, identity]:
            raise ValueError(f"{path}:{number}: expected positive integer frame/ID")
        if values[6] <= 0:
            continue
        if min(values[4:6]) <= 0 or identity in frames[frame]:
            raise ValueError(f"{path}:{number}: invalid box or duplicate frame/ID")
        frames[frame][identity] = tuple(values[2:6])
    if not frames:
        raise ValueError(f"No positive annotations in {path}")
    return dict(frames)


def discover(videos, annotations):
    def files(root, pattern):
        found = {}
        for path in root.rglob("*"):
            match = re.fullmatch(pattern, path.name, flags=re.IGNORECASE)
            if match and "__MACOSX" not in path.parts:
                identity = int(match[1])
                if identity in found:
                    raise ValueError(f"Duplicate clip {identity}: {found[identity]} and {path}")
                found[identity] = path
        return found
    movies = files(videos, r"Clip_(\d+)\.(?:mov|mp4|avi|mkv)")
    labels = files(annotations, r"Clip_(\d+)_refined\.txt")
    if not movies:
        raise ValueError(f"No Clip_N videos in {videos}")
    if set(movies) - set(labels):
        raise ValueError(f"Missing annotations for clips: {sorted(set(movies)-set(labels))}")
    return {i: (movies[i], labels[i]) for i in sorted(movies)}


def clip_selection(spec, available):
    if spec.lower() == "all":
        return list(available)
    selected = set()
    for part in spec.split(","):
        ends = part.strip().split("-")
        if len(ends) == 1:
            selected.add(int(ends[0]))
        elif len(ends) == 2 and int(ends[0]) <= int(ends[1]):
            selected.update(range(int(ends[0]), int(ends[1])+1))
        else:
            raise ValueError(f"Invalid clip selection: {part}")
    if selected - set(available):
        raise ValueError(f"Missing clips: {sorted(selected-set(available))}")
    return sorted(selected)


def box_metrics(predicted, truth):
    a, b = np.asarray(predicted), np.asarray(truth)
    intersection = np.prod(np.maximum(0, np.minimum(a[:2]+a[2:], b[:2]+b[2:]) - np.maximum(a[:2], b[:2])))
    iou = intersection / (np.prod(a[2:])+np.prod(b[2:])-intersection)
    error = np.linalg.norm(a[:2]+a[2:]/2-b[:2]-b[2:]/2)
    return float(iou), float(error)


class SequenceTracker:
    """Keep failed predictions for scoring/display even when resetting state."""
    def __init__(self, engine, *, gt_reset=False, reset_iou=0.1, reset_error_px=None, reset_patience=3,
                 annotated_only=False):
        if not math.isfinite(reset_iou) or not 0 <= reset_iou <= 1:
            raise ValueError('reset_iou must be in [0, 1]')
        if reset_error_px is not None and (not math.isfinite(reset_error_px) or reset_error_px <= 0):
            raise ValueError('reset_error_px must be positive')
        if reset_patience < 1:
            raise ValueError('reset_patience must be >= 1')
        self.engine = engine
        self.annotated_only = annotated_only
        self.pending_dt = defaultdict(float)
        self.gt_reset, self.reset_iou = gt_reset, reset_iou
        self.reset_error_px, self.reset_patience = reset_error_px, reset_patience
        self.targets = {}
        self.births = {}
        self.bad_frames = defaultdict(int)
        self.last_seed = {}
        self.reset_events = []
        self.frame_resets = {}

    def step(self, frame, frame_no, ground_truth, dt):
        results = {}
        self.frame_resets = {}
        for identity, box in ground_truth.items():
            if identity not in self.targets:
                # Refined boxes may straddle an image edge. Keep their geometry
                # and let NanoTrack mean-pad; wait if entirely outside the image.
                if frame is not None and (box[0]+box[2] <= 0 or box[1]+box[3] <= 0 or
                                          box[0] >= frame.shape[1] or box[1] >= frame.shape[0]):
                    continue
                target = self.engine.new_target()
                target.init(frame, box)
                self.targets[identity] = target
                self.births[identity] = frame_no
                self.last_seed[identity] = frame_no
                results[identity] = (True, target.bbox, None)
        for identity, target in self.targets.items():
            if identity not in results:
                if self.annotated_only and identity not in ground_truth:
                    self.bad_frames[identity] = 0
                    self.pending_dt[identity] += dt
                    continue
                results[identity] = target.update(frame, dt=dt+self.pending_dt.pop(identity, 0.0))
                if not self.gt_reset:
                    continue
                truth = ground_truth.get(identity)
                if truth is None or (frame is not None and
                    (truth[0]+truth[2] <= 0 or truth[1]+truth[3] <= 0 or
                     truth[0] >= frame.shape[1] or truth[1] >= frame.shape[0])):
                    self.bad_frames[identity] = 0
                    continue
                success, predicted, confidence = results[identity]
                iou, error = box_metrics(predicted, truth)
                bad = not success or iou < self.reset_iou or (
                    self.reset_error_px is not None and error > self.reset_error_px)
                self.bad_frames[identity] = self.bad_frames[identity]+1 if bad else 0
                if self.bad_frames[identity] >= self.reset_patience:
                    event = dict(frame=frame_no, id=identity, predicted_bbox=predicted,
                                 gt_bbox=truth, confidence=confidence, iou=iou, center_error_px=error,
                                 frames_since_seed=frame_no-self.last_seed[identity])
                    # Prediction returned above stays unchanged; reset affects next frame.
                    target.init(frame, truth)
                    self.frame_resets[identity] = event
                    self.reset_events.append(event)
                    self.last_seed[identity] = frame_no
                    self.bad_frames[identity] = 0
        return results


def summarize(rows):
    if not rows:
        return dict(samples=0, mean_iou=None, fraction_iou_ge_0_5=None, mean_center_error_px=None)
    values = np.asarray(rows)
    return dict(samples=len(rows), mean_iou=float(values[:, 0].mean()),
                fraction_iou_ge_0_5=float((values[:, 0] >= .5).mean()),
                mean_center_error_px=float(values[:, 1].mean()))


def run_clip(clip, video, annotation, engine, args, output):
    gt = load_annotations(annotation)
    sequence = SequenceTracker(engine, gt_reset=args.gt_reset, reset_iou=args.reset_iou,
                               reset_error_px=args.reset_error_px, reset_patience=args.reset_patience,
                               annotated_only=args.annotated_only)
    cap = VideoReader(video, prefetch=args.prefetch)
    writer = log = None
    window = "Purdue | Space pause | n next | g ground truth | q quit"
    trails = defaultdict(lambda: deque(maxlen=60))
    metrics = defaultdict(list)
    count, tracking_seconds = 0, 0.0
    status, quit_all = "complete", False
    show_gt = True
    reset_notice, reset_notice_until = '', 0
    loop_start = perf_counter()
    try:
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open {video}")
        fps = cap.get(cv2.CAP_PROP_FPS)
        if not math.isfinite(fps) or fps <= 0:
            raise ValueError(f"Invalid source FPS: {video}")
        expected = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if max(gt) > expected and expected > 0:
            raise ValueError(f"Annotations exceed {expected} frames in {video}")
        if args.save_tracks:
            log = (output / f"Clip_{clip}_tracking.jsonl").open("w", encoding="utf-8")
        if not args.headless:
            cv2.namedWindow(window, cv2.WINDOW_NORMAL)
        while True:
            start = perf_counter()
            ok, frame = cap.read()
            if not ok:
                if count == 0 or (expected > 0 and count < expected):
                    raise RuntimeError(f"Decode stopped at frame {count}/{expected} in {video}")
                break
            count += 1
            truth = gt.get(count, {})
            infer_start = perf_counter()
            predictions = sequence.step(frame, count, truth, 1/fps)
            tracking_ms = (perf_counter()-infer_start)*1000
            tracking_seconds += tracking_ms/1000
            frame_rows = []
            for identity, (success, box, confidence) in predictions.items():
                seeded = sequence.births[identity] == count
                measurement = None
                if identity in truth and not seeded:
                    measurement = box_metrics(box, truth[identity])
                    metrics[identity].append(measurement)
                trails[identity].append((round(box[0]+box[2]/2), round(box[1]+box[3]/2)))
                frame_rows.append(dict(id=identity, bbox=box, confidence=confidence, success=bool(success),
                                       gt_reset=sequence.frame_resets.get(identity),
                                       initialized=seeded, gt_bbox=truth.get(identity),
                                       iou=None if measurement is None else measurement[0],
                                       center_error_px=None if measurement is None else measurement[1]))
            if log:
                log.write(json.dumps(dict(frame=count, video_time_s=(count-1)/fps,
                                          tracking_ms=tracking_ms, targets=frame_rows)) + "\n")
            if not args.headless or args.save_video:
                display = frame.copy()
                for identity, (success, box, confidence) in predictions.items():
                    color = (0,0,255) if identity in sequence.frame_resets else COLORS[(identity-1) % len(COLORS)]
                    points = np.array(trails[identity], np.int32)
                    if len(points) > 1:
                        cv2.polylines(display, [points], False, color, 1, cv2.LINE_AA)
                    x,y,w,h = box
                    cv2.rectangle(display, (round(x),round(y)), (round(x+w),round(y+h)), color, 2)
                    label = f"ID {identity} " + ("INIT" if confidence is None else f"{confidence:.2f}")
                    if identity in sequence.frame_resets:
                        label += ' FAIL -> GT RESET'
                    cv2.putText(display, label, (max(0,round(x)), max(20,round(y)-7)), 0, .6, color, 2)
                if show_gt:
                    for identity, (x,y,w,h) in truth.items():
                        cv2.rectangle(display, (round(x),round(y)), (round(x+w),round(y+h)), (255,255,0), 1)
                if truth and not args.no_zoom:
                    identity = min(truth)
                    x,y,w,h = truth[identity]
                    # GT centers the visualization ONLY, never the inference crop.
                    side = min(120, frame.shape[0], frame.shape[1])
                    zx = max(0, min(frame.shape[1]-side, round(x+w/2-side/2)))
                    zy = max(0, min(frame.shape[0]-side, round(y+h/2-side/2)))
                    zoom = cv2.resize(display[zy:zy+side,zx:zx+side], (360,360), interpolation=cv2.INTER_NEAREST)
                    if display.shape[0] > 440 and display.shape[1] > 400:
                        display[-370:-10,-370:-10] = zoom
                        cv2.putText(display, f"GT-centered zoom: ID {identity}", (display.shape[1]-370,display.shape[0]-382),
                                    0, .65, (255,255,0), 2)
                text = f"Clip {clip} | frame {count}/{expected} | {len(predictions)} targets | {tracking_ms:.1f} ms | GT cyan"
                cv2.rectangle(display, (0,0), (min(display.shape[1], 1350),75), (0,0,0), -1)
                cv2.putText(display, text, (12,28), 0, .75, (255,255,255), 2)
                cv2.putText(display, f"Auto template {'ON' if args.auto_template else 'OFF'} | Fc {args.template_fc:g} Hz | Space pause, n next, g GT, q quit",
                            (12,60), 0, .65, (255,255,255), 1)
                if sequence.frame_resets:
                    reset_notice = f"GT RESET at frame {count}: IDs {', '.join(map(str, sequence.frame_resets))}"
                    reset_notice_until = count + round(fps)
                mode = f"GT recovery {'ON' if args.gt_reset else 'OFF'} | resets {len(sequence.reset_events)}"
                if args.annotated_only:
                    mode += ' | ANNOTATED TARGETS ONLY'
                if count <= reset_notice_until and reset_notice:
                    mode += ' | ' + reset_notice
                cv2.rectangle(display,(0,76),(min(display.shape[1],1500),110),(0,0,0),-1)
                cv2.putText(display,mode,(12,101),0,.65,(0,165,255) if args.gt_reset else (220,220,220),2)
                if args.save_video:
                    if writer is None:
                        writer = cv2.VideoWriter(str(output / f"Clip_{clip}_tracked.avi"),
                                                 cv2.VideoWriter_fourcc(*"MJPG"), fps, (frame.shape[1],frame.shape[0]))
                        if not writer.isOpened():
                            raise RuntimeError("Cannot open annotated video writer")
                    writer.write(display)
                if not args.headless:
                    ratio = min(1.0, args.display_width/display.shape[1])
                    shown = cv2.resize(display, None, fx=ratio, fy=ratio) if ratio < 1 else display
                    cv2.imshow(window, shown)
                    delay = 1 if args.fast else max(1, round(1000/fps-(perf_counter()-start)*1000))
                    key = cv2.waitKey(delay) & 255
                    while key == 32:
                        # Bounded event pumping keeps close/q/n working while paused.
                        while True:
                            key = cv2.waitKey(30) & 255
                            if key in (32, ord('q'), ord('n')) or cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1:
                                break
                        if key == 32:
                            key = -1
                    if key == ord('g'):
                        show_gt = not show_gt
                    if key == ord('q') or cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1:
                        status, quit_all = "interrupted", True
                        break
                    if key == ord('n'):
                        status = "skipped"
                        break
            for identity in sequence.frame_resets:
                trails[identity].clear()
            if args.max_frames and count >= args.max_frames:
                status = "limited"
                break
        all_rows = [row for values in metrics.values() for row in values]
        summary = dict(clip=clip, video=str(video), annotations=str(annotation), frames=count, status=status,
                       target_update_mode='annotated_only' if args.annotated_only else 'all_initialized',
                       loop_fps=count/max(perf_counter()-loop_start, 1e-9),
                       evaluation_mode='gt_assisted_recovery' if args.gt_reset else 'unassisted',
                       reset_count=len(sequence.reset_events), reset_events=sequence.reset_events,
                       initialization_frames=sequence.births, tracking_fps=count/max(tracking_seconds,1e-9),
                       **summarize(all_rows), per_target={identity: summarize(metrics[identity]) for identity in sequence.targets})
        (output / f"Clip_{clip}_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        iou = summary['mean_iou']
        print(f"Clip {clip}: {count} frames, {len(sequence.targets)} IDs, mean IoU {iou if iou is None else round(iou,3)}, {len(sequence.reset_events)} resets, {summary['tracking_fps']:.1f} tracking FPS, {summary['loop_fps']:.1f} loop FPS [{status}; {summary['target_update_mode']}]", flush=True)
        return summary, quit_all
    finally:
        cap.release()
        if writer is not None:
            writer.release()
        if log is not None:
            log.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--videos', type=Path, default=ROOT/'Videos')
    parser.add_argument('--annotations', type=Path, default=ROOT/'Video_Annotation-v2')
    parser.add_argument('--clips', '--video', default='all', help='clip number, comma list, range, or all (default)')
    parser.add_argument('--headless', action='store_true')
    parser.add_argument('--fast', action='store_true', help='unpaced display; headless is always unpaced')
    parser.add_argument('--display-width', type=int, default=1280)
    parser.add_argument('--no-zoom', action='store_true', help='hide the visualization-only ground-truth-centered inset')
    parser.add_argument('--max-frames', type=int, default=0, help='per-clip limit for smoke tests')
    parser.add_argument('--auto-template', action='store_true')
    parser.add_argument('--template-fc', type=float, default=2.0)
    parser.add_argument('--gt-reset', action='store_true', help='enable ground-truth-assisted failure recovery')
    parser.add_argument('--reset-iou', type=float, default=0.1, help='failure if IoU is below this (0 disables IoU trigger)')
    parser.add_argument('--reset-error-px', type=float, help='also fail if center error exceeds this many original-image pixels')
    parser.add_argument('--reset-patience', type=int, default=3, help='consecutive bad annotated frames before reset (default 3)')
    parser.add_argument('--threads', type=int, default=1)
    parser.add_argument('--cuda-graphs', action='store_true', help='capture/replay GPU inference; requires CUDA first')
    parser.add_argument('--prefetch', type=int, default=2, help='bounded file decode queue; 0 disables (default 2)')
    parser.add_argument('--annotated-only', action='store_true', help='GT-assisted scheduling: skip unannotated IDs; resume existing state on reappearance')
    parser.add_argument('--providers', nargs='+', default=['CPUExecutionProvider'])
    parser.add_argument('--save-video', action='store_true')
    parser.add_argument('--save-tracks', action='store_true')
    parser.add_argument('--output', type=Path, default=ROOT/'evaluation_output'/datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    args = parser.parse_args()
    if args.prefetch < 0:
        parser.error('prefetch must be nonnegative')
    if args.display_width <= 0 or args.max_frames < 0:
        parser.error('display-width must be positive; max-frames must be nonnegative')
    try:
        SequenceTracker(None, reset_iou=args.reset_iou, reset_error_px=args.reset_error_px,
                        reset_patience=args.reset_patience)
    except ValueError as exc:
        parser.error(str(exc))
    pairs = discover(args.videos, args.annotations)
    clips = clip_selection(args.clips, pairs)
    # Validate every selected annotation file before starting a potentially long run.
    for clip in clips:
        load_annotations(pairs[clip][1])
    engine = NanoTrackORT(ROOT/'models/nanotrack_backbone_sim.onnx', ROOT/'models/nanotrack_head_sim.onnx',
                          providers=args.providers, threads=args.threads,
                          auto_update_template=args.auto_template, template_fc_hz=args.template_fc,
                          cuda_graphs=args.cuda_graphs)
    args.output.mkdir(parents=True, exist_ok=False)
    config = {k: str(v) if isinstance(v,Path) else v for k,v in vars(args).items()}
    (args.output/'settings.json').write_text(json.dumps(config,indent=2),encoding='utf-8')
    summaries = []
    try:
        for clip in clips:
            summary, stop = run_clip(clip, *pairs[clip], engine, args, args.output)
            summaries.append(summary)
            (args.output/'summary.json').write_text(json.dumps(summaries,indent=2),encoding='utf-8')
            if stop:
                break
    finally:
        cv2.destroyAllWindows()
        print(f"Results: {args.output.resolve()}", flush=True)


if __name__ == '__main__':
    main()
