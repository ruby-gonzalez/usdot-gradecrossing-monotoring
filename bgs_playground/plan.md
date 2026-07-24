# Plan: merge extract_roi.py (ER) into pipeline.py (PL)

## Goal

Merge ER and PL into a single end-to-end command that derives the rail
track-corridor ROI, runs BGS + DeepSORT tracking, and raises an alarm when an
object dwells inside the ROI — producing one annotated video (detections + ROI
polygon + alarm state).

## Notes / decisions

- Reuse ER's functions (`load_zone_json`, `detect_zone_from_rail_model`,
  `draw_zone_overlay`), don't rewrite them.
- Keep ER's dual ROI source: `--zone-json` reuses a saved polygon (fast path),
  `--rail-model` derives a fresh one (loads the seg model). Corridor is static
  per clip, so derive once and reuse the JSON on later runs.
- torch is unavoidable regardless — DeepSORT's mobilenet embedder loads it.
  The model path only adds the rail-seg weights, not torch itself.
- State machine: per-object dwell dict (`track_id -> consecutive frames inside
  ROI`). Alarm if any confirmed track dwells >= threshold. Reset a track's
  counter the frame it leaves the ROI.
- Annotate in ONE pass: fold `draw_zone_overlay` into the `track_video` loop.
  Do NOT use `annotate_full_video` (it's a second full pass, ROI only).

- DO NOT WRITE CODE unless the user specifies to do so. (He wants to learn adn take more control of the process rather than relying on AI to do it)

## Alarm primitive (verified against installed deep-sort-realtime)

```python
left, top, right, bottom = track.to_ltrb()
foot = (float((left + right) / 2), float(bottom))   # bottom-center
inside = cv2.pointPolygonTest(zone.astype(np.int32), foot, False) >= 0
```

## Build order

- [DONE] **1. CLI first.** Add argparse to PL (`--video`, `--zone-json`,
      `--rail-model`, `--output`, `--min-area`, `--max-area`, `--dwell-frames`,
      `--display`). Just parse + `print(args)`. Kills the stale
      `/home/gaelmarquez/` paths.

```python
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video","-v", required=True,type=Path, help="Input video path")
    parser.add_argument("--zone-json","--zj" ,type=Path,help="Input JSON file path")
    parser.add_argument("--rail-model","-r",type=Path, help="Give seg model path")
    parser.add_argument("--output","-o",type=Path,help="Give path to output directory")
    parser.add_argument("--min-area",type=int,help="give minimum pixel area")
    parser.add_argument("--max-area",type=int,help="give maximum pixel area")
    parser.add_argument("--dwell-frames",type=int,help="give dwell frames")
    parser.add_argument("--display",action="store_true")

    args=parser.parse_args()
    return args

if __name__ == "__main__":
    print(parse_args())
#tested with python3 pipeline.py --video foo.mp4
```


- [DONE] **2. Get a polygon into PL.** Wire ER dual-source (`--zone-json` ->
      `load_zone_json`; `--rail-model` -> `detect_zone_from_rail_model`) into one
      `zone` numpy array. Print its shape.
```python
def poly_shape(args):
    if args.zone_json:
        zone = load_zone_json(args.zone_json)
        zone_metadata = {"source": "zone_json", "path": str(args.zone_json)}
        print(zone.shape)
    elif args.rail_model:
        zone, zone_metadata = detect_zone_from_rail_model(
            args.rail_model,
            args.video,
            args.rail_threshold,
            args.scan_step,
            args.scan_limit,
        )
        print(zone.shape)
    else:
        raise SystemExit("Provide either --rail-model or --zone-json.")

  #added to arguments
    parser.add_argument("--rail-threshold", type=float, default=0.35)
    parser.add_argument("--scan-step", type=int, default=30)
    parser.add_argument("--scan-limit", type=int, default=900)
```
#### Both approaches were tested with the following commands
``` bash
    python3 pipeline.py --video foo.mp4 --zj /path/to/json
    python3 pipeline.py --video /path/to/video --rail-model /path/to/seg_model
```



- [ ] **3. Overlay the ROI.** In the loop, `frame = draw_zone_overlay(frame,
      zone)` BEFORE `draw_tracks` (draw_zone_overlay returns a copy;
      draw_tracks mutates in place). Eyeball the polygon on the rails.



- [ ] **4. Dwell dict.** Create `dwell = {}` OUTSIDE the `while`. Each frame,
      for each confirmed track: foot-point test -> increment or reset
      `dwell[track_id]`. Prune IDs not seen this frame. `print(dwell)`.



- [ ] **5. Alarm from dwell.** `status = "ALARM" if any(v >= args.dwell_frames
      for v in dwell.values()) else "CLEAR"`. Overlay status text (match
      casing). Delete the old `alarm_state`.



- [ ] **6. Verify end-to-end.** Run on a clip where something sits in the
      corridor; confirm ALARM after ~dwell_frames and CLEAR when it leaves.




## Resources (grab each when its step comes up — NOT a separate sequence)

The build order above is the sequence you follow. These are just docs to open
when you reach the step that uses them:

- Step 1 (CLI): [argparse howto](https://docs.python.org/3/howto/argparse.html);
  copy the pattern from ER's `parse_args` (extract_roi.py:228-244).
- Step 2 (polygon): reuse `load_zone_json` (extract_roi.py:41-49) and
  `detect_zone_from_rail_model` (extract_roi.py:84-123) — don't rewrite them.
- Step 3 (overlay): reuse `draw_zone_overlay` (extract_roi.py:126-133).
- Step 4 (dwell dict): Track API in the installed package
  (.myv/.../deep_sort_realtime/deep_sort/track.py:109-277) —
  `is_confirmed()`, `track_id`, `to_ltrb()`; plus
  [cv2.pointPolygonTest](https://docs.opencv.org/4.x/d3/dc0/group__imgproc__shape.html#ga1a539e8db2135af2566103705d7a5722).
- Steps 5-6: no new docs — wiring + verification with what's above.

## ROI stabilization (camera shake) — edge case, not observed yet

Current clips are static; this was added defensively, not for a real shaky
clip. `ROIStabilizer` (Stabilization.py) warps only the ROI *polygon* to
follow camera motion, matching each frame against the reference frame the
polygon was derived from (rail model's `scan_frame`, else first frame) with
Lucas-Kanade. Reference-based => drift-free. A dead-zone guard collapses
sub-threshold motion (<1 px, <0.002 rad) to identity, so it is inert on static
footage. Wired into `annotate_full_video` via `stabilize_roi` (on by default,
`--no-stabilize-roi` to disable); frame pixels stay raw so BGS/DeepSORT see the
real image.

- TRADEOFF: this moves the polygon only. SuBSENSE still sees a shaking frame,
  so real shake would still produce bg-sub false foreground (repeatable
  phantom edge blobs that can survive `min_area` and get promoted by
  `n_init`). Polygon-warp does NOT fix that.
- TODO (only if a genuinely shaky clip appears): go hybrid — stabilize the
  FRAME before SuBSENSE, not just the ROI. Bigger job; deferred under YAGNI.

## Gotchas to watch

- Dwell dict lives OUTSIDE the frame loop, or it resets every frame.
- Reset counter on exit (consecutive dwell = "stopped/loitering" semantic).
- Prune dead track IDs (DeepSORT IDs grow unbounded over long video).
- Call the alarm logic AFTER `update_tracks`, on `tracks` (not `detections`).
- Keep status string + color check on the same casing.
- ROI polygon is in the source video's resolution; only aligns on a
  same-resolution clip.
